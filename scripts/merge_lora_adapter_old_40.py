# medical-triage-agent-ai-poc/backend/scripts/merge_lora_adapter.py

"""
Fusionne l'adaptateur LoRA final (policy post-DPO, dpo/final/ par
défaut — le meilleur checkpoint selon eval_loss, sauvegardé en fin de
run par train_dpo.py) dans le modèle de base
Qwen/Qwen3-1.7B-Base, puis publie le modèle fusionné complet
sur un nouveau repo Hugging Face.

CONTEXTE / CAUSE RACINE
------------------------
vLLM (mode `enable_lora=True` + LoRARequest) ne supporte PAS le
mécanisme PEFT `modules_to_save` :

    RuntimeError: Worker failed with error
    'vLLM only supports modules_to_save being None.'

Or l'adapter_config.json du checkpoint DPO contient :

    "modules_to_save": ["lm_head"]

... ce qui signifie que `lm_head` a été entièrement fine-tuné
(pas via LoRA A/B) pendant l'entraînement — très probablement
parce que des tokens spéciaux ont été ajoutés au tokenizer avant
le SFT/DPO (chat template, tokens de contrôle triage, etc.),
nécessitant un réentraînement complet des embeddings de sortie
pour ces nouveaux tokens.

SOLUTION
--------
Fusionner l'adaptateur (LoRA + modules_to_save) dans les poids du
modèle de base UNE FOIS, offline, puis servir directement ce
modèle fusionné avec vLLM — sans `enable_lora`, sans LoRARequest,
sans dépendance à PEFT au runtime. C'est la solution recommandée
par la documentation vLLM pour tout adapter incompatible avec le
chargement dynamique.

CORRECTIF TOKENIZER (2026-07-20)
---------------------------------
Certains checkpoints sauvegardés contiennent un tokenizer_config.json
avec `"extra_special_tokens": []` (liste vide) au lieu de `{}` (dict
vide). Les versions récentes de `transformers` font :

    self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())

... ce qui plante avec `AttributeError: 'list' object has no
attribute 'keys'` si `extra_special_tokens` est une liste.

Ce script télécharge et patche défensivement ce fichier (list -> dict)
avant tout appel à `AutoTokenizer.from_pretrained`, afin de rester
robuste même si d'autres checkpoints ont le même problème.

USAGE
-----
    # Vérification locale en pleine précision (float32, défaut) :
    python scripts/merge_lora_adapter.py \
        --output-dir ./qwen3-1.7b-triage-merged-fp32

    # Version à pousser/servir sur HF Space Free + vLLM-CPU
    # (bfloat16 : ~2x moins de RAM/disque, précision suffisante) :
    python scripts/merge_lora_adapter.py \
        --merge-dtype bfloat16 \
        --push-to-hub \
        --hub-repo-id RemDev-AI/medical-triage-agent-ai-poc-merged

NOTE DTYPE
----------
Le calcul du merge (application des deltas LoRA + modules_to_save
dans les poids de base) est TOUJOURS effectué en float32 en interne,
quel que soit --merge-dtype. Seule la conversion finale (juste avant
save_pretrained/push_to_hub) change de dtype. Ça évite d'introduire
des erreurs d'arrondi pendant les additions LoRA elles-mêmes tout en
permettant de livrer un modèle final compact (bfloat16/float16).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# NOTE (auth HF) : ce script ne chargeait auparavant aucun fichier
# .env et ne lisait la variable d'environnement QUE sous le nom exact
# "HF_TOKEN" (comportement par défaut de huggingface_hub). Si le
# token est stocké sous un autre nom (ex: HF_TOKEN_06) dans .env, ni
# python-dotenv ni huggingface_hub ne le trouveront automatiquement,
# et push_to_hub()/create_repo() échouent en 401 (requête non
# authentifiée). On charge .env explicitement puis on résout le
# token en essayant plusieurs noms de variable usuels.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    logger.warning(
        "python-dotenv n'est pas installé : le fichier .env ne sera "
        "PAS chargé automatiquement. `pip install python-dotenv` ou "
        "exporte tes variables (ex: HF_TOKEN) manuellement dans le "
        "shell avant de lancer ce script."
    )


def _resolve_hf_token(explicit_env_var: str | None) -> str | None:
    """Résout le token HF en essayant, dans l'ordre : le nom de
    variable explicitement fourni par --hf-token-env-var, puis les
    noms usuels HF_TOKEN et HUGGING_FACE_HUB_TOKEN. Retourne None si
    rien n'est trouvé (dans ce cas on retombe sur le comportement par
    défaut de huggingface_hub, ex: token mis en cache via `huggingface-
    cli login`)."""
    candidate_names = []
    if explicit_env_var:
        candidate_names.append(explicit_env_var)
    candidate_names += ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]

    for name in candidate_names:
        value = os.environ.get(name)
        if value:
            logger.info("Token HF trouvé via la variable d'environnement %s.", name)
            # On expose aussi le token sous le nom standard HF_TOKEN,
            # car certains appels internes de huggingface_hub (ex:
            # hf_hub_download plus haut dans ce script) ne lisent que
            # ce nom précis par défaut.
            os.environ.setdefault("HF_TOKEN", value)
            return value

    logger.warning(
        "Aucun token HF trouvé (variables essayées : %s). Les "
        "requêtes vers le Hub seront non authentifiées, et "
        "--push-to-hub échouera avec une 401 si le repo est privé ou "
        "n'existe pas encore.",
        candidate_names,
    )
    return None

# Doivent rester identiques à vllm_engine.py (_BASE_MODEL_NAME,
# runtime_config.model_repository, _ADAPTER_SUBFOLDER) pour fusionner
# exactement l'adapter actuellement servi en production.
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
ADAPTER_REPO_ID = "RemDev-AI/medical-triage-agent-ai-poc-models"

# FIX STRUCTURE-4 (audit renommage 2026-07-24) — chemin aligné sur le
# nouveau schéma {training_type}/checkpoints/... (cf.
# colab_checkpoint_sync.py). Ancien chemin : checkpoints/dpo/....
DPO_CHECKPOINTS_PREFIX = "dpo/checkpoints/"

# FIX STRUCTURE-5 (audit cohérence 2026-07-25) — pointe désormais sur
# dpo/final/ (le modèle sauvegardé en fin de run par train_dpo.py, avec
# load_best_model_at_end=True — donc le MEILLEUR checkpoint selon
# eval_loss), et non plus sur un checkpoint-dpo-<N> intermédiaire
# arbitraire sous dpo/checkpoints/. Un checkpoint intermédiaire n'est
# pas garanti être le meilleur modèle du run : "dernier sauvegardé"
# (numéro de step le plus élevé) ≠ "meilleur" dès que
# load_best_model_at_end=True restaure un checkpoint antérieur en fin
# d'entraînement. dpo/final/ est un chemin stable : plus besoin de
# résoudre dynamiquement un numéro de step.
DEFAULT_ADAPTER_SUBFOLDER = "dpo/final"

ADAPTER_SUBFOLDER: str | None = None


def resolve_latest_dpo_checkpoint(adapter_repo_id: str) -> str:
    """
    Repli de secours (utilisé seulement si --adapter-subfolder=intermediate
    est explicitement demandé) : retourne le sous-dossier du dernier
    checkpoint DPO INTERMÉDIAIRE disponible sur le Hub (ex:
    "dpo/checkpoints/checkpoint-dpo-47"). Ce n'est PAS le comportement
    par défaut — voir DEFAULT_ADAPTER_SUBFOLDER ci-dessus pour pourquoi
    "dernier" n'est pas la même chose que "meilleur"/"final".
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=adapter_repo_id, repo_type="model")

    checkpoint_dirs = set()
    for file in files:
        if not file.startswith(DPO_CHECKPOINTS_PREFIX):
            continue
        parts = file.split("/")
        if len(parts) >= 3:
            checkpoint_dirs.add(parts[2])

    if not checkpoint_dirs:
        raise RuntimeError(
            f"Aucun checkpoint DPO trouvé sous "
            f"{adapter_repo_id}/{DPO_CHECKPOINTS_PREFIX} — impossible de "
            "résoudre le dernier checkpoint intermédiaire."
        )

    latest_name = max(
        checkpoint_dirs, key=lambda name: int(name.split("-")[-1])
    )
    resolved = f"{DPO_CHECKPOINTS_PREFIX}{latest_name}"
    logger.info(
        "Dernier checkpoint DPO intermédiaire résolu : %s "
        "(mode --adapter-subfolder=intermediate explicitement demandé)",
        resolved,
    )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_NAME,
        help="Modèle de base Hugging Face (défaut : %(default)s).",
    )
    parser.add_argument(
        "--adapter-repo-id",
        default=ADAPTER_REPO_ID,
        help="Repo HF contenant l'adaptateur LoRA (défaut : %(default)s).",
    )
    parser.add_argument(
        "--adapter-subfolder",
        default=None,
        help="Sous-dossier de l'adaptateur à fusionner. Par défaut "
        f"(non fourni) : {DEFAULT_ADAPTER_SUBFOLDER!r} (le modèle final "
        "post-DPO, meilleur checkpoint selon eval_loss). Passer la "
        "valeur spéciale 'intermediate' pour résoudre automatiquement "
        "le dernier checkpoint-dpo-<N> intermédiaire sous "
        f"{DPO_CHECKPOINTS_PREFIX!r} à la place (débogage/reproduction "
        "d'un merge passé), ou un chemin explicite pour figer un "
        "checkpoint précis. NE PAS pointer vers 'ref/'.",
    )
    parser.add_argument(
        "--output-dir",
        default="./qwen3-1.7b-triage-merged",
        help="Dossier local de sortie du modèle fusionné.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Pousse le modèle fusionné sur le Hub HF après la fusion.",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="Repo HF cible pour le push "
        "(ex: RemDev-AI/medical-triage-agent-ai-poc-merged). "
        "Requis si --push-to-hub est utilisé.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Crée le repo HF cible en privé (défaut : True).",
    )
    parser.add_argument(
        "--merge-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help=(
            "Dtype du modèle FUSIONNÉ final (sauvegarde/push). Le calcul "
            "du merge (LoRA + modules_to_save) est TOUJOURS effectué en "
            "float32 en interne pour la précision, quel que soit ce "
            "choix ; seule la conversion finale change. "
            "'float32' (défaut) : précision max, ~7 Go sur disque pour "
            "un modèle 1.7B — utile pour vérifier que le merge est "
            "correct avant de pousser en production. "
            "'bfloat16' : recommandé pour servir sur un Hugging Face "
            "Space Free avec vLLM-CPU (~3.4 Go, empreinte mémoire/disque "
            "réduite de moitié, perte de précision négligeable pour un "
            "merge LoRA). "
            "'float16' : alternative à bfloat16, à éviter sur CPU pur "
            "(moins bien supporté nativement que bfloat16 pour "
            "l'inférence CPU)."
        ),
    )

    parser.add_argument(
        "--hf-token-env-var",
        default=None,
        help=(
            "Nom de la variable d'environnement (chargée depuis .env "
            "ou le shell) contenant le token HF, si celui-ci n'est "
            "PAS stocké sous le nom standard HF_TOKEN. Ex: si ton "
            ".env définit HF_TOKEN_06=hf_xxx, passe "
            "--hf-token-env-var HF_TOKEN_06. Sans cet argument, le "
            "script essaie HF_TOKEN puis HUGGING_FACE_HUB_TOKEN."
        ),
    )

    args = parser.parse_args()

    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id est requis avec --push-to-hub")

    return args


def patch_tokenizer_config_if_malformed(
    adapter_repo_id: str, adapter_subfolder: str
) -> None:
    """
    Certains tokenizer_config.json sauvegardés par des environnements
    d'entraînement plus anciens contiennent :

        "extra_special_tokens": []

    au lieu de :

        "extra_special_tokens": {}

    Les versions récentes de `transformers` appellent
    `special_tokens.keys()` sur ce champ, ce qui plante avec
    `AttributeError: 'list' object has no attribute 'keys'` si c'est
    une liste.

    Cette fonction télécharge le fichier via le cache HF local
    (hf_hub_download), corrige le champ s'il est malformé, et
    réécrit le fichier À L'EMPLACEMENT DU CACHE LOCAL (pas sur le
    Hub). `AutoTokenizer.from_pretrained` réutilisera ensuite ce
    fichier patché depuis le cache au lieu de le retélécharger.
    """
    from huggingface_hub import hf_hub_download

    logger.info(
        "Checking tokenizer_config.json for known malformed fields..."
    )

    try:
        cfg_path = hf_hub_download(
            repo_id=adapter_repo_id,
            subfolder=adapter_subfolder,
            filename="tokenizer_config.json",
        )
    except Exception:
        logger.warning(
            "Could not pre-fetch tokenizer_config.json for patching "
            "(will let AutoTokenizer handle it directly)."
        )
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    patched = False

    extra_special = cfg.get("extra_special_tokens")
    if isinstance(extra_special, list):
        if extra_special:
            # Liste non vide : ce format provient d'environnements
            # d'entraînement Qwen (2-VL/2.5-VL/3-VL) plus anciens qui
            # sérialisaient `extra_special_tokens` comme une simple liste
            # de chaînes au lieu du dict {nom_logique: token} attendu par
            # les versions récentes de `transformers`. La convention de
            # nommage Qwen pour ce champ consiste à retirer les
            # délimiteurs "<|" / "|>" du token pour obtenir le nom de la
            # clé (ex. "<|im_start|>" -> "im_start"), comme on peut le
            # vérifier sur des tokenizer_config.json Qwen3 corrigés de la
            # même façon sur le Hub (même liste de 13 tokens que celle
            # rencontrée ici : im_start/im_end, object_ref_start/end,
            # box_start/end, quad_start/end, vision_start/end/pad,
            # image_pad, video_pad). On applique donc cette conversion
            # automatique plutôt que d'exiger une correction manuelle.
            if not all(
                isinstance(tok, str) and tok.startswith("<|") and tok.endswith("|>")
                for tok in extra_special
            ):
                raise ValueError(
                    "tokenizer_config.json a un champ 'extra_special_tokens' "
                    f"non vide sous forme de liste ({extra_special!r}) dont "
                    "les éléments ne suivent pas le format Qwen attendu "
                    "('<|nom|>') : impossible de le convertir "
                    "automatiquement en dict. Corrige ce fichier "
                    "manuellement avant de continuer."
                )
            converted = {tok[2:-2]: tok for tok in extra_special}
            logger.warning(
                "Patching malformed 'extra_special_tokens' (list -> dict, "
                "%d tokens converti(s) via la convention Qwen "
                "'<|nom|>' -> {'nom': '<|nom|>'}) in cached "
                "tokenizer_config.json (%s).",
                len(converted),
                cfg_path,
            )
            cfg["extra_special_tokens"] = converted
            patched = True
        else:
            logger.warning(
                "Patching malformed 'extra_special_tokens' (empty list -> "
                "empty dict) in cached tokenizer_config.json (%s).",
                cfg_path,
            )
            cfg["extra_special_tokens"] = {}
            patched = True

    if patched:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info("tokenizer_config.json patched successfully.")
    else:
        logger.info("tokenizer_config.json looks fine, no patch needed.")


def get_adapter_lm_head_target_size(
    adapter_repo_id: str, adapter_subfolder: str
) -> int | None:
    """
    Détermine la taille de vocabulaire cible pour lm_head en lisant
    directement la shape du poids sauvegardé dans l'adaptateur
    (modules_to_save.default.weight), plutôt que de se fier à
    len(tokenizer).

    RAISON : de nombreux modèles (dont Qwen) ont un vocab_size dans
    la config strictement supérieur à len(tokenizer) (emplacements
    réservés/paddés pour l'alignement mémoire). Utiliser
    len(tokenizer) comme cible de resize_token_embeddings() est donc
    FAUX en général : ça peut RÉDUIRE lm_head en dessous de la
    taille avec laquelle l'adaptateur a été entraîné, provoquant un
    'size mismatch' au chargement de modules_to_save.

    Retourne None si le poids n'a pas pu être localisé (dans ce cas,
    l'appelant doit se rabattre sur un resize basé sur le tokenizer,
    avec les précautions d'usage).
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch as _torch  # noqa: F401  (safetensors a besoin de torch importé)

    candidate_filenames = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    for filename in candidate_filenames:
        try:
            weight_path = hf_hub_download(
                repo_id=adapter_repo_id,
                subfolder=adapter_subfolder,
                filename=filename,
            )
        except Exception:
            continue

        # NOTE (tie_word_embeddings) : quand le modèle de base a
        # tie_word_embeddings=True (cas de Qwen3), et que l'adapter a
        # été entraîné sans ensure_weight_tying=True côté PEFT, la
        # clé modules_to_save sauvegardée pour la cible "lm_head"
        # n'apparaît PAS forcément sous une clé contenant "lm_head" :
        # PEFT peut la ranger sous la clé du module d'embedding tied
        # (ex: "embed_tokens", "wte"). On ne filtre donc plus sur
        # "lm_head" : on cherche n'importe quelle clé
        # "modules_to_save...weight", en excluant les poids LoRA A/B.
        if filename.endswith(".safetensors"):
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    if (
                        "modules_to_save" in key
                        and key.endswith("weight")
                        and "lora_" not in key
                    ):
                        shape = f.get_slice(key).get_shape()
                        logger.info(
                            "Found modules_to_save target shape from "
                            "adapter weights: %s -> vocab_size=%d",
                            key,
                            shape[0],
                        )
                        return int(shape[0])
        else:
            state_dict = _torch.load(weight_path, map_location="cpu")
            for key, tensor in state_dict.items():
                if (
                    "modules_to_save" in key
                    and key.endswith("weight")
                    and "lora_" not in key
                ):
                    logger.info(
                        "Found modules_to_save target shape from "
                        "adapter weights: %s -> vocab_size=%d",
                        key,
                        tensor.shape[0],
                    )
                    return int(tensor.shape[0])

    logger.warning(
        "Could not locate lm_head modules_to_save weight in adapter "
        "files to determine exact target vocab size."
    )

    # Diagnostic : si vraiment aucune clé modules_to_save n'existe,
    # log la liste complète des clés du fichier trouvé pour lever le
    # doute (probable confirmation que PEFT n'a pas sauvegardé de
    # poids lm_head distinct pour ce checkpoint, cf. tie_word_embeddings
    # — voir docstring du module). Ne bloque jamais l'exécution.
    try:
        for filename in candidate_filenames:
            try:
                weight_path = hf_hub_download(
                    repo_id=adapter_repo_id,
                    subfolder=adapter_subfolder,
                    filename=filename,
                )
            except Exception:
                continue
            if filename.endswith(".safetensors"):
                with safe_open(weight_path, framework="pt") as f:
                    all_keys = list(f.keys())
            else:
                all_keys = list(_torch.load(weight_path, map_location="cpu").keys())
            logger.warning(
                "Diagnostic — clés présentes dans %s (%d au total) : %s",
                filename,
                len(all_keys),
                all_keys,
            )
            break
    except Exception as exc:
        logger.warning("Diagnostic des clés impossible : %s", exc)

    return None


def check_embed_tokens_trained(
    adapter_repo_id: str, adapter_subfolder: str
) -> bool:
    """
    Vérifie que les poids d'ENTRÉE (embed_tokens), et pas seulement
    lm_head (sortie), ont été inclus dans modules_to_save.

    CAUSE RACINE DIAGNOSTIQUÉE (2026-07-27) — génération illisible en
    prod ("salade" de tokens multi-scripts dès les premiers tokens,
    sur /generate/ ET /triage/, peu importe le gabarit de chat utilisé) :

    Le tokenizer a reçu de nouveaux tokens spéciaux (chat template :
    <|im_start|>, <|im_end|>, ...) avant le SFT/DPO, d'où le resize de
    embed_tokens ET lm_head. Mais modules_to_save=["lm_head"] (voir
    get_adapter_lm_head_target_size) ne fine-tune QUE la projection de
    SORTIE. La table d'embeddings d'ENTRÉE (embed_tokens) reste gelée
    sous LoRA, avec des lignes initialisées aléatoirement/par moyenne
    pour ces nouveaux tokens — jamais entraînées.

    Conséquence : le modèle sait très bien PRÉDIRE ces tokens en
    sortie (lm_head entraîné), mais ne sait pas les LIRE en entrée
    (embed_tokens jamais entraîné). Comme ces tokens apparaissent dans
    TOUT prompt basé sur un chat template (gabarit fait main ou vrai
    chat_template.jinja — même symptôme constaté avec les deux), le
    modèle "lit" du bruit dès les premiers tokens du prompt, et la
    génération diverge immédiatement vers des patterns de langues
    aléatoires appris ailleurs pendant le pré-entraînement.

    Ce n'est PAS réparable au merge : les poids embed_tokens n'ont
    jamais été appris nulle part, ce script ne peut que le CONSTATER,
    pas le corriger. Le vrai fix est côté training_model_loader.py /
    get_peft_model() : modules_to_save doit inclure "embed_tokens" en
    plus de "lm_head", suivi d'un ré-entraînement complet SFT -> DPO.

    Retourne True si embed_tokens semble avoir été sauvegardé/entraîné
    (clé modules_to_save correspondante trouvée), False sinon.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import torch as _torch  # noqa: F401

    candidate_filenames = ["adapter_model.safetensors", "adapter_model.bin"]

    for filename in candidate_filenames:
        try:
            weight_path = hf_hub_download(
                repo_id=adapter_repo_id,
                subfolder=adapter_subfolder,
                filename=filename,
            )
        except Exception:
            continue

        if filename.endswith(".safetensors"):
            with safe_open(weight_path, framework="pt") as f:
                all_keys = list(f.keys())
        else:
            all_keys = list(_torch.load(weight_path, map_location="cpu").keys())

        embed_keys = [
            key
            for key in all_keys
            if "modules_to_save" in key
            and "lora_" not in key
            and key.endswith("weight")
            and ("embed_tokens" in key or "wte" in key or "embed_in" in key)
        ]

        if embed_keys:
            logger.info(
                "OK : poids d'embeddings d'entrée trouvés dans "
                "modules_to_save (%s) — embed_tokens semble avoir été "
                "entraîné.",
                embed_keys,
            )
            return True

        logger.error(
            "AUCUNE clé embed_tokens/wte trouvée sous modules_to_save "
            "dans %s (clés modules_to_save présentes : %s). Cela "
            "signifie que la table d'embeddings D'ENTRÉE n'a JAMAIS "
            "été entraînée pour les tokens spéciaux ajoutés (chat "
            "template) — SEULE lm_head (sortie) l'a été. C'est la "
            "cause racine confirmée de la génération illisible "
            "observée en prod (salade multilingue dès les premiers "
            "tokens, sur /generate/ ET /triage/). Corrige "
            "modules_to_save=['lm_head', 'embed_tokens'] côté "
            "training_model_loader.py et RÉENTRAÎNE (SFT puis DPO) "
            "avant de refaire ce merge — ce script ne peut pas "
            "réparer des poids jamais appris.",
            filename,
            [k for k in all_keys if "modules_to_save" in k and "lora_" not in k],
        )
        return False

    logger.warning(
        "Impossible de localiser adapter_model.safetensors/.bin pour "
        "vérifier embed_tokens — vérification embed_tokens sautée."
    )
    return True  # inconnu : ne bloque pas, mais rien n'a pu être confirmé


def main() -> None:
    args = parse_args()

    hf_token = _resolve_hf_token(args.hf_token_env_var)

    # Imports différés : ce script tourne dans un environnement de
    # build/merge séparé (pas le container d'inférence CPU du Space),
    # torch/transformers/peft n'ont pas besoin d'être dans
    # requirements.txt du serveur.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if args.adapter_subfolder is None:
        args.adapter_subfolder = DEFAULT_ADAPTER_SUBFOLDER
        logger.info(
            "--adapter-subfolder non fourni : utilisation du modèle "
            "final post-DPO (%s).",
            args.adapter_subfolder,
        )
    elif args.adapter_subfolder == "intermediate":
        args.adapter_subfolder = resolve_latest_dpo_checkpoint(
            args.adapter_repo_id
        )
    else:
        logger.info(
            "Checkpoint figé explicitement via --adapter-subfolder=%s.",
            args.adapter_subfolder,
        )

    logger.info("Base model: %s", args.base_model)
    logger.info(
        "Adapter: %s (subfolder=%s)",
        args.adapter_repo_id,
        args.adapter_subfolder,
    )
    logger.info(
        "Output dtype: %s (merge computation itself always runs in "
        "float32 for precision; this only affects the final saved/"
        "pushed weights)",
        args.merge_dtype,
    )

    logger.info("Loading tokenizer from adapter (may include new/resized tokens)...")

    # Correctif défensif : patche tokenizer_config.json dans le cache
    # local si 'extra_special_tokens' est une liste au lieu d'un dict
    # (voir docstring de patch_tokenizer_config_if_malformed).
    patch_tokenizer_config_if_malformed(
        adapter_repo_id=args.adapter_repo_id,
        adapter_subfolder=args.adapter_subfolder,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_repo_id,
        subfolder=args.adapter_subfolder,
    )

    logger.info("Loading base model weights (fp32 for a clean merge)...")

    # CORRECTIF (2026-07-23) — cohérence avec training_model_loader.py :
    # suite au diagnostic confirmé par diagnose_lm_head_merge.py (le
    # lm_head du modèle publié était quasi-identique à celui du modèle
    # de base non fine-tuné), l'entraînement (SFT/DPO) charge désormais
    # le modèle de base avec tie_word_embeddings=False, AVANT
    # get_peft_model(), pour que modules_to_save=["lm_head"] produise
    # un poids réellement indépendant et sauvegardé.
    #
    # Ce script de merge doit charger le modèle de base de la MÊME
    # manière : sans ce untie ici aussi, on chargerait une architecture
    # tied pour y appliquer un adapter conçu/entraîné sur une
    # architecture untied — décalage d'architecture entre entraînement
    # et merge, source de comportements PEFT incertains même si le
    # poids se charge techniquement. Tout futur checkpoint (entraîné
    # avec le correctif) doit être fusionné avec ce même réglage.
    #
    # Les checkpoints ANTÉRIEURS au correctif (dont checkpoint-dpo-32,
    # confirmé sans poids lm_head distinct) restent fusionnables avec ce
    # script quel que soit ce réglage, puisqu'ils n'ont de toute façon
    # aucun poids modules_to_save réel à charger (cf.
    # get_adapter_lm_head_target_size -> None pour ce checkpoint).
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        tie_word_embeddings=False,
    )

    # IMPORTANT : la cible de resize ne doit PAS être déterminée par
    # len(tokenizer). De nombreux modèles (dont Qwen) ont un
    # vocab_size de config strictement supérieur à len(tokenizer)
    # (emplacements réservés/paddés). Se fier à len(tokenizer) peut
    # RÉDUIRE lm_head en dessous de la taille avec laquelle
    # l'adaptateur a été entraîné (modules_to_save=["lm_head"]),
    # ce qui provoque un 'size mismatch' lors du chargement PEFT.
    #
    # On lit donc directement la shape du poids lm_head sauvegardé
    # dans l'adaptateur : c'est la seule source de vérité fiable.
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]

    adapter_target_size = get_adapter_lm_head_target_size(
        adapter_repo_id=args.adapter_repo_id,
        adapter_subfolder=args.adapter_subfolder,
    )

    if adapter_target_size is not None:
        if adapter_target_size != base_vocab_size:
            logger.info(
                "Resizing token embeddings to match adapter's saved "
                "lm_head: base=%d -> adapter=%d",
                base_vocab_size,
                adapter_target_size,
            )
            base_model.resize_token_embeddings(adapter_target_size)
        else:
            logger.info(
                "Base model vocab size (%d) already matches adapter's "
                "saved lm_head size, no resize needed.",
                base_vocab_size,
            )
    else:
        # Repli : impossible de lire la shape réelle du poids.
        # On NE resize PAS automatiquement vers len(tokenizer), car
        # cela peut réduire lm_head à tort (cf. explication ci-dessus).
        # On avertit plutôt et on laisse PeftModel.from_pretrained
        # échouer avec un message clair si les tailles ne matchent
        # vraiment pas.
        tokenizer_vocab_size = len(tokenizer)
        logger.warning(
            "Skipping automatic resize (could not confirm target size "
            "from adapter weights). base=%d, len(tokenizer)=%d. "
            "If loading the adapter fails with a size mismatch, "
            "inspect adapter_model.safetensors manually to find the "
            "true target lm_head size.",
            base_vocab_size,
            tokenizer_vocab_size,
        )

    embed_tokens_trained = check_embed_tokens_trained(
        adapter_repo_id=args.adapter_repo_id,
        adapter_subfolder=args.adapter_subfolder,
    )

    if not embed_tokens_trained and args.push_to_hub:
        raise RuntimeError(
            "--push-to-hub annulé par sécurité : la table d'embeddings "
            "d'entrée (embed_tokens) n'a jamais été entraînée pour les "
            "tokens spéciaux ajoutés (voir logs ERROR ci-dessus). "
            "Publier ce merge reproduirait le bug de génération "
            "illisible déjà observé en prod. Corrige "
            "modules_to_save=['lm_head', 'embed_tokens'] côté "
            "training_model_loader.py, réentraîne, puis relance ce "
            "script. Retire temporairement --push-to-hub si tu veux "
            "seulement inspecter un merge local à des fins de "
            "diagnostic."
        )

    logger.info("Loading PEFT adapter (LoRA + modules_to_save=['lm_head'])...")
    # ensure_weight_tying=True : filet de sécurité pour d'éventuels
    # anciens checkpoints entraînés AVANT le correctif
    # tie_word_embeddings=False côté training_model_loader.py (dont
    # checkpoint-dpo-32). Pour tout nouveau checkpoint entraîné avec le
    # correctif, base_model est déjà untied ci-dessus : ce paramètre ne
    # devrait plus avoir d'effet significatif, mais reste sans danger à
    # laisser actif pour la rétrocompatibilité. Repli sur l'appel sans
    # l'argument si la version de PEFT installée est trop ancienne pour
    # le supporter.
    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.adapter_repo_id,
            subfolder=args.adapter_subfolder,
            ensure_weight_tying=True,
        )
    except TypeError:
        logger.warning(
            "La version de PEFT installée ne supporte pas "
            "ensure_weight_tying (nécessite peft récent, cf. "
            "https://github.com/huggingface/peft/issues/2777). "
            "Repli sans ce paramètre : le vocab_size final sera "
            "vérifié par le sanity check en fin de script, mais "
            "n'est PAS garanti correct. Envisage `pip install -U "
            "peft` si l'assertion finale échoue à nouveau."
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.adapter_repo_id,
            subfolder=args.adapter_subfolder,
        )

    logger.info(
        "Merging LoRA deltas and modules_to_save into base weights "
        "(merge_and_unload)..."
    )
    merged_model = peft_model.merge_and_unload()

    target_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.merge_dtype]

    if target_dtype != torch.float32:
        logger.info(
            "Casting merged model from float32 to %s before saving...",
            args.merge_dtype,
        )
        merged_model = merged_model.to(dtype=target_dtype)
    else:
        logger.info("Keeping merged model in float32 (no cast needed).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to %s ...", output_dir)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Sanity check minimal avant tout push : recharger le modèle
    # fusionné à froid et vérifier qu'il n'a plus de dépendance PEFT
    # (plus de fichier adapter_config.json), et que la taille du
    # vocabulaire est bien cohérente avec le tokenizer sauvegardé.
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(output_dir), torch_dtype=target_dtype
    )

    # Contournement d'un bug connu de certaines versions de
    # `transformers` : AutoTokenizer.from_pretrained() peut planter
    # avec `AttributeError: 'dict' object has no attribute
    # 'model_type'` sur un rechargement local, car un chemin interne
    # récupère parfois le config.json brut (dict) au lieu de
    # l'objet PretrainedConfig attendu. Ce n'est pas lié au merge
    # lui-même (les poids sont déjà sauvegardés à ce stade) — on
    # tente donc un repli avant d'abandonner la vérification.
    try:
        reloaded_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    except AttributeError as exc:
        logger.warning(
            "AutoTokenizer.from_pretrained a échoué avec un bug connu "
            "de transformers (%s). Nouvelle tentative avec "
            "use_fast=False...",
            exc,
        )
        try:
            reloaded_tokenizer = AutoTokenizer.from_pretrained(
                str(output_dir), use_fast=False
            )
        except Exception as exc2:
            logger.error(
                "Impossible de recharger le tokenizer pour le sanity "
                "check (bug transformers). LE MERGE EST DÉJÀ SAUVEGARDÉ "
                "sur %s et n'est PAS affecté par ce bug — seule la "
                "vérification automatique échoue. "
                "Vérifie manuellement : "
                "`python -c \"from transformers import AutoTokenizer; "
                "t=AutoTokenizer.from_pretrained('%s'); print(len(t))\"` "
                "et compare à la taille de vocab du modèle. "
                "Si le problème persiste, essaie de mettre à jour ou "
                "figer la version de `transformers` (pip install "
                "-U transformers ou pip install "
                "'transformers==<version connue stable>').",
                exc2,
                output_dir,
            )
            logger.info(
                "Done (sanity check du tokenizer sauté). Merged model "
                "available at %s", output_dir
            )
            if args.push_to_hub:
                logger.warning(
                    "--push-to-hub demandé mais sanity check incomplet : "
                    "abandon du push par sécurité. Relance avec "
                    "--push-to-hub une fois la vérification manuelle "
                    "effectuée, ou corrige l'environnement transformers."
                )
            return

    _reloaded_vocab_size = reloaded.get_input_embeddings().weight.shape[0]
    _reloaded_tokenizer_len = len(reloaded_tokenizer)

    # NOTE IMPORTANTE (2026-07-21) : NE PAS vérifier une ÉGALITÉ stricte
    # entre vocab_size et len(tokenizer). Les modèles Qwen (dont
    # Qwen3-1.7B-Base) ont volontairement un vocab_size de config
    # supérieur à len(tokenizer) — emplacements réservés/paddés pour
    # l'alignement mémoire et les extensions futures. C'est vrai même
    # sur le modèle de base SANS aucun adapter (ex: base=151936 vs
    # tokenizer=151669 pour Qwen3-1.7B-Base). Une égalité stricte ici
    # est donc un faux négatif garanti pour cette famille de modèles ;
    # l'invariant qui compte réellement est que la table d'embeddings
    # soit AU MOINS aussi grande que le nombre de tokens du tokenizer
    # (sinon certains ids de tokens seraient hors limites -> crash à
    # l'inférence).
    assert _reloaded_vocab_size >= _reloaded_tokenizer_len, (
        "La table d'embeddings du modèle fusionné est plus petite que "
        f"le tokenizer — ne pas déployer. model.vocab_size="
        f"{_reloaded_vocab_size}, len(tokenizer)={_reloaded_tokenizer_len}. "
        "Certains ids de tokens seraient hors limites à l'inférence."
    )

    # Vérification complémentaire : si aucune resize n'a été appliquée
    # (adapter_target_size introuvable, cf. get_adapter_lm_head_target_size),
    # le vocab_size du modèle fusionné doit rester IDENTIQUE à celui du
    # modèle de base d'origine. S'il a changé sans qu'on l'ait demandé,
    # c'est le signe d'une corruption réelle du merge (à ne pas confondre
    # avec l'écart normal vocab_size/tokenizer ci-dessus).
    if adapter_target_size is None and _reloaded_vocab_size != base_vocab_size:
        raise AssertionError(
            "Le vocab_size du modèle fusionné a changé de manière "
            f"inattendue (base={base_vocab_size} -> "
            f"mergé={_reloaded_vocab_size}) alors qu'aucun resize n'a "
            "été demandé — ne pas déployer, le merge est probablement "
            "corrompu."
        )

    assert not (output_dir / "adapter_config.json").exists(), (
        "adapter_config.json présent dans la sortie : le merge n'a "
        "pas correctement 'dé-PEFTisé' le modèle."
    )

    logger.info("Sanity check OK: merged model reloads cleanly, no PEFT artifacts.")

    if args.push_to_hub:
        if not hf_token:
            raise RuntimeError(
                "--push-to-hub demandé mais aucun token HF trouvé. "
                "Ajoute HF_TOKEN=hf_xxx à ton .env (ou passe "
                "--hf-token-env-var NOM_DE_TA_VARIABLE si le token "
                "est stocké sous un autre nom, ex: HF_TOKEN_06)."
            )
        logger.info("Pushing merged model to %s ...", args.hub_repo_id)
        merged_model.push_to_hub(
            args.hub_repo_id,
            private=args.private,
            token=hf_token,
        )
        tokenizer.push_to_hub(
            args.hub_repo_id,
            private=args.private,
            token=hf_token,
        )
        logger.info(
            "Done. Update vllm_engine.py _BASE_MODEL_NAME to '%s' "
            "and remove all LoRA-loading logic.",
            args.hub_repo_id,
        )
    else:
        logger.info(
            "Done (local only). Merged model available at %s", output_dir
        )


if __name__ == "__main__":
    sys.exit(main())
