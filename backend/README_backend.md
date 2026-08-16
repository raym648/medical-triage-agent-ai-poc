---
title: Medical Triage Agent AI Poc
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Medical Triage Agent — AI POC

Backend API de triage médical assisté par IA, déployé sur Hugging Face Spaces via Docker.

## Stack technique

- **FastAPI** (`app/main.py`) comme serveur applicatif
- **vLLM** (backend CPU) pour l'inférence du modèle de langage
- Conteneur basé sur `python:3.12-slim`

## Endpoints

- `GET /health` — endpoint de santé léger (ne charge pas le modèle)
- `POST /triage/...` — endpoint principal de triage (chargement paresseux du moteur vLLM au premier appel)

## Notes de déploiement

- Le moteur vLLM est initialisé en lazy loading : le premier appel de génération peut être plus lent (chargement du modèle).
- Variables d'environnement clés définies dans le `Dockerfile` : `VLLM_CPU_KVCACHE_SPACE`, `VLLM_CPU_OMP_THREADS_BIND`, `VLLM_TARGET_DEVICE`.
- Le support AVX512/AVX2 du CPU alloué par le tier Hugging Face n'est pas garanti ; voir les commentaires en bas du `Dockerfile` pour la commande de diagnostic.

## Avertissement

Ce projet est un POC à but de démonstration / recherche. Il ne doit pas être utilisé pour de véritables décisions médicales sans validation clinique et réglementaire appropriée.
