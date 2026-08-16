---
title: Medical Triage Agent AI Poc
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Medical Triage Agent — AI POC (Frontend UI)

Interface utilisateur (Streamlit) du POC d'agent IA de triage médical, déployée sur Hugging Face Spaces via Docker.

## Stack technique

- **Streamlit** comme serveur d'interface (`streamlit_app/app.py`)
- Conteneur basé sur `python:3.12-slim`
- Exécution en utilisateur non-root (`appuser`, UID 1000)

## Notes de déploiement

- Fichier source local : `Dockerfile.hf` (VSCode), renommé en `Dockerfile` avant déploiement sur ce Space.
- Le port Streamlit est forcé à `7860` (au lieu de `8501` en local) pour respecter la convention Hugging Face Spaces.
- Ce Space frontend appelle le backend de triage (agent IA / vLLM) déployé séparément.

## Avertissement

Ce projet est un POC à but de démonstration / recherche. Il ne doit pas être utilisé pour de véritables décisions médicales sans validation clinique et réglementaire appropriée.

