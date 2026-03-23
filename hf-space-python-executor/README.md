---
title: KGoT Python Executor
emoji: 🐍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

Remote Python executor for Knowledge Graph of Thoughts.

Exposes:
- `GET /` basic service info
- `GET /health` health check
- `POST /run` code execution endpoint compatible with the KGoT Python tool
