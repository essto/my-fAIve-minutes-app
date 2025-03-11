#!/bin/bahs

docker run -d -p 3000:8080 --network host -e OLLAMA_API_BASE_URL=http://localhost:11434 -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
