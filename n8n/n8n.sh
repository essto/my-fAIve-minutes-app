#!/bin/bash

docker run -d --name n8n --rm -p 35678:5678 -v ~/.n8n:/home/node/.n8n -e N8N_SECURE_COOKIE=false n8nio/n8n:latest
