#!/bin/bash
docker login -u cibiai -p Bz/ICRzZlL0rdIPWwWdg0ZecTeYrwefj+1CwtS0IOO+ACRBNvMaR cibiai.azurecr.io
docker build --platform linux/amd64 -t cibiai.azurecr.io/platform-llm:latest .
docker push cibiai.azurecr.io/platform-llm:latest
docker build --platform linux/amd64 -t cibiai.azurecr.io/llm:realtime -f Dockerfile_rti .
docker push cibiai.azurecr.io/llm:realtime
docker build --platform linux/amd64 -t cibiai.azurecr.io/llm:batch -f Dockerfile_batch .
docker push cibiai.azurecr.io/llm:batch
