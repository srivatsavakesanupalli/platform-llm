#!/bin/bash
docker login -u cibiai -p Bz/ICRzZlL0rdIPWwWdg0ZecTeYrwefj+1CwtS0IOO+ACRBNvMaR cibiai.azurecr.io
docker build --platform linux/amd64 -t cibiai.azurecr.io/platform-cv:latest .
docker push cibiai.azurecr.io/platform-cv:latest
docker build --platform linux/amd64 -t cibiai.azurecr.io/cv:realtime -f Dockerfile_rti .
docker push cibiai.azurecr.io/cv:realtime
docker build --platform linux/amd64 -t cibiai.azurecr.io/cv:batch -f Dockerfile_batch .
docker push cibiai.azurecr.io/cv:batch