#!/bin/bash

az group create --name vm-az-01_group --location northeurope
az deployment group create \
  --resource-group vm-az-01_group \
  --template-file ./vm-az-01//template.json \
  --parameters ./vm-az-01/parameters.json

az group create --name vm-az-02_group --location northeurope
az deployment group create \
  --resource-group vm-az-02_group \
  --template-file ./vm-az-02/template.json \
  --parameters ./vm-az-02/parameters.json

# peering-01
az deployment group create \
  --resource-group vm-az-01_group \
  --template-file ./peering-01/template.json \
  --parameters ./peering-01/parameters.json

# peering-02
az deployment group create \
  --resource-group vm-az-02_group \
  --template-file ./peering-02/template.json \
  --parameters ./peering-02/parameters.json
