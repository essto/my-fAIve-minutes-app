#!/bin/bash

# Parameters
STORAGE_ACCOUNT_NAME=""
CONTAINER_NAME=""
LOCATION=""

# Get the first available resource group
RESOURCE_GROUP=$(az group list --query "[0].name" -o tsv)
echo "Using resource group: $RESOURCE_GROUP"

# Create the storage account
az storage account create \
  --name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2

# Get the storage account key
ACCOUNT_KEY=$(az storage account keys list \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].value" -o tsv)

# Create a public blob container (anonymous read access for containers and blobs)
az storage container create \
  --name "$CONTAINER_NAME" \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$ACCOUNT_KEY" \
  --public-access container

echo "Storage account and public container created successfully."

# walidacja ("container")
az storage container show \
  --name $CONTAINER_NAME \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $ACCOUNT_KEY \
  --query "properties.publicAccess"
