#!/bin/bash

# Konfiguracja
STORAGE_ACCOUNT=""
RESOURCE_GROUP=""
CONTAINER_NAME=""
LOCAL_FILE=""
BLOB_NAME=""

# Pobierz connection string
echo "[INFO] Fetching connection string..."
CONNECTION_STRING=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

# Prześlij plik
echo "[INFO] Uploading $LOCAL_FILE to $CONTAINER_NAME..."
az storage blob upload \
  --container-name "$CONTAINER_NAME" \
  --file "$LOCAL_FILE" \
  --name "$BLOB_NAME" \
  --connection-string "$CONNECTION_STRING"

echo "[DONE] Upload completed."
