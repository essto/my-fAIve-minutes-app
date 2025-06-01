#!/bin/bash

# Konfiguracja
STORAGE_ACCOUNT=""
RESOURCE_GROUP=""
CONTAINER_NAME=""

# Pobierz connection string
echo "[INFO] Getting connection string..."
CONNECTION_STRING=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

if [ -z "$CONNECTION_STRING" ]; then
  echo "[ERROR] Failed to get connection string. Exiting."
  exit 1
fi

# Ustaw poziom dostępu kontenera na 'private'
echo "[INFO] Setting access level of $CONTAINER_NAME to private..."
az storage container set-permission \
  --name "$CONTAINER_NAME" \
  --connection-string "$CONNECTION_STRING" \
  --public-access off

# Weryfikacja
echo "[INFO] Verifying access level..."
ACCESS_LEVEL=$(az storage container show \
  --name "$CONTAINER_NAME" \
  --connection-string "$CONNECTION_STRING" \
  --query "properties.publicAccess" -o tsv)

if [ "$ACCESS_LEVEL" == "null" ] || [ -z "$ACCESS_LEVEL" ]; then
  echo "[SUCCESS] Container '$CONTAINER_NAME' is private (no public access)."
else
  echo "[FAIL] Container '$CONTAINER_NAME' is still public: $ACCESS_LEVEL"
  exit 2
fi
