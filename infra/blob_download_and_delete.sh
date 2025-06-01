#!/bin/bash

# Zmienna z nazwą Storage Account i kontenera
STORAGE_ACCOUNT=""
RESOURCE_GROUP=""
CONTAINER_NAME=""
DOWNLOAD_DIR=""

# Pobranie connection string
echo "[INFO] Fetching connection string..."
CONNECTION_STRING=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

# Stworzenie katalogu, jeśli nie istnieje
mkdir -p "$DOWNLOAD_DIR"

# Pobranie listy blobów i ściągnięcie każdego do /opt
echo "[INFO] Downloading blobs from $CONTAINER_NAME..."
blob_list=$(az storage blob list \
  --container-name "$CONTAINER_NAME" \
  --connection-string "$CONNECTION_STRING" \
  --query "[].name" -o tsv)

for blob in $blob_list; do
  echo "[INFO] Downloading blob: $blob"
  az storage blob download \
    --container-name "$CONTAINER_NAME" \
    --name "$blob" \
    --file "$DOWNLOAD_DIR/$blob" \
    --connection-string "$CONNECTION_STRING"
done

# Usunięcie kontenera
echo "[INFO] Deleting blob container: $CONTAINER_NAME..."
az storage container delete \
  --name "$CONTAINER_NAME" \
  --connection-string "$CONNECTION_STRING"

echo "[DONE] All blobs downloaded to $DOWNLOAD_DIR and container deleted."
