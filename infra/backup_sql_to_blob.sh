#!/bin/bash

# jeśli nie działa poniższy skrypt, zadziała przez GUI i private link (+firewall "all network, +v "Allow All Azure)
# https://learn.microsoft.com/en-us/azure/azure-sql/database/database-import-export-private-link?view=azuresql#limitations

# Konfiguracja
RESOURCE_GROUP=""
SQL_SERVER=""
SQL_DB=""
SQL_ADMIN=""
SQL_PASSWORD=""
STORAGE_ACCOUNT=""
CONTAINER_NAME=""
BLOB_NAME=""
DOWNLOAD_PATH="/opt/${BLOB_NAME}"

# Pobierz klucz storage
echo "[INFO] Retrieving storage account key..."
STORAGE_KEY=$(az storage account keys list \
  --resource-group "$RESOURCE_GROUP" \
  --account-name "$STORAGE_ACCOUNT" \
  --query '[0].value' -o tsv)

# Przygotuj pełny URI do blob
STORAGE_URI="https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}/${BLOB_NAME}"

# Eksportuj bazę danych do kontenera blob
echo "[INFO] Starting SQL DB export to Blob Storage..."
EXPORT_OPERATION=$(az sql db export \
  --admin-user "$SQL_ADMIN" \
  --admin-password "$SQL_PASSWORD" \
  --name "$SQL_DB" \
  --server "$SQL_SERVER" \
  --storage-key-type StorageAccessKey \
  --storage-key "$STORAGE_KEY" \
  --storage-uri "$STORAGE_URI" \
  --resource-group "$RESOURCE_GROUP" \
  --no-wait)

# Poczekaj na zakończenie eksportu
echo "[INFO] Waiting for export to complete..."
sleep 10
while true; do
  STATUS=$(az sql db export status \
    --admin-user "$SQL_ADMIN" \
    --admin-password "$SQL_PASSWORD" \
    --name "$SQL_DB" \
    --server "$SQL_SERVER" \
    --resource-group "$RESOURCE_GROUP" \
    --query "status" -o tsv)
  
  echo "[INFO] Export status: $STATUS"
  
  if [[ "$STATUS" == "Succeeded" ]]; then
    echo "[SUCCESS] Export completed successfully."
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    echo "[ERROR] Export failed. Exiting."
    exit 1
  fi
  sleep 10
done

# Sprawdź, czy plik istnieje w kontenerze
echo "[INFO] Verifying blob exists in container..."
az storage blob show \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$CONTAINER_NAME" \
  --name "$BLOB_NAME" \
  --account-key "$STORAGE_KEY" \
  --output table || {
    echo "[ERROR] Blob not found."
    exit 1
}

# Pobierz plik na lokalnego hosta
echo "[INFO] Downloading blob to $DOWNLOAD_PATH..."
az storage blob download \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$CONTAINER_NAME" \
  --name "$BLOB_NAME" \
  --file "$DOWNLOAD_PATH" \
  --account-key "$STORAGE_KEY" \
  --output none

# Weryfikacja lokalna
echo "[INFO] Verifying local file..."
if [[ -f "$DOWNLOAD_PATH" ]]; then
  echo "[SUCCESS] File downloaded: $DOWNLOAD_PATH"
  ls -lh "$DOWNLOAD_PATH"
else
  echo "[ERROR] File was not downloaded."
  exit 1
fi
