
# Create Azure Storage Account and Public Blob Container

This script performs the following actions:
- Retrieves the first available resource group using `az group list`.
- Creates a new Azure Storage Account.
- Creates a public Blob container with anonymous read access enabled.

## Script

```bash
#!/bin/bash

# Parameters
STORAGE_ACCOUNT_NAME="datacenterst30930"
CONTAINER_NAME="datacenter-blob-20088"
LOCATION="westus"

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
```
