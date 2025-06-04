# Parametry
RESOURCE_GROUP=""
VM_NAME=""
LOCATION=""
IMAGE=""
SIZE=""
ADMIN_USER=""
OS_DISK_SIZE=

# Tworzenie maszyny wirtualnej
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --image "$IMAGE" \
  --size "$SIZE" \
  --admin-username "$ADMIN_USER" \
  --generate-ssh-keys \
  --storage-sku Standard_LRS \
  --os-disk-size-gb "$OS_DISK_SIZE" \
  --location "$LOCATION"

# Weryfikacja statusu maszyny
az vm get-instance-view \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --query "instanceView.statuses[?starts_with(code,'PowerState/')].displayStatus" \
  -o tsv
