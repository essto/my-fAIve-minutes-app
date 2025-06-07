#!/bin/bash

# Zmienna z grupą zasobów
RESOURCE_GROUP=""
VM_NAME="devops-vm"
LOCATION="westus"
ADMIN_USER="azureuser"
SSH_KEY_PATH="$HOME/.ssh/devops-key.pub"

# Nazwy zasobów sieciowych
VNET_NAME="devops-vnet"
SUBNET_NAME="devops-subnet"
IP_NAME="devops-pip"
NIC_NAME="devops-nic"

echo "⛔ Usuwanie starej VM (jeśli istnieje)..."
az vm delete --name $VM_NAME --resource-group $RESOURCE_GROUP --yes --no-wait

echo "🧹 Usuwanie starego dysku (jeśli istnieje)..."
DISK_NAME=$(az disk list --resource-group $RESOURCE_GROUP --query "[?contains(name, '$VM_NAME')].name" -o tsv)
if [ ! -z "$DISK_NAME" ]; then
  az disk delete --name $DISK_NAME --resource-group $RESOURCE_GROUP --yes --no-wait
fi

echo "🌐 Tworzenie VNET i Subnet..."
az network vnet create \
  --resource-group $RESOURCE_GROUP \
  --name $VNET_NAME \
  --subnet-name $SUBNET_NAME \
  --location $LOCATION

echo "🌍 Tworzenie Static Public IP..."
az network public-ip create \
  --resource-group $RESOURCE_GROUP \
  --name $IP_NAME \
  --allocation-method Static \
  --sku Basic \
  --location $LOCATION

echo "🔗 Tworzenie NIC..."
az network nic create \
  --resource-group $RESOURCE_GROUP \
  --name $NIC_NAME \
  --vnet-name $VNET_NAME \
  --subnet $SUBNET_NAME \
  --public-ip-address $IP_NAME \
  --location $LOCATION

echo "💻 Tworzenie nowej VM..."
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --nics $NIC_NAME \
  --image Ubuntu2204 \
  --size Standard_B1s \
  --storage-sku Standard_LRS \
  --admin-username $ADMIN_USER \
  --ssh-key-values $SSH_KEY_PATH \
  --location $LOCATION \
  --no-wait

echo "✅ Gotowe. VM '$VM_NAME' tworzona asynchronicznie."
