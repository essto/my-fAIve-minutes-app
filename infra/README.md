# Deploy vm-az-01

## Using PowerShell

```powershell
New-AzResourceGroupDeployment -ResourceGroupName vm-az-01_group -TemplateFile ./template.json -TemplateParameterFile ./parameters.json
```

## Using Azure CLI

```bash
az deployment group create \
  --resource-group vm-az-01_group \
  --template-file ./template.json \
  --parameters ./parameters.json
```

```bash
az vm list -o table
az group list -o table
az vm delete --name <vm-name> --resource-group <resource-grup-name>
```

```bash
az vm list -o table
az vm update --name <vm-name> --resource-group <resource-grup-name> --set tags.<key=value>
vm show --name <vm-name> --resource-group <resource-grup-name> --query "<tag>"
```

```bash
az vm list -o table
az disk list -o table
az vm disk attach --vm-name <vm-name> --resource-group <resource-grup-name> --name <data-disk-name>
az vm show --name <vm-name> --resource-group <resource-grup-name> --query "storageProfile.dataDisks" -o table
```

```bash
az group list -o table
az vm list -o table
az network nic list -o table
az vm deallocate --name <vm-name> --resource-group <resource-grup-name>
az vm nic add --vm-name <vm-name> --resource-group <resource-grup-name> --nics <nic-name>
az vm start --name <vm-name> --resource-group <resource-grup-name>
az network nic show --name <nic-name> --resource-group <resource-grup-name> --query "virtualMachine"
```

## Assigning a Public IP to an Azure Virtual Machine

### Step 1: Identify the Resource Group

```bash
az vm list --query "[?name=='<VM_NAME>'].[name,resourceGroup]" -o table
```

> Retrieves the resource group name associated with `<VM_NAME>`.

---

### Step 2: Get the Network Interface (NIC) Attached to the VM

```bash
az vm show --name <VM_NAME> --resource-group <RESOURCE_GROUP> \
  --query "networkProfile.networkInterfaces[0].id" -o tsv
```

> Outputs the NIC ID of the virtual machine.

---

### Step 3: List All IP Configurations on the NIC

```bash
az network nic show --name <NIC_NAME> --resource-group <RESOURCE_GROUP> \
  --query "ipConfigurations[].name" -o table
```

> Lists all IP configuration names for the specified NIC.

---

### Step 4: Assign a Public IP Address to the Primary IP Configuration

```bash
az network nic ip-config update \
  --nic-name <NIC_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --name <IPCONFIG_NAME> \
  --public-ip-address <PUBLIC_IP_NAME>
```

> Attaches an existing public IP resource to the specified IP configuration of the NIC.

---

### Step 5: Verify the Public IP Assignment

```bash
az network nic show --name <NIC_NAME> --resource-group <RESOURCE_GROUP> \
  --query "ipConfigurations[].publicIpAddress.id"
```

> Confirms that the public IP is assigned to the IP configuration.

---

### Placeholders Reference

| Placeholder        | Description                        |
| ------------------ | ---------------------------------- |
| `<VM_NAME>`        | Name of the Virtual Machine        |
| `<RESOURCE_GROUP>` | Azure Resource Group Name          |
| `<NIC_NAME>`       | Name of the Network Interface Card |
| `<IPCONFIG_NAME>`  | IP Configuration Name on the NIC   |
| `<PUBLIC_IP_NAME>` | Name of the Public IP Resource     |

---

### 📘 Resize a Virtual Machine in Azure CLI

This guide shows how to change the size of a virtual machine (VM) and ensure it is running after the change using the Azure CLI.

---

#### 🔹 Step 1: List All Virtual Machines

```bash
az vm list -o table
```

---

#### 🔹 Step 2: Deallocate the VM (Required Before Resizing)

```bash
az vm deallocate \
  --resource-group <RESOURCE_GROUP> \
  --name <VM_NAME>
```

---

#### 🔹 Step 3: Resize the VM

```bash
az vm resize \
  --resource-group <RESOURCE_GROUP> \
  --name <VM_NAME> \
  --size Standard_B2s
```

---

#### 🔹 Step 4: Start the VM

```bash
az vm start \
  --resource-group <RESOURCE_GROUP> \
  --name <VM_NAME>
```

---

#### 🔹 Step 5: Verify the New Size and Power State

```bash
az vm show \
  --resource-group <RESOURCE_GROUP> \
  --name <VM_NAME> \
  --query "{VMSize:hardwareProfile.vmSize, PowerState:instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus | [0]}" \
  --output table
```

---

### 📝 Notes:
- Replace `<RESOURCE_GROUP>` and `<VM_NAME>` with your actual values.
- VM must be deallocated before resizing.


# Creating a Managed Disk in Azure

This guide demonstrates how to create a managed disk in Azure using the Azure CLI.

## Requirements

- **Disk Name:** `xfusion-disk`
- **Disk Type:** `Standard_LRS` (Standard HDD)
- **Disk Size:** `2 GiB`
- **Resource Group:** Replace `<RESOURCE_GROUP>` with your actual resource group name.
- **Location:** Replace `<LOCATION>` with your desired Azure region (e.g., `eastus`, `westeurope`).

## Azure CLI Command

```bash
az disk create \
  --name xfusion-disk \
  --resource-group <RESOURCE_GROUP> \
  --size-gb 2 \
  --sku Standard_LRS \
  --location <LOCATION>
```

## Notes

- Ensure you are logged into your Azure account using `az login`.
- The specified resource group must already exist.
- You can view available locations with `az account list-locations -o table`.


# Creating a Network Security Group (NSG) in Azure

This guide shows how to create a Network Security Group (NSG) and add inbound security rules using the Azure CLI.

## Requirements

- **NSG Name:** `nautilus-nsg`
- **Rules:**
  - `Allow-HTTP`:
    - Protocol: TCP
    - Port: 80
    - Source: `0.0.0.0/0`
  - `Allow-SSH`:
    - Protocol: TCP
    - Port: 22
    - Source: `0.0.0.0/0`
- **Resource Group:** Replace `<RESOURCE_GROUP>` with your actual resource group.
- **Location:** Replace `<LOCATION>` with your desired Azure region (e.g., `eastus`, `westeurope`).

## Azure CLI Commands

### Step 1: Create the NSG

```bash
az network nsg create \
  --resource-group <RESOURCE_GROUP> \
  --name nautilus-nsg \
  --location <LOCATION>
```

### Step 2: Add Inbound Rule for HTTP

```bash
az network nsg rule create \
  --resource-group <RESOURCE_GROUP> \
  --nsg-name nautilus-nsg \
  --name Allow-HTTP \
  --priority 1001 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes 0.0.0.0/0 \
  --destination-port-ranges 80
```

### Step 3: Add Inbound Rule for SSH

```bash
az network nsg rule create \
  --resource-group <RESOURCE_GROUP> \
  --nsg-name nautilus-nsg \
  --name Allow-SSH \
  --priority 1002 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes 0.0.0.0/0 \
  --destination-port-ranges 22
```

## Notes

- Rule priority values must be unique and within the range 100–4096.
- Use descriptive names and verify rules with `az network nsg rule list`.

