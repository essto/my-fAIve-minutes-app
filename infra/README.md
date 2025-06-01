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

> ⚠️ Ensure that the public IP resource exists prior to assignment.
