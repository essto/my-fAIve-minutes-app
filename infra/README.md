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
