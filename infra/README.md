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
