
# Azure Storage Account and Blob Container Setup

## Overview

This document outlines the process to create a new Azure Storage Account and a private Blob container.

## Prerequisites

Ensure that the Azure CLI is installed and authenticated with sufficient permissions.

## Steps

### 1. Create a Storage Account

```bash
az storage account create   --name <STORAGE_ACCOUNT_NAME>   --resource-group <RESOURCE_GROUP>   --location <LOCATION>   --sku Standard_LRS   --kind StorageV2
```

### 2. Retrieve the Storage Account Key

```bash
az storage account keys list   --account-name <STORAGE_ACCOUNT_NAME>   --resource-group <RESOURCE_GROUP>   --query "[0].value"   --output tsv
```

Store the key in a variable for convenience:

```bash
STORAGE_KEY=$(az storage account keys list   --account-name <STORAGE_ACCOUNT_NAME>   --resource-group <RESOURCE_GROUP>   --query "[0].value"   --output tsv)
```

### 3. Create a Private Blob Container

```bash
az storage container create   --name <CONTAINER_NAME>   --account-name <STORAGE_ACCOUNT_NAME>   --account-key $STORAGE_KEY   --public-access off
```

### Notes

- Replace `<STORAGE_ACCOUNT_NAME>`, `<RESOURCE_GROUP>`, `<LOCATION>`, and `<CONTAINER_NAME>` with actual values.
- Ensure naming conventions are respected (lowercase, globally unique names for storage accounts).

## Verification

To list containers:

```bash
az storage container list   --account-name <STORAGE_ACCOUNT_NAME>   --account-key $STORAGE_KEY   --output table
```
