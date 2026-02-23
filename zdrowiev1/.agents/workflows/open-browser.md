---
description: How to open the Zdrowie App in the browser and login with demo credentials
---

# Open Browser and Login

This workflow opens the Zdrowie App in the browser and logs in. It requires servers to already be running (use `/run-servers` first).

## Prerequisites
Run the `/run-servers` workflow first, or verify the stack is healthy:
// turbo
```bash
powershell -File scripts/check-stack.ps1
```
If any check fails, follow `/run-servers` workflow first.

## Steps

### 1. Open login page in the default browser
// turbo
```bash
Start-Process "http://localhost:3000/login"
```

### 2. Login credentials
Use these credentials in the browser:
- **Email:** `demo@example.com`
- **Password:** `Password123!`

### 3. After login
You should be redirected to `http://localhost:3000/dashboard` which shows the main health dashboard with:
- Health metrics (Heart Rate, Sleep, Weight)
- Notification bell
- Sidebar navigation to all pages (Health, Diet, Diagnosis, OCR, Reports)
