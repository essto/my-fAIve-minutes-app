# check-stack.ps1 - Smoke test for the full Zdrowie App stack
# Run: powershell -File scripts/check-stack.ps1
# Returns exit code 0 if all checks pass, 1 if any fail.

$ErrorActionPreference = "Continue"
$failed = 0

Write-Host "=== Zdrowie App Stack Health Check ===" -ForegroundColor Cyan
Write-Host ""

# 1. Docker containers
Write-Host "[1/4] Checking Docker containers..." -NoNewline
$db = docker ps --filter "name=gesundheit-db" --format "{{.Status}}" 2>$null
$redis = docker ps --filter "name=gesundheit-redis" --format "{{.Status}}" 2>$null
if ($db -and $redis) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " FAIL - run: docker-compose up -d" -ForegroundColor Red
    $failed++
}

# 2. Database tables
Write-Host "[2/4] Checking database tables..." -NoNewline
try {
    $tables = docker exec gesundheit-db psql -U postgres -d health -c "\dt" 2>&1
    $tablesStr = $tables | Out-String
    if ($tablesStr -match "users") {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAIL - users table missing" -ForegroundColor Red
        $failed++
    }
} catch {
    Write-Host " FAIL - cannot query database" -ForegroundColor Red
    $failed++
}

# 3. NestJS API on port 3001
Write-Host "[3/4] Checking NestJS API (port 3001)..." -NoNewline
try {
    $null = Invoke-WebRequest -Uri http://localhost:3001/api -Method GET -TimeoutSec 5 -ErrorAction Stop
    Write-Host " OK" -ForegroundColor Green
} catch {
    $statusCode = $null
    try { $statusCode = $_.Exception.Response.StatusCode.value__ } catch {}
    if ($statusCode -eq 404) {
        Write-Host " OK (running)" -ForegroundColor Green
    } else {
        Write-Host " FAIL - API not responding. Build and start it first." -ForegroundColor Red
        $failed++
    }
}

# 4. Login endpoint
Write-Host "[4/4] Testing login endpoint..." -NoNewline
try {
    $body = '{"email":"demo@example.com","password":"Password123!"}'
    $loginResult = Invoke-RestMethod -Uri http://localhost:3001/api/auth/login -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 5
    if ($loginResult.access_token) {
        Write-Host " OK - JWT received" -ForegroundColor Green
    } else {
        Write-Host " FAIL - no token returned" -ForegroundColor Red
        $failed++
    }
} catch {
    Write-Host " FAIL - login error" -ForegroundColor Red
    $failed++
}

Write-Host ""
if ($failed -eq 0) {
    Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
    Write-Host "Open http://localhost:3000/login in your browser"
    Write-Host "Login: demo@example.com / Password123!"
    exit 0
} else {
    Write-Host ("=== {0} CHECK(S) FAILED ===" -f $failed) -ForegroundColor Red
    exit 1
}
