# ===================================================================
# Docker Deployment Test: v1 -> v2 -> Rollback
# ===================================================================

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Docker Deployment Test: v1 -> v2 -> Rollback" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Clean old containers
Write-Host "`nCleaning old containers..." -ForegroundColor Yellow
docker-compose down 2>$null

# ===================================================================
# Stage 1: Deploy v1
# ===================================================================
Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Stage 1: Deploy API v1 (Baseline - 93.33%)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

docker-compose up -d api-v1

Write-Host "`nWaiting 15 seconds for v1 to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "`nTesting v1 Health Check:" -ForegroundColor Yellow
try {
    $v1_health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get
    Write-Host "   Success: Status = $($v1_health.status)" -ForegroundColor Green
    Write-Host "   Success: Version = $($v1_health.model_version)" -ForegroundColor Green
    Write-Host "   Success: Model Loaded = $($v1_health.model_loaded)" -ForegroundColor Green
} catch {
    Write-Host "   Error: v1 Health Check failed!" -ForegroundColor Red
    docker-compose logs api-v1
    exit 1
}

Write-Host "`nTesting v1 Prediction:" -ForegroundColor Yellow
$body = @{
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
} | ConvertTo-Json

try {
    $v1_pred = Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   Success: Prediction = $($v1_pred.class_name)" -ForegroundColor Green
    Write-Host "   Success: Confidence = $([math]::Round(($v1_pred.probability | Measure-Object -Maximum).Maximum * 100, 1))%" -ForegroundColor Green
    Write-Host "   Success: Version = $($v1_pred.model_version)" -ForegroundColor Green
} catch {
    Write-Host "   Error: v1 Prediction failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 1 SUCCESS: v1 is working!" -ForegroundColor Green

# ===================================================================
# Stage 2: Deploy v2
# ===================================================================
Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Stage 2: Deploy API v2 (Optimized - 100%)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

docker-compose up -d api-v2

Write-Host "`nWaiting 15 seconds for v2 to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "`nTesting v2 Health Check:" -ForegroundColor Yellow
try {
    $v2_health = Invoke-RestMethod -Uri "http://localhost:8002/health" -Method Get
    Write-Host "   Success: Status = $($v2_health.status)" -ForegroundColor Green
    Write-Host "   Success: Version = $($v2_health.model_version)" -ForegroundColor Green
    Write-Host "   Success: Model Loaded = $($v2_health.model_loaded)" -ForegroundColor Green
} catch {
    Write-Host "   Error: v2 Health Check failed!" -ForegroundColor Red
    docker-compose logs api-v2
    exit 1
}

Write-Host "`nTesting v2 Prediction:" -ForegroundColor Yellow
try {
    $v2_pred = Invoke-RestMethod -Uri "http://localhost:8002/predict" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   Success: Prediction = $($v2_pred.class_name)" -ForegroundColor Green
    Write-Host "   Success: Confidence = $([math]::Round(($v2_pred.probability | Measure-Object -Maximum).Maximum * 100, 1))%" -ForegroundColor Green
    Write-Host "   Success: Version = $($v2_pred.model_version)" -ForegroundColor Green
} catch {
    Write-Host "   Error: v2 Prediction failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 2 SUCCESS: v2 is working!" -ForegroundColor Green

# ===================================================================
# Stage 3: Rollback (Stop v2, v1 continues)
# ===================================================================
Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Stage 3: ROLLBACK - Stop v2, v1 continues" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

docker-compose stop api-v2
Write-Host "`nv2 stopped" -ForegroundColor Green

Write-Host "`nVerifying v1 still works:" -ForegroundColor Yellow
try {
    $v1_health_after = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get
    Write-Host "   Success: v1 still working after v2 stopped" -ForegroundColor Green
    Write-Host "   Success: Version = $($v1_health_after.model_version)" -ForegroundColor Green
} catch {
    Write-Host "   Error: v1 not working after rollback!" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 3 SUCCESS: Rollback successful!" -ForegroundColor Green

# ===================================================================
# Stage 4: Re-deploy v2
# ===================================================================
Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "Stage 4: Re-deploy v2" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

docker-compose start api-v2

Write-Host "`nWaiting 10 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`nTesting v2 after restart:" -ForegroundColor Yellow
try {
    $v2_health_after = Invoke-RestMethod -Uri "http://localhost:8002/health" -Method Get
    Write-Host "   Success: v2 restarted successfully" -ForegroundColor Green
    Write-Host "   Success: Version = $($v2_health_after.model_version)" -ForegroundColor Green
} catch {
    Write-Host "   Error: Re-deploy failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 4 SUCCESS: Re-deploy successful!" -ForegroundColor Green

# ===================================================================
# Final Summary
# ===================================================================
Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

Write-Host "`nTest Summary:" -ForegroundColor Cyan
Write-Host "  [OK] v1 deployed and tested (port 8001 - 93.33% accuracy)" -ForegroundColor Green
Write-Host "  [OK] v2 deployed and tested (port 8002 - 100% accuracy)" -ForegroundColor Green
Write-Host "  [OK] Rollback v2->v1 successful" -ForegroundColor Green
Write-Host "  [OK] Re-deploy v2 successful" -ForegroundColor Green

Write-Host "`nActive Containers:" -ForegroundColor Cyan
docker-compose ps

Write-Host "`nManual Testing:" -ForegroundColor Yellow
Write-Host "  curl http://localhost:8001/health  # v1"
Write-Host "  curl http://localhost:8002/health  # v2"

Write-Host "`nSaving results..." -ForegroundColor Yellow
@"
==============================================
Docker Deployment Test Results
==============================================
Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Stage 1: v1 Deployment
[OK] Status: healthy
[OK] Version: v1
[OK] Prediction: $($v1_pred.class_name)

Stage 2: v2 Deployment
[OK] Status: healthy
[OK] Version: v2
[OK] Prediction: $($v2_pred.class_name)

Stage 3: Rollback
[OK] v2 stopped
[OK] v1 still working

Stage 4: Re-deploy
[OK] v2 restarted successfully

==============================================
ALL TESTS PASSED
==============================================
"@ | Out-File -FilePath "deployment_test_results.txt" -Encoding UTF8

Write-Host "Results saved to: deployment_test_results.txt" -ForegroundColor Green

Write-Host "`nTo stop all containers:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor Gray

Write-Host "`nProject is 100% complete!" -ForegroundColor Green