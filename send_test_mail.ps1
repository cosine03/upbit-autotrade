# send_test_mail.ps1  (PS5 안정판)

# --- UTF-8 콘솔 ---
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($true)

param(
  [switch]$UseEnv,
  [string]$EnvPath = ".\.env"
)

function Load-DotEnv([string]$Path){
  if (-not (Test-Path $Path)) { Write-Warning "[WARN] .env not found: $Path"; return }
  $raw = Get-Content -Path $Path -Raw -Encoding UTF8
  if ($raw.Length -gt 0 -and $raw[0] -eq [char]0xFEFF) { $raw = $raw.Substring(1) } # BOM 제거
  foreach($line in $raw -split "`n"){
    $line = $line.Trim()
    if (-not $line -or $line.StartsWith('#')) { continue }
    $eq = $line.IndexOf('=')
    if ($eq -lt 1) { continue }
    $k = $line.Substring(0,$eq).Trim()
    $v = $line.Substring($eq+1).Trim()
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1,$v.Length-2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1,$v.Length-2) }
    Set-Item -Path ("Env:{0}" -f $k) -Value $v
  }
  Write-Host "[INFO] .env loaded (SMTP_* / MAIL_*):" -ForegroundColor Cyan
  Get-ChildItem Env:SMTP_* , Env:MAIL_* | Sort-Object Name | Format-Table Name,Value -Auto
}

if ($UseEnv) { Load-DotEnv -Path $EnvPath }

# .env 키(둘 다 허용)
$from = if ($env:SMTP_FROM) { $env:SMTP_FROM } else { $env:MAIL_FROM }
$toRaw = if ($env:SMTP_TO) { $env:SMTP_TO } else { $env:MAIL_TO }

if (-not $from -or -not $toRaw -or -not $env:SMTP_HOST -or -not $env:SMTP_PORT -or -not $env:SMTP_USER -or -not $env:SMTP_PASS) {
  throw "SMTP_FROM/SMTP_TO/SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS 중 누락. (.env 또는 환경변수 확인)"
}

$sec  = ConvertTo-SecureString $env:SMTP_PASS -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential($env:SMTP_USER, $sec)
$to   = $toRaw -split '\s*,\s*'
$useSsl = $true
if ($env:SMTP_TLS) { try { $useSsl = [bool]::Parse($env:SMTP_TLS) } catch {} }

$params = @{
  From        = $from
  To          = $to
  Subject     = '[PaperTrader] SMTP test'
  Body        = '<b>This is a test.</b>'
  BodyAsHtml  = $true
  SmtpServer  = $env:SMTP_HOST
  Port        = [int]$env:SMTP_PORT
  UseSsl      = $useSsl
  Credential  = $cred
  ErrorAction = 'Stop'
}

Send-MailMessage @params
Write-Host "[OK] Test mail sent to $($to -join ', ')" -ForegroundColor Green