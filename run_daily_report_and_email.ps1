param(
  [int]$Days = 1,
  [string]$CsvPath = ".\logs\paper\trades_closed.csv",
  [string]$SmtpServer = "smtp.example.com",
  [int]$SmtpPort = 587,
  [string]$From = "bot@example.com",
  [string]$To = "you@example.com",
  [string]$User = "bot@example.com",
  [string]$Password = "app-password",
  [string]$Subject = "[PaperTrader] Closed Trades Report"
)

function As-HtmlTable($rows) {
@"
<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-family:Segoe UI,Arial">
  <thead>
    <tr>
      <th>opened_at</th><th>symbol</th><th>event</th><th>side</th><th>level</th>
      <th>closed_at</th><th>entry_price</th><th>exit_price</th><th>pnl</th><th>reason</th><th>fee</th>
    </tr>
  </thead>
  <tbody>
"@ +
($rows | ForEach-Object {
  "<tr><td>$($_.opened_at)</td><td>$($_.symbol)</td><td>$($_.event)</td><td>$($_.side)</td><td>$($_.level)</td><td>$($_.closed_at)</td><td>$($_.entry_price)</td><td>$($_.exit_price)</td><td>$($_.pnl)</td><td>$($_.reason)</td><td>$($_.fee)</td></tr>"
}) + @"
  </tbody>
</table>
"@
}

if (-not (Test-Path $CsvPath)) {
  $body = "<p>No trades_closed.csv found.</p>"
} else {
  try {
    $since = [datetimeoffset]::UtcNow.AddDays(-$Days)
    $rows = Import-Csv $CsvPath | Where-Object {
      try { [datetimeoffset]$_.closed_at -ge $since } catch { $false }
    }
    if (-not $rows -or $rows.Count -eq 0) {
      $body = "<p>No closed trades in the last $Days day(s).</p>"
    } else {
      $body = As-HtmlTable $rows
    }
  } catch {
    $body = "<p>Error reading CSV: $($_.Exception.Message)</p>"
  }
}

$secure = ConvertTo-SecureString $Password -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential($User, $secure)
Send-MailMessage -To $To -From $From -Subject $Subject -BodyAsHtml -Body $body -SmtpServer $SmtpServer -Port $SmtpPort -UseSsl -Credential $cred
Write-Host "[OK] Report emailed."