# C: every 10 minutes, log system free memory and the top 5 non-Python processes by working set (one JSON line per sample).
#   powershell -NoProfile -File e2e/c/mem_watch.ps1 [-Out e2e/c/out/mem/system.jsonl] [-Every 600]
param([string]$Out = "e2e/c/out/mem/system.jsonl", [int]$Every = 600)
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $procs = Get-Process
    $py = $procs | Where-Object { $_.Name -like "python*" }
    $top = $procs | Where-Object { $_.Name -notlike "python*" } | Group-Object Name |
        ForEach-Object { [pscustomobject]@{ name = $_.Name; n = $_.Count; gb = [math]::Round(($_.Group | Measure-Object WorkingSet64 -Sum).Sum / 1GB, 2) } } |
        Sort-Object gb -Descending | Select-Object -First 5
    $rec = [ordered]@{
        t = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        free_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        total_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        python_n = @($py).Count
        python_gb = [math]::Round(($py | Measure-Object WorkingSet64 -Sum).Sum / 1GB, 2)
        top5_non_python = @($top)
    }
    Add-Content -Path $Out -Value ($rec | ConvertTo-Json -Compress -Depth 4) -Encoding utf8
    Start-Sleep -Seconds $Every
}
