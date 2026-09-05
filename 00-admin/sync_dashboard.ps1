<#
  sync_dashboard.ps1
  ------------------------------------------------------------------
  Single source of truth = 00-admin/timeline.md
  This script regenerates the milestone data block + KPI META inside
  00-admin/dashboard.html so the dashboard always matches timeline.md.

  It also runs "evidence" rules: for milestones that have a concrete
  artifact signal (e.g. a Track B compile log containing
  "DPU subgraph number 1"), it auto-ticks that row in timeline.md.

  Usage:
    powershell -NoProfile -ExecutionPolicy Bypass -File sync_dashboard.ps1
    ... -HookStdin   (called from a Claude Code PostToolUse hook; reads
                      the tool JSON on stdin and only runs when the edited
                      file was timeline.md)
    ... -Quiet       (suppress console output)

  NOTE: intentionally ASCII-only source. Thai / emoji literals are built
  from code points so Windows PowerShell 5.1 reads the file correctly
  regardless of BOM.
#>
param(
  [switch]$HookStdin,
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

function Log($m){ if(-not $Quiet){ Write-Host $m } }

# ---- non-ASCII tokens, built from code points ----
$CHK = [char]0x2705                                                   # done mark
$YES = -join ([char]0x0E43,[char]0x0E0A,[char]0x0E48)                 # board = yes
$UPD = -join ([char]0x0E2D,[char]0x0E31,[char]0x0E1B,[char]0x0E40,[char]0x0E14,[char]0x0E15) # "updated"

# ---- paths ----
$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root     = Split-Path -Parent $Here
$Timeline = Join-Path $Here 'timeline.md'
$Dash     = Join-Path $Here 'dashboard.html'

# ---- hook mode: only proceed when timeline.md was the edited file ----
if($HookStdin){
  $raw = [Console]::In.ReadToEnd()
  if($raw){
    $j = $null
    try { $j = $raw | ConvertFrom-Json } catch { $j = $null }
    $fp = $null
    if($j -and $j.tool_input){ $fp = $j.tool_input.file_path }
    if($fp -and ($fp -notmatch 'timeline\.md$')){ exit 0 }
  }
}

if(-not (Test-Path $Timeline)){ Log "timeline.md not found: $Timeline"; exit 1 }
if(-not (Test-Path $Dash)){ Log "dashboard.html not found: $Dash"; exit 1 }

$utf8 = New-Object System.Text.UTF8Encoding($false)
function ReadText($p){ [System.IO.File]::ReadAllText($p, [System.Text.Encoding]::UTF8) }
function WriteText($p,$t){ [System.IO.File]::WriteAllText($p, $t, $utf8) }

# ================= parse timeline.md =================
$tl      = ReadText $Timeline
$tlLines = $tl -split "`r?`n"
$idRe    = '^(M\d+(?:-\w+)?|P\d+)$'
$status  = @{}   # id -> 'done' | 'blocked' | 'todo'
$boardOf = @{}   # id -> [bool]

foreach($line in $tlLines){
  if($line -notmatch '^\s*\|'){ continue }
  if($line -match '^\s*\|\s*-'){ continue }               # table separator
  $cells = ($line.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
  if($cells.Count -lt 3){ continue }
  $id = ($cells[0] -replace '\*','').Trim()
  if($id -notmatch $idRe){ continue }
  if($cells.Count -ge 5){ $board = $cells[3]; $st = $cells[4] }
  else                  { $board = '';        $st = $cells[$cells.Count-1] }
  $done     = $st.Contains($CHK)
  $boardYes = $board.Contains($YES)
  $boardOf[$id] = $boardYes
  if($done)         { $status[$id] = 'done' }
  elseif($boardYes) { $status[$id] = 'blocked' }
  else              { $status[$id] = 'todo' }
}

if($status.Count -eq 0){ Log 'no milestones parsed from timeline.md - aborting'; exit 1 }

# ================= evidence rules (auto-tick) =================
# Each rule: id + a test that returns $true when the artifact proves it done.
# Conservative on purpose: only tick, never un-tick.
$evidence = @(
  @{ id = 'M2-B3'; label = 'Track B compile gate'; test = {
      # A genuine Track B compile log for YOLO26n. CAUTION: the recovered
      # backup contains 'vai_c_xir_yolo26n_kv260.log' that is actually
      # YOLOv8n (filename misnomer, proven in P1) -> exclude backup / Track-A
      # / v8n paths so this cannot false-positive on the mislabeled file.
      $dir = Join-Path $Root '03-model'
      if(-not (Test-Path $dir)){ return $false }
      $exclude = 'phase0-wsl-recovered|track-a|yolov8n|recovered|backup'
      $logs = Get-ChildItem -LiteralPath $dir -Recurse -File -Include *.log,*.txt -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -match 'yolo26n' -and $_.Name -match 'compile|vai_c|xir' -and $_.FullName -notmatch $exclude }
      foreach($f in $logs){
        if(Select-String -LiteralPath $f.FullName -Pattern 'DPU subgraph number 1' -SimpleMatch -Quiet -ErrorAction SilentlyContinue){ return $true }
      }
      return $false
  }}
  # ,@{ id='M5'; label='CPU emulation verify'; test={ Test-Path (Join-Path $Root '04-deploy\emu_verify_PASS') } }
)

$flipped = @()
foreach($rule in $evidence){
  if($status[$rule.id] -ne 'done'){
    $ok = $false
    try { $ok = & $rule.test } catch { $ok = $false }
    if($ok){ $status[$rule.id] = 'done'; $flipped += $rule.id }
  }
}

# write auto-ticks back into timeline.md (format-preserving: only the status cell)
if($flipped.Count -gt 0){
  $eval = [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $m.Groups[1].Value + ' ' + $CHK + ' |' }
  for($i=0; $i -lt $tlLines.Count; $i++){
    $line = $tlLines[$i]
    if($line -notmatch '^\s*\|'){ continue }
    $cells = ($line.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
    if($cells.Count -lt 1){ continue }
    $id = ($cells[0] -replace '\*','').Trim()
    if($flipped -contains $id){
      $tlLines[$i] = [regex]::Replace($line, '^(.*\|)\s*[^|]*\s*\|\s*$', $eval)
    }
  }
  WriteText $Timeline (($tlLines -join "`r`n"))
  Log ("timeline.md auto-ticked: " + ($flipped -join ', '))
}

# ================= recompute META =================
$done    = @($status.Values | Where-Object { $_ -eq 'done' }).Count
$blocked = @($status.Values | Where-Object { $_ -eq 'blocked' }).Count
$total   = $status.Count
$ready   = $total - $done - $blocked
$pct     = [math]::Round($done * 100.0 / $total)

$updated = '-'
$updRe = [regex]::Escape($UPD) + '\s*([0-9][^\r\n]*)'
if($tl -match $updRe){ $updated = $Matches[1].Trim() }

# ================= rewrite dashboard.html data block =================
$dhLines = (ReadText $Dash) -split "`r?`n"
$inBlock = $false
$touched = 0
for($i=0; $i -lt $dhLines.Count; $i++){
  $l = $dhLines[$i]
  if($l -match '/\*==MILESTONES:START==\*/'){ $inBlock = $true;  continue }
  if($l -match '/\*==MILESTONES:END==\*/'){   $inBlock = $false; continue }
  if(-not $inBlock){ continue }

  if($l -match 'var\s+META\s*='){
    $dhLines[$i] = "  var META={pct:$pct,done:$done,ready:$ready,blocked:$blocked,total:$total,updated:'$updated'};"
    continue
  }
  if($l -match "id:'([^']+)'"){
    $id = $Matches[1]
    if($status.ContainsKey($id)){
      $s = $status[$id]
      $b = if($boardOf[$id]){ 'true' } else { 'false' }
      $l = [regex]::Replace($l, "s:'[^']*'", "s:'$s'")
      $l = [regex]::Replace($l, 'board:(?:true|false)', "board:$b")
      $dhLines[$i] = $l
      $touched++
    }
  }
}

WriteText $Dash (($dhLines -join "`r`n"))
Log "dashboard.html synced: done=$done ready=$ready blocked=$blocked total=$total ($pct%) - $touched rows"
