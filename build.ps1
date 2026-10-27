param([string[]]$Targets)
$ErrorActionPreference='Stop'
$arguments=@('library','build',$PSScriptRoot)
if($Targets){$arguments+=@('--targets',($Targets -join ','))}
& aqua @arguments
if($LASTEXITCODE -ne 0){throw "OpenCV SDK build failed: $LASTEXITCODE"}
