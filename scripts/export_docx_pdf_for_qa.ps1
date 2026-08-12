param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,
    [Parameter(Mandatory = $true)]
    [string]$OutputPdf
)

$docxPath = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfParent = Split-Path -Parent $OutputPdf
if (-not (Test-Path -LiteralPath $pdfParent)) {
    New-Item -ItemType Directory -Path $pdfParent -Force | Out-Null
}
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($docxPath, $false, $true)
    foreach ($toc in @($document.TablesOfContents)) {
        $toc.Update() | Out-Null
    }
    $document.Fields.Update() | Out-Null
    $document.ExportAsFixedFormat($pdfPath, 17)
    Write-Output $pdfPath
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
