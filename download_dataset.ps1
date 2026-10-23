
$ErrorActionPreference = "Stop"

$Raw = Join-Path $PSScriptRoot "data/raw"
$GoogleDriveUrl = "https://drive.google.com/file/d/1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR/view"
$GoogleDriveZip = Join-Path $Raw "TBX11K.zip"

# $Datasets = @(
#     @{
#         Name = "montgomery"
#         Base = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"
#         Folders = @(
#             ""
#             "CXR_png/"
#             "ClinicalReadings/"
#             "ManualMask/leftMask/"
#             "ManualMask/rightMask/"
#         )
#     },
#     @{
#         Name = "shenzhen"
#         Base = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/"
#         Folders = @(
#             ""
#             "CXR_png/"
#             "ClinicalReadings/"
#             "Annotations/"
#             "Annotations/masks/"
#             "Annotations/Annotations_json/"
#             "Annotations/Annotations_json/SeparateFiles/"
#             "Annotations-2/"
#             "Annotations-2/masks/"
#             "Annotations-2/Annotations_json/"
#             "Annotations-2/Annotations_json/SeparateFiles/"
#         )
#     }
# )

# $Failed = [System.Collections.Generic.List[string]]::new()
# $Downloaded = 0
# $Skipped = 0

# foreach ($DS in $Datasets) {
#     $Dest = Join-Path $Raw $DS.Name

#     foreach ($Folder in $DS.Folders) {
#         $FolderUrl = $DS.Base + $Folder
#         $IndexUrl = $FolderUrl + "index.html"
#         $LocalDir = Join-Path $Dest ($Folder.Replace("/", "\"))

#         New-Item -ItemType Directory -Path $LocalDir -Force |
#             Out-Null

#         Write-Host "`n[$($DS.Name)] $Folder" -ForegroundColor Cyan

#         try {
#             $R = Invoke-WebRequest `
#                 -Uri $IndexUrl `
#                 -Headers @{ "User-Agent" = "Mozilla/5.0" } `
#                 -UseBasicParsing `
#                 -TimeoutSec 60

#             $Html = $R.Content
#         }
#         catch {
#             Write-Host "GAGAL MEMBACA INDEX: $IndexUrl" -ForegroundColor Red
#             $Failed.Add($IndexUrl)
#             continue
#         }

#         $Links = [regex]::Matches(
#             $Html,
#             '(?is)href\s*=\s*["'']([^"'']+)["'']'
#         )

#         foreach ($M in $Links) {
#             $Href = [System.Net.WebUtility]::HtmlDecode(
#                 $M.Groups[1].Value.Trim()
#             )

#             if (
#                 !$Href -or
#                 $Href.StartsWith("#") -or
#                 $Href.StartsWith("?") -or
#                 $Href -match '^(?i)(https?:|mailto:|javascript:)'
#             ) {
#                 continue
#             }

#             try {
#                 $Uri = [uri]::new([uri]$IndexUrl, $Href)
#             }
#             catch {
#                 continue
#             }

#             if (
#                 $Uri.Host -ne "data.lhncbc.nlm.nih.gov" -or
#                 !$Uri.AbsolutePath.StartsWith(
#                     ([uri]$DS.Base).AbsolutePath,
#                     [StringComparison]::OrdinalIgnoreCase
#                 )
#             ) {
#                 continue
#             }

#             # Folder links are handled by the explicit folder list.
#             if ($Uri.AbsolutePath.EndsWith("/")) {
#                 continue
#             }

#             $FileName = [uri]::UnescapeDataString(
#                 [IO.Path]::GetFileName($Uri.AbsolutePath)
#             )

#             if (!$FileName -or $FileName -eq "index.html") {
#                 continue
#             }

#             $Path = Join-Path $LocalDir $FileName

#             if (
#                 (Test-Path -LiteralPath $Path) -and
#                 (Get-Item -LiteralPath $Path).Length -gt 0
#             ) {
#                 $Skipped++
#                 continue
#             }

#             $Part = "$Path.part"
#             Write-Host "GET $FileName"

#             & curl.exe -fLsS -L `
#                 --retry 3 `
#                 --retry-delay 2 `
#                 --connect-timeout 30 `
#                 -A "Mozilla/5.0" `
#                 $Uri.AbsoluteUri `
#                 -o $Part

#             if (
#                 $LASTEXITCODE -eq 0 -and
#                 (Test-Path -LiteralPath $Part) -and
#                 (Get-Item -LiteralPath $Part).Length -gt 0
#             ) {
#                 Move-Item -LiteralPath $Part -Destination $Path -Force
#                 $Downloaded++
#             }
#             else {
#                 Remove-Item -LiteralPath $Part -Force `
#                     -ErrorAction SilentlyContinue

#                 Write-Host "GAGAL: $Uri" -ForegroundColor Red
#                 $Failed.Add($Uri.AbsoluteUri)
#             }
#         }
#     }
# }


Write-Host "`n[Google Drive] TBX11K.zip" -ForegroundColor Cyan

try {
    New-Item -ItemType Directory -Path $Raw -Force |
        Out-Null

    if (
        (Test-Path -LiteralPath $GoogleDriveZip) -and
        (Get-Item -LiteralPath $GoogleDriveZip).Length -gt 0
    ) {
        Write-Host "ZIP sudah ada, dilewati." -ForegroundColor DarkYellow
        $Skipped++
    }
    else {
        $Part = "$GoogleDriveZip.part"

        # Pastikan gdown tersedia
        & py -m pip show gdown *> $null

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Menginstal gdown..." -ForegroundColor Yellow

            & py -m pip install gdown

            if ($LASTEXITCODE -ne 0) {
                throw "Gagal menginstal gdown."
            }
        }

        # Hapus file sementara dari percobaan sebelumnya.
        # gdown tidak otomatis melanjutkan file .part ini.
        Remove-Item -LiteralPath $Part -Force `
            -ErrorAction SilentlyContinue

        $Success = $false

        for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
            Write-Host "Percobaan $Attempt dari 3..."

            & py -m gdown `
                --fuzzy `
                $GoogleDriveUrl `
                -O $Part

            if (
                $LASTEXITCODE -eq 0 -and
                (Test-Path -LiteralPath $Part) -and
                (Get-Item -LiteralPath $Part).Length -gt 0
            ) {
                # Verifikasi bahwa file benar-benar ZIP
                $Stream = [System.IO.File]::OpenRead($Part)

                try {
                    $B1 = $Stream.ReadByte()
                    $B2 = $Stream.ReadByte()
                }
                finally {
                    $Stream.Dispose()
                }

                if ($B1 -eq 0x50 -and $B2 -eq 0x4B) {
                    Move-Item -LiteralPath $Part `
                        -Destination $GoogleDriveZip -Force

                    $Downloaded++
                    $Success = $true

                    Write-Host "ZIP berhasil diunduh." -ForegroundColor Green
                    Write-Host "Lokasi: $GoogleDriveZip"
                    break
                }
                else {
                    Write-Host "File bukan ZIP yang valid." -ForegroundColor Red
                }
            }

            Remove-Item -LiteralPath $Part -Force `
                -ErrorAction SilentlyContinue

            if ($Attempt -lt 3) {
                Start-Sleep -Seconds 3
            }
        }

        if (-not $Success) {
            throw "Download Google Drive gagal setelah 3 percobaan."
        }
    }
}
catch {
    $Message = "Google Drive: $GoogleDriveUrl | $($_.Exception.Message)"

    Write-Host $Message -ForegroundColor Red
    $Failed.Add($Message)
}


$FailedLog = Join-Path $Raw "download-failures.txt"
$Failed | Set-Content -LiteralPath $FailedLog -Encoding UTF8

Write-Host "`nResult" -ForegroundColor Yellow
Write-Host "Berhasil diunduh : $Downloaded"
Write-Host "Dilewati         : $Skipped"
Write-Host "Gagal            : $($Failed.Count)"
Write-Host "Log kegagalan    : $FailedLog"
