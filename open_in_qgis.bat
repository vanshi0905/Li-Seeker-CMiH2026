@echo off
REM =============================================================================
REM Li-Seeker QGIS Launcher (CLI & Desktop App)
REM Automatically launches QGIS Desktop with Katghora prospectivity layers
REM =============================================================================

echo Searching for QGIS installation...

set QGIS_BAT=
for /d %%i in ("C:\Program Files\QGIS*") do (
    if exist "%%i\bin\qgis-ltr.bat" set QGIS_BAT="%%i\bin\qgis-ltr.bat"
    if exist "%%i\bin\qgis.bat" if not defined QGIS_BAT set QGIS_BAT="%%i\bin\qgis.bat"
)

if not defined QGIS_BAT (
    if exist "C:\OSGeo4W\bin\qgis-ltr.bat" set QGIS_BAT="C:\OSGeo4W\bin\qgis-ltr.bat"
    if exist "C:\OSGeo4W\bin\qgis.bat" if not defined QGIS_BAT set QGIS_BAT="C:\OSGeo4W\bin\qgis.bat"
)

if not defined QGIS_BAT (
    echo [ERROR] QGIS installation not found in C:\Program Files\QGIS* or C:\OSGeo4W.
    echo Please ensure the installation completes, or run:
    echo   winget install --id OSGeo.QGIS_LTR -e
    pause
    exit /b 1
)

echo [OK] Found QGIS at: %QGIS_BAT%
echo Launching QGIS with Katghora GeoTIFF and Drill Targets...

set PROJECT_DIR=%~dp0
set GEOTIFF=%PROJECT_DIR%CMIH 2\katghora_lithium_prospectivity.tif
set TARGETS=%PROJECT_DIR%CMIH 2\katghora_drill_targets.geojson

start "" %QGIS_BAT% "%GEOTIFF%" "%TARGETS%"
echo QGIS launched successfully!
