@echo off
REM =============================================================================
REM QGIS CLI & OSGeo4W Environment Launcher
REM Opens the command-line interface with full GDAL, PROJ, and PyQGIS bindings
REM =============================================================================

set OSGEO_DIR=
for /d %%i in ("C:\Program Files\QGIS*") do (
    if exist "%%i\OSGeo4W.bat" set OSGEO_DIR="%%i"
)
if not defined OSGEO_DIR (
    if exist "C:\OSGeo4W\OSGeo4W.bat" set OSGEO_DIR="C:\OSGeo4W"
)

if not defined OSGEO_DIR (
    echo [ERROR] QGIS installation not found.
    pause
    exit /b 1
)

echo Starting QGIS / OSGeo4W CLI Environment from %OSGEO_DIR%...
cd /d %~dp0
call %OSGEO_DIR%\OSGeo4W.bat
