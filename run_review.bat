@echo off
rem Wall plan review tool: opens the combined plan over the scan, with your drawing in red.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" review_tool.py "out\v2\Plan_murs_combine.dxf" --reference "\\GENAS\Public\ARCHIVE2026\132_HMILIT\132_BdTerrain\132_SCAN19112026\Public (GENAS).dxf"
