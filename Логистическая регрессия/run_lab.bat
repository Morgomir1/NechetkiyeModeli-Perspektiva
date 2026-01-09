@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Запуск лабораторной работы по логистической регрессии...
echo.
python run_lab.py
echo.
pause

