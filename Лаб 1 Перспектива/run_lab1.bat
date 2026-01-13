@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Выполнение лабораторной работы 1: Перспективные информационные технологии...
echo.
python lab1_perspective.py
if %errorlevel% neq 0 (
    echo Ошибка при выполнении lab1_perspective.py
    pause
    exit /b %errorlevel%
)
echo.
echo Генерация HTML отчета...
python generate_lab1_report.py
if %errorlevel% neq 0 (
    echo Ошибка при выполнении generate_lab1_report.py
    pause
    exit /b %errorlevel%
)
echo.
echo Лабораторная работа выполнена успешно!
echo HTML отчет: Timoshinov_E_B_Lab_1_Perspective.html
pause
