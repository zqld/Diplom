@echo off
chcp 65001 >nul
echo ============================================
echo   NeuroFocus - Сборка EXE
echo ============================================
echo.

echo 1. Активация виртуального окружения...
call .venv\Scripts\activate.bat

echo 2. Очистка предыдущей сборки...
if exist dist\NeuroFocus rmdir /s /q dist\NeuroFocus
if exist build\NeuroFocus rmdir /s /q build\NeuroFocus

echo 3. Запуск PyInstaller...
pyinstaller neurofocus.spec

echo.
echo ============================================
if %errorlevel% equ 0 (
    echo   Сборка завершена успешно!
    echo   Файл: dist\NeuroFocus\NeuroFocus.exe
) else (
    echo   ОШИБКА: Сборка не удалась (код %errorlevel%)
)
echo ============================================

pause
