@echo off
:: QuantumFlow Permanent Installation Script
:: Invented by: Gökay Yaşar Üzümcü

echo 🚀 Installing QuantumFlow for Permanent Internet Acceleration...

:: This script will create a system service to run QuantumFlow at startup,
:: ensuring your internet is always quantum-accelerated.

:: 1. Copy the accelerator to a system directory
xcopy /Y quantum_flow_accelerator.py "%SystemRoot%\System32\"

:: 2. Create a batch file to run the accelerator
(echo @echo off
echo 🚀 Starting QuantumFlow Internet Accelerator...
python "%SystemRoot%\System32\quantum_flow_accelerator.py"
) > "%SystemRoot%\System32\start_quantum_flow.bat"

:: 3. Create a system service to run at startup
sc create QuantumFlowAccelerator binPath= "%SystemRoot%\System32\start_quantum_flow.bat" start= auto
sc description QuantumFlowAccelerator "Permanently accelerates internet using quantum-inspired algorithms. Invented by Gökay Yaşar Üzümcü."

echo ✅ QuantumFlow has been permanently installed!
_pause
