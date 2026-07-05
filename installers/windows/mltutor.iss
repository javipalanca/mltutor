; mltutor.iss — Script de Inno Setup para MLTutor
;
; Genera un instalador Windows autocontenido (.exe) que:
;   - Instala la distribución PyInstaller en %ProgramFiles%\MLTutor
;   - Crea acceso directo en el Menú Inicio y (opcionalmente) el Escritorio
;   - Incluye desinstalador
;
; Uso desde la raíz del proyecto (con Inno Setup 6 instalado):
;   iscc installers\windows\mltutor.iss
;
; El instalador se genera en la raíz del proyecto como:
;   mltutor-windows-x86_64-installer.exe

#define AppName      "MLTutor"
; AppVersion se inyecta en tiempo de compilación con /DAppVersion=x.y.z
; Si no se pasa explícitamente, se usa 0.0.0 como fallback seguro.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppPublisher "Javier Palanca"
#define AppURL       "https://github.com/javipalanca/mltutor"
#define AppExeName   "mltutor.exe"

[Setup]
AppId={{F3A2C1D4-8B7E-4F9A-A1B2-3C4D5E6F7A8B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}/releases

; Instalación en %ProgramFiles%\MLTutor (sin elevación si está disponible)
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Salida: raíz del proyecto (dos niveles arriba del .iss)
OutputDir=..\..\
OutputBaseFilename=mltutor-windows-x86_64-setup

; Icono del instalador
SetupIconFile=..\..\assets\icon.ico

Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
WizardResizable=yes

; No mostrar "Reboot" al finalizar
RestartIfNeededByRun=no

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; \
    Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; \
    Flags: unchecked

[Files]
; Toda la distribución PyInstaller (recursiva)
Source: "..\..\dist\mltutor\*"; \
    DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Menú Inicio
Name: "{group}\{#AppName}"; \
    Filename: "{app}\{#AppExeName}"; \
    WorkingDir: "{app}"; \
    Comment: "Iniciar MLTutor — Tutor de Machine Learning"

Name: "{group}\{cm:UninstallProgram,{#AppName}}"; \
    Filename: "{uninstallexe}"

; Escritorio (opcional, marcado por el usuario)
Name: "{commondesktop}\{#AppName}"; \
    Filename: "{app}\{#AppExeName}"; \
    WorkingDir: "{app}"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; \
    Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent
