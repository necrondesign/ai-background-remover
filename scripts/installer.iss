; ---------------------------------------------------------------------------
; Inno Setup Script — AI Background Remover Windows Installer
;
; Использование:
;   1. Установить Inno Setup: https://jrsoftware.org/isinfo.php
;   2. Собрать приложение: scripts\build_windows.bat
;   3. Открыть этот файл в Inno Setup Compiler → Build
; ---------------------------------------------------------------------------

#define MyAppName "AI Background Remover"
#define MyAppVersion "1.1"
#define MyAppPublisher "Stepan Andrushkevich"
#define MyAppURL "https://t.me/necrondesign"
#define MyAppExeName "AI Background Remover.exe"

[Setup]
AppId={{8B2F4A3C-9D1E-4F5A-B7C6-2E8D0A1F3B5C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=AI-Background-Remover-{#MyAppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\textures\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\AI Background Remover\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
