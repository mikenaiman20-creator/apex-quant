Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
projectRoot = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = projectRoot
shell.Run """" & projectRoot & "\start_unified.bat" & """", 0, False
