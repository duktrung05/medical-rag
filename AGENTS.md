# Terminal behavior

- On Windows, run task commands without opening a visible CMD or PowerShell window.
- When starting a background server or helper with `Start-Process`, use `-WindowStyle Hidden` and redirect stdout/stderr to files when logs are needed.
- Keep the background process running until the task requires stopping it; do not repeatedly open and close console windows for polling.
- Report relevant errors or the location of captured logs in the chat instead of showing a transient console window.
