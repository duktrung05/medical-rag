# Terminal behavior

- On Windows, run every task command without opening a visible CMD, PowerShell,
  Windows Terminal, or console-host window.
- Run short commands such as `git`, `rg`, file inspection, tests, and one-off
  Python checks in the existing integrated terminal/tool session. Do not launch
  them through `Start-Process`, `cmd /c start`, `start`, `wt.exe`, or a new shell
  window.
- Reuse one terminal/session for sequential commands. Batch independent
  read-only checks into one invocation when practical instead of repeatedly
  creating and closing shell processes.
- Use `Start-Process` only for a process that must continue in the background,
  such as a server or long-running helper. Always pass `-WindowStyle Hidden`
  and `-PassThru`, redirect stdout and stderr to stable log files, and retain
  the process ID so the same process can be reused and stopped later.
- Keep a background process running until the task requires stopping it. Poll
  its PID or captured log from the existing session; do not spawn transient
  console windows for polling.
- VS Code tasks must use the integrated terminal with a shared or dedicated
  reusable panel and must not reveal the terminal on successful runs.
- Report relevant errors and log paths in chat instead of showing a transient
  console window.
