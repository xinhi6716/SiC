const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let pythonBackend = null;

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = '8000';
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 980,
    minHeight: 680,
    backgroundColor: '#f5f5f7',
    title: 'RRAM Material AI Lab',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'hidden',
    trafficLightPosition: { x: 18, y: 18 },
    vibrancy: process.platform === 'darwin' ? 'under-window' : undefined,
    visualEffectState: process.platform === 'darwin' ? 'active' : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonBackend() {
  if (pythonBackend) {
    return;
  }

  const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
  const uvicornArgs = [
    '-m',
    'uvicorn',
    'api_server:app',
    '--host',
    BACKEND_HOST,
    '--port',
    BACKEND_PORT
  ];

  pythonBackend = spawn(pythonExecutable, uvicornArgs, {
    cwd: __dirname,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1'
    }
  });

  pythonBackend.stdout.on('data', (data) => {
    console.log(`[FastAPI] ${data.toString().trim()}`);
  });

  pythonBackend.stderr.on('data', (data) => {
    console.error(`[FastAPI] ${data.toString().trim()}`);
  });

  pythonBackend.on('error', (error) => {
    dialog.showErrorBox(
      'Python Backend Failed',
      `Unable to start FastAPI backend.\n\n${error.message}`
    );
  });

  pythonBackend.on('exit', (code, signal) => {
    console.log(`[FastAPI] exited with code=${code}, signal=${signal}`);
    pythonBackend = null;
  });
}

function stopPythonBackend() {
  if (!pythonBackend) {
    return;
  }

  const backendProcess = pythonBackend;
  pythonBackend = null;

  // Windows 的 kill() 對子程序樹支援有限，但 Uvicorn 在本專案中是單一 Python 子進程。
  // Electron 結束時先送 SIGTERM；若程序仍存活，再補一次 SIGKILL，避免 port 8000 被占住。
  try {
    backendProcess.kill('SIGTERM');
  } catch (error) {
    console.error(`[FastAPI] failed to terminate backend: ${error.message}`);
  }

  setTimeout(() => {
    if (!backendProcess.killed) {
      try {
        backendProcess.kill('SIGKILL');
      } catch (error) {
        console.error(`[FastAPI] failed to force-kill backend: ${error.message}`);
      }
    }
  }, 1200);
}

app.whenReady().then(() => {
  startPythonBackend();
  createMainWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonBackend();
});

app.on('quit', () => {
  stopPythonBackend();
});
