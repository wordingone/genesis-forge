const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let backendProcess;

// Application root (genesis-forge directory)
const appRoot = path.resolve(__dirname, '..', '..');

// Backend server configuration - all paths relative to appRoot
const BACKEND_CONFIG = {
  pythonExe: path.join(appRoot, 'venv', 'Scripts', 'python.exe'),
  backendPath: path.join(__dirname, '..', 'backend'),
  port: 8000,
  // MSVC path for gsplat JIT compilation (required for textures)
  // Can be overridden via VCVARS64 environment variable
  vcvarsPath: process.env.VCVARS64 || 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat'
};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#0a0a0a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    title: 'Genesis Forge - 3D Generation',
    autoHideMenuBar: true
  });

  // Load the app
  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function checkBackendRunning() {
  try {
    const http = require('http');
    return new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${BACKEND_CONFIG.port}/docs`, (res) => {
        resolve(res.statusCode === 200);
      });
      req.on('error', () => resolve(false));
      req.setTimeout(2000, () => {
        req.destroy();
        resolve(false);
      });
    });
  } catch {
    return false;
  }
}

function startBackend() {
  return new Promise(async (resolve, reject) => {
    // Check if backend is already running
    const alreadyRunning = await checkBackendRunning();
    if (alreadyRunning) {
      console.log('Backend server is already running on port', BACKEND_CONFIG.port);
      resolve();
      return;
    }

    console.log('Starting backend server...');

    // Verify Python exists
    if (!fs.existsSync(BACKEND_CONFIG.pythonExe)) {
      reject(new Error(`Python not found at ${BACKEND_CONFIG.pythonExe}`));
      return;
    }

    // Check if MSVC is available for gsplat JIT compilation
    const hasMSVC = fs.existsSync(BACKEND_CONFIG.vcvarsPath);

    if (hasMSVC) {
      console.log('Starting backend with MSVC environment (for texture rendering)...');
      // Use cmd.exe to call vcvars64.bat first, then start uvicorn
      const cmdScript = `call "${BACKEND_CONFIG.vcvarsPath}" >nul 2>&1 && "${BACKEND_CONFIG.pythonExe}" -m uvicorn main:app --host 127.0.0.1 --port ${BACKEND_CONFIG.port}`;

      backendProcess = spawn('cmd.exe', ['/c', cmdScript], {
        cwd: BACKEND_CONFIG.backendPath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          SPCONV_ALGO: 'native',
          ATTN_BACKEND: 'sdpa'
        }
      });
    } else {
      console.log('WARNING: MSVC not found - textures may render black');
      // Start backend process without MSVC
      backendProcess = spawn(BACKEND_CONFIG.pythonExe, [
        '-m', 'uvicorn',
        'main:app',
        '--host', '127.0.0.1',
        '--port', BACKEND_CONFIG.port.toString()
      ], {
        cwd: BACKEND_CONFIG.backendPath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          SPCONV_ALGO: 'native',
          ATTN_BACKEND: 'sdpa'
        }
      });
    }

    let backendReady = false;

    backendProcess.stdout.on('data', (data) => {
      const message = data.toString();
      console.log('[Backend]', message);

      if (message.includes('Application startup complete') && !backendReady) {
        backendReady = true;
        console.log('Backend server is ready!');
        resolve();
      }
    });

    backendProcess.stderr.on('data', (data) => {
      const message = data.toString();
      console.log('[Backend Info]', message);

      // Check for port already in use error
      if (message.includes('error while attempting to bind') || message.includes('10048')) {
        console.log('[Backend] Port already in use, checking if server is accessible...');
        checkBackendRunning().then(running => {
          if (running && !backendReady) {
            backendReady = true;
            console.log('Backend server detected on port', BACKEND_CONFIG.port);
            resolve();
          }
        });
      }

      if (message.includes('Application startup complete') && !backendReady) {
        backendReady = true;
        console.log('Backend server is ready!');
        resolve();
      }
    });

    backendProcess.on('error', (error) => {
      console.error('[Backend] Failed to start:', error);
      reject(error);
    });

    backendProcess.on('close', (code) => {
      console.log(`[Backend] Process exited with code ${code}`);
      if (!backendReady && code === 1) {
        // Check if backend is accessible despite exit
        checkBackendRunning().then(running => {
          if (running) {
            backendReady = true;
            console.log('Backend server detected running externally');
            resolve();
          } else {
            reject(new Error(`Backend failed to start (exit code ${code})`));
          }
        });
      }
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (!backendReady) {
        reject(new Error('Backend startup timeout (30s)'));
      }
    }, 30000);
  });
}

function stopBackend() {
  // Only stop backend if we started it (not if it was already running)
  if (backendProcess && backendProcess.pid) {
    console.log('Stopping backend server (PID:', backendProcess.pid, ')...');
    backendProcess.kill();
    backendProcess = null;
  } else if (backendProcess === null) {
    console.log('Backend was running externally, leaving it running');
  }
}

// IPC handlers
ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'bmp', 'webp'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const imagePath = result.filePaths[0];
    const imageData = fs.readFileSync(imagePath);
    const base64 = imageData.toString('base64');
    const ext = path.extname(imagePath).slice(1);
    return {
      path: imagePath,
      data: `data:image/${ext};base64,${base64}`
    };
  }
  return null;
});

ipcMain.handle('save-model', async (event, jobId, fileType) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: `trellis_model.${fileType}`,
    filters: [
      { name: fileType.toUpperCase(), extensions: [fileType] }
    ]
  });

  if (!result.canceled && result.filePath) {
    return result.filePath;
  }
  return null;
});

ipcMain.handle('get-backend-url', () => {
  return `http://127.0.0.1:${BACKEND_CONFIG.port}`;
});

// App lifecycle
app.whenReady().then(async () => {
  try {
    // Start backend first
    await startBackend();

    // Create window once backend is ready
    createWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start backend server:\n\n${error.message}\n\nPlease check that Python and all dependencies are installed correctly.`
    );
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackend();
});

// Handle crashes
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  stopBackend();
});
