const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  saveModel: (jobId, fileType) => ipcRenderer.invoke('save-model', jobId, fileType),
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url')
});
