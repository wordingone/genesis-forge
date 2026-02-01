import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// State
let backendUrl = '';
let currentMode = 'text';
let imageFile = null;
let jobs = [];
let pollInterval = null;
let currentModel = null;

// Three.js viewer
let scene, camera, renderer, controls, currentObject;

// Initialize
async function init() {
  try {
    // Get backend URL from Electron API or fallback to localhost
    if (window.electronAPI && window.electronAPI.getBackendUrl) {
      backendUrl = await window.electronAPI.getBackendUrl();
    } else {
      // Fallback for development/testing
      backendUrl = 'http://127.0.0.1:8000';
      console.log('Using fallback backend URL:', backendUrl);
    }

    console.log('[Init] Backend URL:', backendUrl);
    console.log('[Init] Starting app...');
    initViewer();
    setupEventListeners();
    startPolling();
    console.log('[Init] Polling started (500ms interval)');
  } catch (error) {
    console.error('Initialization error:', error);
    alert('Failed to initialize application: ' + error.message);
  }
}

// Three.js 3D Viewer
function initViewer() {
  const container = document.getElementById('viewer3d');
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  // Camera
  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.set(2, 2, 2);
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.8; // Increased for brighter models
  container.appendChild(renderer.domElement);

  // Orbit Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 0.5;
  controls.maxDistance = 10;
  controls.maxPolarAngle = Math.PI;

  // Efficient lighting setup for PBR materials
  // Bright ambient light for base illumination
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  // Use HemisphereLight for cheap global illumination
  const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
  scene.add(hemisphereLight);

  // Strong key directional light
  const directionalLight = new THREE.DirectionalLight(0xffffff, 2.0);
  directionalLight.position.set(5, 10, 7.5);
  directionalLight.castShadow = false; // Disable shadows for performance
  scene.add(directionalLight);

  // Fill light from opposite side
  const fillLight = new THREE.DirectionalLight(0xffffff, 1.0);
  fillLight.position.set(-5, 3, -7);
  scene.add(fillLight);

  // Grid
  const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
  scene.add(gridHelper);

  // Axes helper
  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  // Handle window resize
  window.addEventListener('resize', onWindowResize, false);

  // Animation loop
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById('viewer3d');
  const width = container.clientWidth;
  const height = container.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function loadModel(url, type = 'glb') {
  // Remove existing model
  if (currentObject) {
    scene.remove(currentObject);
    currentObject = null;
  }

  const viewerInfo = document.getElementById('viewerInfo');
  viewerInfo.textContent = 'Loading model...';

  if (type === 'glb') {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        currentObject = gltf.scene;

        console.log('GLB loaded, inspecting materials and textures:');

        // Fix PBR materials - ensure proper rendering
        let meshCount = 0;
        currentObject.traverse((child) => {
          if (child.isMesh) {
            meshCount++;

            // Ensure material renders correctly
            if (child.material) {
              console.log(`Mesh ${meshCount} material:`, {
                type: child.material.type,
                color: child.material.color ? child.material.color.getHexString() : 'none',
                hasMap: !!child.material.map,
                hasNormalMap: !!child.material.normalMap,
                hasRoughnessMap: !!child.material.roughnessMap,
                hasMetalnessMap: !!child.material.metalnessMap,
                metalness: child.material.metalness,
                roughness: child.material.roughness
              });

              // CRITICAL: If the material has no base color texture but has a color, ensure it's not black
              if (!child.material.map && child.material.color) {
                const colorHex = child.material.color.getHexString();
                if (colorHex === '000000') {
                  console.warn('Material has black color and no texture! Setting to white as fallback.');
                  child.material.color.setHex(0xcccccc); // Light gray fallback
                }
              }

              // Enable proper PBR rendering
              child.material.needsUpdate = true;

              // Force side rendering (helps with transparency)
              child.material.side = THREE.DoubleSide;

              // Ensure metalness and roughness are valid
              if (child.material.metalness !== undefined) {
                child.material.metalness = Math.max(0, Math.min(1, child.material.metalness));
              }
              if (child.material.roughness !== undefined) {
                child.material.roughness = Math.max(0.1, Math.min(1, child.material.roughness));
              }

              // Enable proper texture encoding
              if (child.material.map) {
                console.log('Base color texture found! Size:', child.material.map.image?.width, 'x', child.material.map.image?.height);
                child.material.map.encoding = THREE.sRGBEncoding;
                child.material.map.needsUpdate = true;
              } else {
                console.warn('No base color texture (map) on this mesh!');
              }

              if (child.material.normalMap) {
                console.log('Normal map found');
                child.material.normalMap.encoding = THREE.LinearEncoding;
              }

              if (child.material.roughnessMap) {
                console.log('Roughness map found');
                child.material.roughnessMap.encoding = THREE.LinearEncoding;
              }

              if (child.material.metalnessMap) {
                console.log('Metalness map found');
                child.material.metalnessMap.encoding = THREE.LinearEncoding;
              }

              if (child.material.emissiveMap) {
                child.material.emissiveMap.encoding = THREE.sRGBEncoding;
              }
            }

            // Enable shadows (receiving only for performance)
            child.castShadow = false;
            child.receiveShadow = false;
          }
        });

        // Center and scale model
        const box = new THREE.Box3().setFromObject(currentObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;

        currentObject.scale.multiplyScalar(scale);
        currentObject.position.sub(center.multiplyScalar(scale));

        scene.add(currentObject);

        viewerInfo.textContent = `Model loaded: ${Math.round(maxDim * 100) / 100} units, ${meshCount} meshes`;
      },
      (progress) => {
        const percent = (progress.loaded / progress.total * 100).toFixed(0);
        viewerInfo.textContent = `Loading model... ${percent}%`;
      },
      (error) => {
        console.error('Error loading GLB:', error);
        console.error('URL attempted:', url);
        viewerInfo.textContent = `Error: ${error.message || 'Failed to load model'}`;
      }
    );
  } else if (type === 'ply') {
    const loader = new PLYLoader();
    loader.load(
      url,
      (geometry) => {
        geometry.computeVertexNormals();

        const material = new THREE.PointsMaterial({
          size: 0.01,
          vertexColors: true
        });

        currentObject = new THREE.Points(geometry, material);

        // Center and scale
        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;

        currentObject.scale.multiplyScalar(scale);
        currentObject.position.sub(center.multiplyScalar(scale));

        scene.add(currentObject);

        viewerInfo.textContent = `Point cloud loaded: ${geometry.attributes.position.count} points`;
      },
      undefined,
      (error) => {
        console.error('Error loading PLY:', error);
        viewerInfo.textContent = 'Error loading point cloud';
      }
    );
  }
}

function resetCamera() {
  camera.position.set(2, 2, 2);
  camera.lookAt(0, 0, 0);
  controls.target.set(0, 0, 0);
  controls.update();
}

function toggleWireframe() {
  if (currentObject) {
    currentObject.traverse((child) => {
      if (child.isMesh) {
        child.material.wireframe = !child.material.wireframe;
      }
    });
  }
}

// Event Listeners
function setupEventListeners() {
  // Mode toggle
  document.getElementById('textModeBtn').addEventListener('click', () => setMode('text'));
  document.getElementById('imageModeBtn').addEventListener('click', () => setMode('image'));

  // Image upload
  document.getElementById('imageUpload').addEventListener('click', async () => {
    const result = await window.electronAPI.selectImage();
    if (result) {
      imageFile = result;
      document.getElementById('uploadPlaceholder').classList.add('hidden');
      const preview = document.getElementById('imagePreview');
      preview.src = result.data;
      preview.classList.remove('hidden');
    }
  });

  // Advanced settings toggle - shows the appropriate panel based on current mode
  document.getElementById('advancedToggle').addEventListener('click', () => {
    if (currentMode === 'text') {
      document.getElementById('textAdvancedSettings').classList.toggle('hidden');
      document.getElementById('imageAdvancedSettings').classList.add('hidden');
    } else {
      document.getElementById('imageAdvancedSettings').classList.toggle('hidden');
      document.getElementById('textAdvancedSettings').classList.add('hidden');
    }
  });

  // Range sliders for TEXT-TO-3D (TRELLIS.1)
  const textSliders = [
    { id: 'textSparseSteps', valueId: 'textSparseStepsValue' },
    { id: 'textSparseCfg', valueId: 'textSparseCfgValue' },
    { id: 'textSlatSteps', valueId: 'textSlatStepsValue' },
    { id: 'textSlatCfg', valueId: 'textSlatCfgValue' },
    { id: 'textSimplify', valueId: 'textSimplifyValue', format: v => `${Math.round(v * 100)}%` }
  ];

  // Range sliders for IMAGE-TO-3D (TRELLIS.2)
  // Matches official HuggingFace demo parameters
  const imageSliders = [
    // Stage 1: Sparse Structure
    { id: 'imageSparseSteps', valueId: 'imageSparseStepsValue', format: v => Math.round(v) },
    { id: 'imageSparseGuidance', valueId: 'imageSparseGuidanceValue' },
    { id: 'imageSparseRescale', valueId: 'imageSparseRescaleValue' },
    { id: 'imageSparseRescaleT', valueId: 'imageSparseRescaleTValue', format: v => Math.round(v) },
    // Stage 2: Shape Generation
    { id: 'imageShapeSteps', valueId: 'imageShapeStepsValue', format: v => Math.round(v) },
    { id: 'imageShapeGuidance', valueId: 'imageShapeGuidanceValue' },
    { id: 'imageShapeRescale', valueId: 'imageShapeRescaleValue' },
    { id: 'imageShapeRescaleT', valueId: 'imageShapeRescaleTValue', format: v => Math.round(v) },
    // Stage 3: Material Generation
    { id: 'imageTexSteps', valueId: 'imageTexStepsValue', format: v => Math.round(v) },
    { id: 'imageTexGuidance', valueId: 'imageTexGuidanceValue' },
    { id: 'imageTexRescale', valueId: 'imageTexRescaleValue' },
    { id: 'imageTexRescaleT', valueId: 'imageTexRescaleTValue', format: v => Math.round(v) },
    // Output settings
    { id: 'imageDecimation', valueId: 'imageDecimationValue', format: v => Math.round(v) },
    { id: 'imageSimplify', valueId: 'imageSimplifyValue', format: v => `${Math.round(v * 100)}%` }
  ];

  // Setup all sliders
  [...textSliders, ...imageSliders].forEach(({ id, valueId, format }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(valueId);
    if (slider && valueDisplay) {
      slider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        valueDisplay.textContent = format ? format(value) : value.toFixed(1);
      });
    }
  });

  // Generate button
  document.getElementById('generateBtn').addEventListener('click', generateModel);

  // Viewer controls
  document.getElementById('resetCameraBtn').addEventListener('click', resetCamera);
  document.getElementById('wireframeBtn').addEventListener('click', toggleWireframe);
}

function setMode(mode) {
  currentMode = mode;

  // Update buttons
  document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));

  // Get settings panels
  const textSettings = document.getElementById('textAdvancedSettings');
  const imageSettings = document.getElementById('imageAdvancedSettings');

  // Check if any settings panel was visible
  const textWasVisible = textSettings && !textSettings.classList.contains('hidden');
  const imageWasVisible = imageSettings && !imageSettings.classList.contains('hidden');
  const anySettingsVisible = textWasVisible || imageWasVisible;

  if (mode === 'text') {
    document.getElementById('textModeBtn').classList.add('active');
    document.getElementById('textInput').classList.remove('hidden');
    document.getElementById('imageInput').classList.add('hidden');

    // Switch settings panels if settings were visible
    if (anySettingsVisible && textSettings) {
      textSettings.classList.remove('hidden');
    }
    if (imageSettings) {
      imageSettings.classList.add('hidden');
    }
  } else {
    document.getElementById('imageModeBtn').classList.add('active');
    document.getElementById('textInput').classList.add('hidden');
    document.getElementById('imageInput').classList.remove('hidden');

    // Switch settings panels if settings were visible
    if (anySettingsVisible && imageSettings) {
      imageSettings.classList.remove('hidden');
    }
    if (textSettings) {
      textSettings.classList.add('hidden');
    }
  }
}

// Notification System
function showNotification(message, type = 'info') {
  const viewerInfo = document.getElementById('viewerInfo');
  const colors = {
    success: '#4ade80',
    error: '#ff6b6b',
    info: '#60a5fa',
    warning: '#fbbf24'
  };

  viewerInfo.textContent = message;
  viewerInfo.style.color = colors[type];
  viewerInfo.style.opacity = '1';

  // Auto-hide after 4 seconds
  setTimeout(() => {
    viewerInfo.style.opacity = '0.5';
    setTimeout(() => {
      if (viewerInfo.textContent === message) {
        viewerInfo.textContent = 'No model loaded. Generate a 3D model to view it here.';
        viewerInfo.style.color = '';
        viewerInfo.style.opacity = '';
      }
    }, 500);
  }, 4000);
}

// API Functions
async function generateModel() {
  console.log('Generate button clicked');
  console.log('Backend URL:', backendUrl);
  console.log('Current mode:', currentMode);

  // Disable button during request
  const btn = document.getElementById('generateBtn');
  btn.disabled = true;
  btn.innerHTML = '<svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24" class="spinner"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" opacity="0.25"/><path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/></svg> Submitting...';

  try {
    let response;

    if (currentMode === 'text') {
      // TEXT-TO-3D: TRELLIS.1 parameters
      const params = {
        seed: parseInt(document.getElementById('textSeedInput').value),
        sparse_steps: parseInt(document.getElementById('textSparseSteps').value),
        sparse_cfg: parseFloat(document.getElementById('textSparseCfg').value),
        slat_steps: parseInt(document.getElementById('textSlatSteps').value),
        slat_cfg: parseFloat(document.getElementById('textSlatCfg').value),
        simplify: parseFloat(document.getElementById('textSimplify').value),
        texture_size: parseInt(document.getElementById('textTextureSize').value)
      };

      const prompt = document.getElementById('promptInput').value.trim();
      if (!prompt) {
        console.warn('No prompt entered');
        btn.disabled = false;
        btn.innerHTML = '<svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/></svg> Generate 3D Model';
        return;
      }

      console.log('Submitting text-to-3D request:', prompt, params);

      response = await fetch(`${backendUrl}/api/generate/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, ...params })
      });
    } else {
      // IMAGE-TO-3D: TRELLIS.2 parameters (3-stage pipeline)
      // Matches official HuggingFace demo defaults
      const params = {
        seed: parseInt(document.getElementById('imageSeedInput').value),
        pipeline_version: 'v2',  // Force TRELLIS.2
        resolution: parseInt(document.getElementById('imageResolution').value),
        // Stage 1: Sparse Structure
        sparse_steps: parseInt(document.getElementById('imageSparseSteps').value),
        sparse_cfg: parseFloat(document.getElementById('imageSparseGuidance').value),
        guidance_rescale_sparse: parseFloat(document.getElementById('imageSparseRescale').value),
        rescale_t_sparse: parseFloat(document.getElementById('imageSparseRescaleT').value),
        // Stage 2: Shape Generation
        slat_steps: parseInt(document.getElementById('imageShapeSteps').value),
        slat_cfg: parseFloat(document.getElementById('imageShapeGuidance').value),
        guidance_rescale_shape: parseFloat(document.getElementById('imageShapeRescale').value),
        rescale_t_shape: parseFloat(document.getElementById('imageShapeRescaleT').value),
        // Stage 3: Material Generation
        tex_steps: parseInt(document.getElementById('imageTexSteps').value),
        tex_cfg: parseFloat(document.getElementById('imageTexGuidance').value),
        guidance_rescale_material: parseFloat(document.getElementById('imageTexRescale').value),
        rescale_t_material: parseFloat(document.getElementById('imageTexRescaleT').value),
        // Output settings
        decimation_target: parseInt(document.getElementById('imageDecimation').value),
        simplify: parseFloat(document.getElementById('imageSimplify').value),
        texture_size: parseInt(document.getElementById('imageTextureSize').value)
      };

      if (!imageFile) {
        console.warn('No image uploaded');
        btn.disabled = false;
        btn.innerHTML = '<svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/></svg> Generate 3D Model';
        return;
      }

      console.log('Submitting image-to-3D request (TRELLIS.2):', params);

      const formData = new FormData();
      const blob = await fetch(imageFile.data).then(r => r.blob());
      formData.append('file', blob, 'image.png');
      Object.entries(params).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });

      response = await fetch(`${backendUrl}/api/generate/image`, {
        method: 'POST',
        body: formData
      });
    }

    console.log('Response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Response error:', errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log('[Generate] Job created:', data.job_id, '| Backend:', backendUrl);
    // Immediately fetch jobs to show the new job
    await fetchJobs();

  } catch (error) {
    console.error('Generation failed:', error);
    // Show error in viewer info instead of alert
    document.getElementById('viewerInfo').textContent = `Error: ${error.message}`;
    document.getElementById('viewerInfo').style.color = '#ff6b6b';
    setTimeout(() => {
      document.getElementById('viewerInfo').textContent = 'No model loaded. Generate a 3D model to view it here.';
      document.getElementById('viewerInfo').style.color = '';
    }, 5000);
  } finally {
    // Re-enable button
    btn.disabled = false;
    btn.innerHTML = '<svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/></svg> Generate 3D Model';
  }
}

async function fetchJobs() {
  try {
    const response = await fetch(`${backendUrl}/api/jobs`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    const prevJobs = jobs;
    jobs = data.jobs || [];

    // Debug: Log ALL job statuses on every fetch (temporary)
    const statuses = jobs.map(j => `${j.job_id.slice(0,8)}:${j.status}`).join(', ');
    console.log(`[fetchJobs] ${jobs.length} jobs: ${statuses || 'none'}`);

    const activeJobs = jobs.filter(j => j.status === 'processing' || j.status === 'queued');
    if (activeJobs.length > 0) {
      console.log('[fetchJobs] ACTIVE:', activeJobs.map(j => `${j.job_id.slice(0,8)}:${j.status}:${Math.round(j.progress*100)}%`).join(', '));
    }

    renderJobs();
  } catch (error) {
    console.error('[fetchJobs] FAILED:', error.message, '| URL:', backendUrl);
  }
}

function renderJobs() {
  const activeContainer = document.getElementById('activeJobsList');
  const completedContainer = document.getElementById('completedJobsList');

  if (!activeContainer) {
    console.error('[renderJobs] activeJobsList element NOT FOUND');
    return;
  }

  // Split jobs into active (processing/queued) and completed (completed/failed)
  const activeJobs = jobs.filter(job => job.status === 'processing' || job.status === 'queued');
  const completedJobs = jobs.filter(job => job.status === 'completed' || job.status === 'failed');

  // Debug: log when we're about to render active jobs
  if (activeJobs.length > 0) {
    console.log('[renderJobs] Rendering', activeJobs.length, 'active job(s) to sidebar');
  }

  // Render active jobs in sidebar
  if (activeJobs.length === 0) {
    activeContainer.innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24" style="opacity: 0.3;">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
        <p>No active jobs</p>
      </div>
    `;
  } else {
    activeContainer.innerHTML = activeJobs.map(job => `
      <div class="job-card compact">
        <div class="job-header">
          <div class="job-info">
            ${getStatusIcon(job.status)}
            <div>
              <h4>${job.type === 'text_to_3d' ? 'Text-to-3D' : 'Image-to-3D'}</h4>
              <p class="job-message">${job.message}</p>
            </div>
          </div>
        </div>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${job.progress * 100}%"></div>
          </div>
          <div class="progress-text">${Math.round(job.progress * 100)}% complete</div>
        </div>
      </div>
    `).join('');
  }

  // Render completed jobs in main content
  if (completedJobs.length === 0) {
    completedContainer.innerHTML = `
      <div class="empty-state">
        <p>No completed jobs yet</p>
      </div>
    `;
  } else {
    completedContainer.innerHTML = completedJobs.map(job => `
      <div class="job-card">
        <div class="job-header">
          <div class="job-info">
            ${getStatusIcon(job.status)}
            <div>
              <h3>${job.type === 'text_to_3d' ? 'Text-to-3D' : 'Image-to-3D'}</h3>
              <p class="job-message">${job.message}</p>
            </div>
          </div>
        </div>

        ${job.status === 'completed' && job.outputs ? `
          <div class="job-actions">
            ${job.outputs.glb ? `
              <button class="job-btn" onclick="viewModel('${job.job_id}', 'glb')">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                </svg>
                View GLB
              </button>
            ` : `
              <span class="job-btn disabled" title="${job.glb_error || 'GLB export failed'}" style="opacity: 0.5; cursor: not-allowed; color: #ff6b6b;">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
                GLB Failed
              </span>
            `}
            <button class="job-btn" onclick="viewModel('${job.job_id}', 'ply')">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
              </svg>
              View PLY
            </button>
            ${job.outputs.glb ? `
              <button class="job-btn" onclick="downloadModel('${job.job_id}', 'glb')">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                </svg>
                Save GLB
              </button>
            ` : ''}
            <button class="job-btn" onclick="downloadModel('${job.job_id}', 'ply')">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
              </svg>
              Save PLY
            </button>
            <button class="job-btn danger" onclick="deleteJob('${job.job_id}')">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
              </svg>
              Delete
            </button>
          </div>
        ` : ''}

        ${job.status === 'failed' && job.error ? `
          <div style="padding: 0.75rem; background: rgba(255, 107, 107, 0.1); border-radius: 6px; color: #ff6b6b; font-size: 0.875rem; margin-top: 0.75rem;">
            <strong>Error:</strong> ${job.error}
          </div>
        ` : ''}
      </div>
    `).join('');
  }
}

function getStatusIcon(status) {
  const icons = {
    completed: '<svg width="20" height="20" fill="#4ade80" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>',
    failed: '<svg width="20" height="20" fill="#ff6b6b" viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
    processing: '<svg width="20" height="20" fill="#60a5fa" viewBox="0 0 24 24" class="spinner"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" opacity="0.25"/><path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/></svg>',
    queued: '<svg width="20" height="20" fill="#fbbf24" viewBox="0 0 24 24"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/></svg>'
  };
  return icons[status] || icons.queued;
}

function viewModel(jobId, type) {
  const url = `${backendUrl}/api/download/${jobId}/${type}`;
  loadModel(url, type);
  currentModel = { jobId, type };
}

async function downloadModel(jobId, type) {
  const savePath = await window.electronAPI.saveModel(jobId, type);
  if (savePath) {
    try {
      showNotification('Downloading...', 'info');
      const response = await fetch(`${backendUrl}/api/download/${jobId}/${type}`);
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      require('fs').writeFileSync(savePath, buffer);
      showNotification(`Model saved successfully!`, 'success');
    } catch (error) {
      console.error('Download failed:', error);
      showNotification(`Failed to save: ${error.message}`, 'error');
    }
  }
}

async function deleteJob(jobId) {
  try {
    const response = await fetch(`${backendUrl}/api/jobs/${jobId}`, {
      method: 'DELETE'
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    // If viewing this model, clear viewer
    if (currentModel && currentModel.jobId === jobId) {
      if (currentObject) {
        scene.remove(currentObject);
        currentObject = null;
      }
      document.getElementById('viewerInfo').textContent = 'Model deleted';
      currentModel = null;
    }

    showNotification('Job deleted', 'success');
    fetchJobs();
  } catch (error) {
    console.error('Delete failed:', error);
    showNotification(`Failed to delete: ${error.message}`, 'error');
  }
}

function startPolling() {
  fetchJobs();
  // Poll every 500ms for responsive progress bar updates
  pollInterval = setInterval(fetchJobs, 500);
}

// Expose functions to global scope for HTML onclick handlers
window.viewModel = viewModel;
window.downloadModel = downloadModel;
window.deleteJob = deleteJob;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Add CSS for spinner animation
const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .spinner {
    animation: spin 1s linear infinite;
  }
`;
document.head.appendChild(style);
