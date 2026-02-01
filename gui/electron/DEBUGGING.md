# TRELLIS GUI - Debugging Black Models & Progress Bar

## Issues Addressed

1. ✅ Progress bar not showing percentage text
2. 🔍 Text-to-3D models appearing black (debugging added)
3. 🔍 Image-to-3D verification (debugging added)

---

## Changes Made

### 1. Progress Bar - Percentage Display Added ✅

**File**: `app.js` (lines 473-478)

**Before**:
```javascript
<div class="progress-bar">
  <div class="progress-fill" style="width: ${job.progress * 100}%"></div>
</div>
```

**After**:
```javascript
<div class="progress-container">
  <div class="progress-bar">
    <div class="progress-fill" style="width: ${job.progress * 100}%"></div>
  </div>
  <div class="progress-text">${Math.round(job.progress * 100)}% complete</div>
</div>
```

**Result**: Progress bar now shows "X% complete" below the visual bar

---

### 2. Enhanced Lighting ✅

**File**: `app.js` (lines 72-90)

**Changes**:
- Added `AmbientLight(0xffffff, 0.8)` for base illumination
- Increased `HemisphereLight` intensity to 1.5 (was 1.0)
- Increased key `DirectionalLight` to 2.0 (was 1.5)
- Added `fillLight` (DirectionalLight) from opposite side (1.0 intensity)
- Increased `toneMappingExposure` to 1.8 (was 1.0)

**Result**: Much brighter scene overall, should prevent dark models

---

### 3. Backend Texture Debugging 🔍

**File**: `backend/main.py` (lines 167-203, 276-299)

**Added logging**:
```python
print(f"[TEXTURE DEBUG] Starting GLB export with texture_size={params.texture_size}")
print(f"[TEXTURE DEBUG] Gaussian representation type: {type(outputs['gaussian'][0])}")
print(f"[TEXTURE DEBUG] Mesh type: {type(outputs['mesh'][0])}")
# ... after export:
print(f"[TEXTURE DEBUG] GLB file size: {file_size_mb:.2f} MB")
print(f"[TEXTURE DEBUG] GLB has materials: {len(glb.visual.material.materials)}")
```

**What to check**:
1. **File size**: Should be >1MB if textures are baked
   - <1MB = No textures or low-res textures
   - >5MB = Full 1024x1024 textures included

2. **Material count**: Should be >0
   - 0 = No materials assigned
   - >0 = Materials present

**How to verify**: Run backend and watch console during generation

---

### 4. Frontend Material Debugging 🔍

**File**: `app.js` (lines 140-217)

**Added detailed logging**:
```javascript
console.log(`Mesh ${meshCount} material:`, {
  type: child.material.type,
  color: child.material.color.getHexString(),
  hasMap: !!child.material.map,
  hasNormalMap: !!child.material.normalMap,
  hasRoughnessMap: !!child.material.roughnessMap,
  hasMetalnessMap: !!child.material.metalnessMap,
  metalness: child.material.metalness,
  roughness: child.material.roughness
});

if (child.material.map) {
  console.log('Base color texture found! Size:', child.material.map.image?.width, 'x', child.material.map.image?.height);
} else {
  console.warn('No base color texture (map) on this mesh!');
}
```

**Added fallback**:
```javascript
// If material has no texture and is black, set to light gray
if (!child.material.map && child.material.color) {
  const colorHex = child.material.color.getHexString();
  if (colorHex === '000000') {
    console.warn('Material has black color and no texture! Setting to white as fallback.');
    child.material.color.setHex(0xcccccc);
  }
}
```

**What to check in browser console**:
1. **hasMap: true** = Base color texture present ✅
2. **hasMap: false** = No texture (PROBLEM if model should have texture)
3. **color: "000000"** = Black material (PROBLEM)
4. **color: "cccccc"** = Fallback gray (original was black)

---

## Testing Procedure

### Step 1: Generate Text-to-3D Model

1. Start the GUI: `.\start_gui.ps1`
2. Enter prompt: "A wooden chair shaped like an avocado"
3. Click "Generate 3D Model"
4. **Watch sidebar**: Progress bar should show "X% complete"

### Step 2: Check Backend Console

Look for these logs:
```
[TEXTURE DEBUG] Starting GLB export with texture_size=1024
[TEXTURE DEBUG] Gaussian representation type: <class '...Gaussian'>
[TEXTURE DEBUG] Mesh type: <class '...MeshExtractResult'>
[TEXTURE DEBUG] GLB export complete
[TEXTURE DEBUG] GLB file size: X.XX MB  <-- Should be >1MB
```

**Expected**: File size >1MB (textures included)

### Step 3: Load Model in Viewer

1. Click "View GLB" when complete
2. Open browser DevTools (F12 or right-click > Inspect)
3. Go to Console tab

Look for these logs:
```javascript
GLB loaded, inspecting materials and textures:
Mesh 1 material: {
  type: "MeshStandardMaterial",
  color: "xxxxxx",  <-- Should NOT be "000000"
  hasMap: true,     <-- Should be true
  hasNormalMap: ...,
  metalness: 0.X,
  roughness: 0.X
}
Base color texture found! Size: 1024 x 1024  <-- Texture loaded!
```

**Good Signs**:
- ✅ `hasMap: true` = Texture present
- ✅ `color: "xxxxxx"` (not black)
- ✅ `Size: 1024 x 1024` = Full resolution texture

**Bad Signs**:
- ❌ `hasMap: false` = No texture in GLB file
- ❌ `color: "000000"` = Black material
- ❌ `No base color texture (map) on this mesh!` = Texture missing

### Step 4: Visual Check

**Model should be**:
- Bright and visible
- Colored (not solid black/gray)
- Has surface detail (texture visible)

**If still black**:
1. Check backend console for texture baking errors
2. Check frontend console for `hasMap: false`
3. Check GLB file size (<1MB = no textures)

---

## Diagnosis Chart

| Symptom | Backend Log | Frontend Log | Diagnosis |
|---------|-------------|--------------|-----------|
| Black model | File size <1MB | `hasMap: false` | **Texture not baked** - TRELLIS issue |
| Black model | File size >1MB | `hasMap: true` | **Texture not rendering** - Three.js issue |
| Black model | File size >1MB | `hasMap: false` | **Texture in file but not loaded** - GLB format issue |
| Gray model | File size <1MB | `color: "cccccc"` | **Fallback triggered** - Original was black |
| Bright colored model | File size >1MB | `hasMap: true`, texture size shown | ✅ **Working correctly** |

---

## Known TRELLIS Texture Baking Process

From official TRELLIS code:

```python
# trellis/utils/postprocessing_utils.py
def to_glb(app_rep, mesh, texture_size=1024, ...):
    """
    Bakes appearance (color from Gaussian/Strivec) onto mesh UV map
    1. Unwrap mesh UVs
    2. Sample appearance representation at each UV coordinate
    3. Render to texture image (texture_size x texture_size)
    4. Embed texture in GLB as baseColorTexture
    """
```

**This should automatically create textures**. If it doesn't:
- Check if `outputs['gaussian'][0]` has color data
- Check if mesh has valid UV coordinates
- Check if texture_size parameter is being passed correctly

---

## Potential Issues & Fixes

### Issue 1: Texture Baking Fails Silently

**Symptom**: GLB file <1MB, backend shows no errors

**Fix**: Check TRELLIS logs for UV unwrapping failures
```python
# In backend, add before to_glb():
print(f"Mesh has UVs: {mesh.uv is not None}")
print(f"Gaussian has colors: {hasattr(outputs['gaussian'][0], 'colors')}")
```

### Issue 2: GLB Exports with Vertex Colors Instead of Texture

**Symptom**: GLB file small, colors present but not as texture

**Fix**: TRELLIS might be using vertex colors. Check GLB in external viewer:
- Blender: File > Import > glTF 2.0
- Check if Shading > Material has "Base Color" image texture

### Issue 3: Three.js Not Loading Embedded Textures

**Symptom**: Backend logs show textures, frontend shows `hasMap: false`

**Fix**: GLB might use non-standard texture format. Try:
```javascript
// In GLTFLoader callback, force texture load:
if (child.material && !child.material.map) {
  // Check if texture is in GLB but not assigned
  if (gltf.parser && gltf.parser.json.images) {
    console.log('GLB has images:', gltf.parser.json.images.length);
  }
}
```

---

## Next Steps

1. **Run test generation** with new logging
2. **Collect all logs** (backend + frontend console)
3. **Share results** to diagnose exact issue:
   - Backend: `[TEXTURE DEBUG]` lines
   - Frontend: Material inspection logs
   - GLB file size

4. **If textures are baking correctly** (file size >1MB, `hasMap: true`):
   - Model should now be visible with brighter lighting
   - Issue was lighting, not textures

5. **If textures are NOT baking** (file size <1MB, `hasMap: false`):
   - TRELLIS postprocessing issue
   - May need to investigate UV unwrapping or appearance sampling

---

## Updated Files

1. `app.js` - Progress percentage, lighting, material debugging
2. `backend/main.py` - Texture baking debugging
3. `DEBUGGING.md` - This file

All changes are **non-breaking** - existing functionality preserved, debugging added.

---

## Quick Reference: Expected vs Actual

**Expected Workflow**:
```
TRELLIS generates Gaussian splats with color
  ↓
to_glb() bakes colors onto mesh UV map
  ↓
GLB file created with embedded texture (>1MB)
  ↓
GLTFLoader loads GLB with texture
  ↓
Three.js renders textured material
  ↓
Model visible with colors
```

**If black**:
- ❌ Step 1-2: Gaussian has no color / Baking failed
- ❌ Step 3: GLB has no texture (<1MB)
- ❌ Step 4: Texture not loaded (`hasMap: false`)
- ❌ Step 5: Lighting too dark / Material black

**Our fixes address**:
- ✅ Step 5: Lighting increased significantly
- 🔍 Step 1-4: Comprehensive logging to identify break point

---

Generated: 2026-01-19
Status: DEBUGGING ENABLED - Run test and check logs
