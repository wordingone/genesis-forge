import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import axios from 'axios'
import {
  Loader2,
  Download,
  Trash2,
  Image as ImageIcon,
  Type,
  Settings,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://localhost:8000'

interface Job {
  job_id: string
  type: string
  status: string
  progress: number
  message: string
  outputs?: {
    glb?: string
    ply?: string
    preview?: string
  }
  error?: string
  created_at: string
  updated_at: string
  params: any
}

function App() {
  const [mode, setMode] = useState<'text' | 'image'>('text')
  const [prompt, setPrompt] = useState('A wooden chair shaped like an avocado')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  // Advanced settings
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [seed, setSeed] = useState(42)
  const [sparseSteps, setSparseSteps] = useState(12)
  const [sparseCfg, setSparseCfg] = useState(7.5)
  const [slatSteps, setSlatSteps] = useState(12)
  const [slatCfg, setSlatCfg] = useState(7.5)
  const [simplify, setSimplify] = useState(0.95)
  const [textureSize, setTextureSize] = useState(1024)

  // Poll jobs
  const { data: jobsData, refetch: refetchJobs } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const res = await axios.get(`${API_BASE}/api/jobs`)
      return res.data.jobs as Job[]
    },
    refetchInterval: 2000,
  })

  // Text-to-3D mutation
  const textMutation = useMutation({
    mutationFn: async () => {
      const res = await axios.post(`${API_BASE}/api/generate/text`, {
        prompt,
        seed,
        sparse_steps: sparseSteps,
        sparse_cfg: sparseCfg,
        slat_steps: slatSteps,
        slat_cfg: slatCfg,
        simplify,
        texture_size: textureSize,
      })
      return res.data
    },
    onSuccess: () => {
      refetchJobs()
    }
  })

  // Image-to-3D mutation
  const imageMutation = useMutation({
    mutationFn: async () => {
      if (!imageFile) throw new Error('No image selected')

      const formData = new FormData()
      formData.append('file', imageFile)
      formData.append('seed', seed.toString())
      formData.append('sparse_steps', sparseSteps.toString())
      formData.append('sparse_cfg', sparseCfg.toString())
      formData.append('slat_steps', slatSteps.toString())
      formData.append('slat_cfg', slatCfg.toString())
      formData.append('simplify', simplify.toString())
      formData.append('texture_size', textureSize.toString())

      const res = await axios.post(`${API_BASE}/api/generate/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      return res.data
    },
    onSuccess: () => {
      refetchJobs()
    }
  })

  // Delete job
  const deleteMutation = useMutation({
    mutationFn: async (jobId: string) => {
      await axios.delete(`${API_BASE}/api/jobs/${jobId}`)
    },
    onSuccess: () => {
      refetchJobs()
    }
  })

  const handleGenerate = () => {
    if (mode === 'text') {
      textMutation.mutate()
    } else {
      imageMutation.mutate()
    }
  }

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target?.result as string)
      reader.readAsDataURL(file)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'processing':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>TRELLIS GUI</h1>
        <p>Intuitive 3D Generation Interface</p>
      </header>

      <div className="container">
        {/* Input Panel */}
        <div className="panel input-panel">
          <div className="mode-selector">
            <button
              className={`mode-btn ${mode === 'text' ? 'active' : ''}`}
              onClick={() => setMode('text')}
            >
              <Type className="w-5 h-5" />
              Text-to-3D
            </button>
            <button
              className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
              onClick={() => setMode('image')}
            >
              <ImageIcon className="w-5 h-5" />
              Image-to-3D
            </button>
          </div>

          {mode === 'text' ? (
            <div className="input-section">
              <label>Text Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe your 3D object..."
                rows={4}
              />
            </div>
          ) : (
            <div className="input-section">
              <label>Upload Image</label>
              <div className="image-upload">
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                ) : (
                  <div className="upload-placeholder">
                    <ImageIcon className="w-12 h-12 text-gray-500" />
                    <p>Click to upload image</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="file-input"
                />
              </div>
            </div>
          )}

          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Settings className="w-4 h-4" />
            Advanced Settings
          </button>

          {showAdvanced && (
            <div className="advanced-settings">
              <div className="setting">
                <label>Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value))}
                />
              </div>

              <div className="setting-group">
                <h4>Sparse Structure</h4>
                <div className="setting">
                  <label>Steps: {sparseSteps}</label>
                  <input
                    type="range"
                    min="4"
                    max="50"
                    value={sparseSteps}
                    onChange={(e) => setSparseSteps(parseInt(e.target.value))}
                  />
                </div>
                <div className="setting">
                  <label>CFG Strength: {sparseCfg.toFixed(1)}</label>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="0.5"
                    value={sparseCfg}
                    onChange={(e) => setSparseCfg(parseFloat(e.target.value))}
                  />
                </div>
              </div>

              <div className="setting-group">
                <h4>SLat (Appearance)</h4>
                <div className="setting">
                  <label>Steps: {slatSteps}</label>
                  <input
                    type="range"
                    min="4"
                    max="50"
                    value={slatSteps}
                    onChange={(e) => setSlatSteps(parseInt(e.target.value))}
                  />
                </div>
                <div className="setting">
                  <label>CFG Strength: {slatCfg.toFixed(1)}</label>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="0.5"
                    value={slatCfg}
                    onChange={(e) => setSlatCfg(parseFloat(e.target.value))}
                  />
                </div>
              </div>

              <div className="setting">
                <label>Mesh Simplification: {(simplify * 100).toFixed(0)}%</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={simplify}
                  onChange={(e) => setSimplify(parseFloat(e.target.value))}
                />
              </div>

              <div className="setting">
                <label>Texture Size</label>
                <select
                  value={textureSize}
                  onChange={(e) => setTextureSize(parseInt(e.target.value))}
                >
                  <option value={512}>512×512</option>
                  <option value={1024}>1024×1024</option>
                  <option value={2048}>2048×2048</option>
                </select>
              </div>
            </div>
          )}

          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={textMutation.isPending || imageMutation.isPending}
          >
            {(textMutation.isPending || imageMutation.isPending) ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Starting Generation...
              </>
            ) : (
              <>
                Generate 3D Model
              </>
            )}
          </button>
        </div>

        {/* Jobs Panel */}
        <div className="panel jobs-panel">
          <h2>Generation Queue</h2>

          {!jobsData || jobsData.length === 0 ? (
            <div className="empty-state">
              <p>No generations yet. Create your first 3D model!</p>
            </div>
          ) : (
            <div className="jobs-list">
              {jobsData.map((job) => (
                <div key={job.job_id} className="job-card">
                  <div className="job-header">
                    <div className="job-info">
                      {getStatusIcon(job.status)}
                      <div>
                        <h3>{job.type === 'text_to_3d' ? 'Text-to-3D' : 'Image-to-3D'}</h3>
                        <p className="job-message">{job.message}</p>
                      </div>
                    </div>
                    <button
                      className="delete-btn"
                      onClick={() => deleteMutation.mutate(job.job_id)}
                      title="Delete job"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {job.status === 'processing' && (
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${job.progress * 100}%` }}
                      />
                    </div>
                  )}

                  {job.type === 'text_to_3d' && job.params?.prompt && (
                    <p className="job-prompt">"{job.params.prompt}"</p>
                  )}

                  {job.status === 'completed' && job.outputs && (
                    <div className="job-outputs">
                      {job.outputs.preview && (
                        <img
                          src={`${API_BASE}/api/download/${job.job_id}/preview`}
                          alt="Preview"
                          className="output-preview"
                        />
                      )}
                      <div className="download-buttons">
                        {job.outputs.glb && (
                          <a
                            href={`${API_BASE}/api/download/${job.job_id}/glb`}
                            download
                            className="download-btn"
                          >
                            <Download className="w-4 h-4" />
                            GLB
                          </a>
                        )}
                        {job.outputs.ply && (
                          <a
                            href={`${API_BASE}/api/download/${job.job_id}/ply`}
                            download
                            className="download-btn"
                          >
                            <Download className="w-4 h-4" />
                            PLY
                          </a>
                        )}
                      </div>
                    </div>
                  )}

                  {job.status === 'failed' && job.error && (
                    <div className="error-message">
                      Error: {job.error}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
