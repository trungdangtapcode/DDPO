import { useState, useRef, useEffect } from 'react'
import { Sparkles, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select } from '@/components/ui/select'

interface StreamData {
  step: number
  total_steps: number
  image: string
  progress: number
  done: boolean
  model?: string
}

const AVAILABLE_MODELS = [
  { value: 'aesthetic', label: 'Aesthetic Quality' },
  { value: 'alignment', label: 'Text Alignment' },
  { value: 'compressibility', label: 'Compressibility' },
  { value: 'incompressibility', label: 'Incompressibility' },
]

function App() {
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [selectedModel, setSelectedModel] = useState('compressibility')
  const [numSteps, setNumSteps] = useState(20)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [startStep, setStartStep] = useState(0)
  const [useCurrentImage, setUseCurrentImage] = useState(false)
  const [noiseMaskType, setNoiseMaskType] = useState('none')
  const [noiseStrength, setNoiseStrength] = useState(1.0)
  const [injectAtStep, setInjectAtStep] = useState(-1)
  const [initImage, setInitImage] = useState<string | null>(null)
  const [strength, setStrength] = useState(0.75)
  const [promptImage, setPromptImage] = useState<string | null>(null) // Image to use as prompt
  const [, setUseImageAsPrompt] = useState(false) // Toggle for image-as-prompt mode
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(20)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)

  const handleGenerate = async () => {
    if ((!prompt.trim() && !promptImage) || isGenerating) return

    setIsGenerating(true)
    setCurrentImage(null)
    setProgress(0)
    setCurrentStep(0)
    setCurrentModel(null)

    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    try {
      // Build request body
      const requestBody: any = {
        prompt: promptImage ? "" : prompt,  // Empty prompt if using image as prompt
        steps: numSteps,
        model: selectedModel,
        guidance_scale: guidanceScale,
      }

      // Add optional parameters
      if (negativePrompt.trim()) {
        requestBody.negative_prompt = negativePrompt
      }

      if (startStep > 0) {
        requestBody.start_step = startStep
        if (useCurrentImage && currentImage) {
          const base64Image = currentImage.split(',')[1]
          requestBody.start_image = base64Image
        }
      }

      if (noiseMaskType !== 'none' && injectAtStep >= 0 && injectAtStep < numSteps) {
        requestBody.noise_mask_type = noiseMaskType
        requestBody.noise_strength = noiseStrength
        requestBody.inject_at_step = injectAtStep
      }

      if (initImage) {
        const base64Image = initImage.split(',')[1] || initImage
        requestBody.init_image = base64Image
        requestBody.strength = strength
      }

      // Add image-as-prompt if enabled
      if (promptImage) {
        const base64Image = promptImage.split(',')[1] || promptImage
        requestBody.prompt_image = base64Image
      }

      // Get API URL from environment variable
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3001'

      // Use fetch with POST for streaming
      const response = await fetch(`${apiUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No reader available')
      }

      // Buffer for incomplete chunks
      let buffer = ''

      // Read the stream
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk

        // Process complete lines
        const lines = buffer.split('\n')
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            
            if (data === '[DONE]') {
              setIsGenerating(false)
              break
            }

            if (data) {
              try {
                const parsedData: StreamData = JSON.parse(data)
                
                setProgress(parsedData.progress)
                setCurrentStep(parsedData.step)
                setTotalSteps(parsedData.total_steps)

                if (parsedData.model) {
                  setCurrentModel(parsedData.model)
                }

                const imageUrl = `data:image/jpeg;base64,${parsedData.image}`
                setCurrentImage(imageUrl)

                if (parsedData.done) {
                  setIsGenerating(false)
                }
              } catch (error) {
                console.error('Error parsing stream data:', error, 'Data:', data.substring(0, 100))
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Fetch error:', error)
      setIsGenerating(false)
    }
  }

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isGenerating) {
      handleGenerate()
    }
  }

  const handleResetNoiseSettings = () => {
    setStartStep(0)
    setUseCurrentImage(false)
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image file is too large (max 10MB)')
      return
    }

    // Read file as base64
    const reader = new FileReader()
    reader.onload = (event) => {
      const result = event.target?.result as string
      setInitImage(result)
    }
    reader.readAsDataURL(file)
  }

  const handleClearInitImage = () => {
    setInitImage(null)
  }

  const handlePromptImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image file is too large (max 10MB)')
      return
    }

    // Read file as base64
    const reader = new FileReader()
    reader.onload = (event) => {
      const result = event.target?.result as string
      setPromptImage(result)
      setUseImageAsPrompt(true)
    }
    reader.readAsDataURL(file)
  }

  const handleClearPromptImage = () => {
    setPromptImage(null)
    setUseImageAsPrompt(false)
  }

  return (
    <div className="h-screen flex overflow-hidden bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Left Panel - Controls */}
      <div className="w-1/2 flex flex-col border-r border-slate-200 dark:border-slate-700">
        {/* Header */}
        <div className="p-6 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600">
              Diffusion Studio
            </h1>
          </div>
          <p className="text-sm text-muted-foreground">
            Real-time streaming diffusion with DDPO models
          </p>
        </div>

        {/* Controls - Scrollable */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Model Selection</CardTitle>
              <CardDescription>
                Choose from 4 DDPO-optimized models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <label htmlFor="model-select" className="text-sm font-medium">
                  Model Type
                </label>
                <Select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isGenerating}
                >
                  {AVAILABLE_MODELS.map((model) => (
                    <option key={model.value} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </Select>
                <p className="text-xs text-muted-foreground mt-2">
                  {selectedModel === 'aesthetic' && '📸 Optimized for visual appeal and beauty'}
                  {selectedModel === 'alignment' && '🎯 Optimized for accurate prompt matching'}
                  {selectedModel === 'compressibility' && '💾 Optimized for smaller file sizes'}
                  {selectedModel === 'incompressibility' && '✨ Optimized for maximum detail'}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Image-as-Prompt Mode */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>🎨 Image-as-Prompt (CLIP)</span>
                <span className="text-xs font-normal text-muted-foreground">(Experimental)</span>
              </CardTitle>
              <CardDescription>
                Use an image to guide generation instead of text prompt
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {!promptImage ? (
                  <div>
                    <label
                      htmlFor="prompt-image-upload"
                      className="flex flex-col items-center justify-center w-full h-28 border-2 border-dashed border-purple-300 dark:border-purple-600 rounded-lg cursor-pointer bg-purple-50 dark:bg-purple-950 hover:bg-purple-100 dark:hover:bg-purple-900 transition"
                    >
                      <div className="flex flex-col items-center justify-center pt-4 pb-5">
                        <svg className="w-7 h-7 mb-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <p className="mb-1 text-sm text-purple-600 dark:text-purple-400">
                          <span className="font-semibold">Upload image as prompt</span>
                        </p>
                        <p className="text-xs text-purple-500 dark:text-purple-500">Uses CLIP image encoder</p>
                      </div>
                      <input
                        id="prompt-image-upload"
                        type="file"
                        accept="image/*"
                        onChange={handlePromptImageUpload}
                        disabled={isGenerating}
                        className="hidden"
                      />
                    </label>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Image Preview */}
                    <div className="relative">
                      <img
                        src={promptImage}
                        alt="Prompt image"
                        className="w-full h-48 object-contain rounded-lg border-2 border-purple-300 dark:border-purple-600 bg-purple-50 dark:bg-purple-950"
                      />
                      <button
                        onClick={handleClearPromptImage}
                        disabled={isGenerating}
                        className="absolute top-2 right-2 bg-purple-600 hover:bg-purple-700 text-white rounded-full p-2 shadow-lg"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    {/* Info Box */}
                    <div className="bg-purple-50 dark:bg-purple-950 border border-purple-200 dark:border-purple-800 p-3 rounded text-xs space-y-1">
                      <p className="font-medium text-purple-900 dark:text-purple-200">
                        🔬 Using CLIP image embeddings as prompt
                      </p>
                      <p className="text-purple-700 dark:text-purple-300">
                        The AI will generate images semantically similar to this reference image
                      </p>
                      <p className="text-purple-600 dark:text-purple-400 mt-1">
                        Text prompt will be ignored when image-as-prompt is active
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Prompt Input */}
          <Card>
            <CardHeader>
              <CardTitle>Prompt</CardTitle>
              <CardDescription>
                Describe the image you want to generate {promptImage && "(disabled when using image-as-prompt)"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Positive Prompt</label>
                  <Input
                    placeholder="A beautiful sunset over mountains..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={isGenerating || !!promptImage}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium mb-1.5 block">
                    Negative Prompt <span className="text-xs text-muted-foreground font-normal">(Optional)</span>
                  </label>
                  <Input
                    placeholder="blur, low quality, distorted..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    disabled={isGenerating}
                    className="w-full"
                  />
                </div>

                <Button 
                  onClick={handleGenerate} 
                  disabled={isGenerating || (!prompt.trim() && !promptImage)}
                  className="w-full"
                  size="lg"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-5 w-5" />
                      {promptImage ? 'Generate from Image' : 'Generate Image'}
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Image-to-Image Mode */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>🖼️ Image-to-Image</span>
                <span className="text-xs font-normal text-muted-foreground">(Optional)</span>
              </CardTitle>
              <CardDescription>
                Upload an image to transform it instead of starting from noise
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {!initImage ? (
                  <div>
                    <label
                      htmlFor="file-upload"
                      className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg cursor-pointer bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 transition"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <svg className="w-8 h-8 mb-2 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <p className="mb-1 text-sm text-slate-500 dark:text-slate-400">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-slate-400 dark:text-slate-500">PNG, JPG, JPEG (MAX. 10MB)</p>
                      </div>
                      <input
                        id="file-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        disabled={isGenerating}
                        className="hidden"
                      />
                    </label>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Image Preview */}
                    <div className="relative">
                      <img
                        src={initImage}
                        alt="Init image"
                        className="w-full h-64 object-contain rounded-lg border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900"
                      />
                      <button
                        onClick={handleClearInitImage}
                        disabled={isGenerating}
                        className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-2 shadow-lg"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    {/* Strength Slider */}
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-sm font-medium">Transformation Strength</label>
                        <span className="text-sm text-muted-foreground">{strength.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={strength}
                        onChange={(e) => setStrength(Number(e.target.value))}
                        disabled={isGenerating}
                        className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer dark:bg-blue-900"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>0.0 (subtle)</span>
                        <span>0.5 (moderate)</span>
                        <span>1.0 (major)</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Lower values preserve more of the original image structure
                      </p>
                    </div>

                    {/* Info Box */}
                    <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 p-3 rounded text-xs">
                      <p className="font-medium text-blue-900 dark:text-blue-200">
                        💡 Your prompt will guide the transformation of this image
                      </p>
                      <p className="text-blue-700 dark:text-blue-300 mt-1">
                        Strength: {strength < 0.4 ? 'Very subtle changes' : strength < 0.7 ? 'Moderate transformation' : 'Major transformation'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Advanced Settings */}
          <Card>
            <CardHeader className="cursor-pointer" onClick={() => setShowAdvanced(!showAdvanced)}>
              <CardTitle className="flex items-center justify-between text-base">
                <span>Advanced Settings</span>
                <span className="text-xs text-muted-foreground font-normal">
                  {showAdvanced ? '▼' : '▶'} Click to {showAdvanced ? 'hide' : 'show'}
                </span>
              </CardTitle>
            </CardHeader>
            {showAdvanced && (
              <CardContent className="space-y-4">
                {/* Inference Steps */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium">Inference Steps</label>
                    <span className="text-sm text-muted-foreground">{numSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    step="5"
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                    disabled={isGenerating}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer dark:bg-slate-700"
                  />
                  <p className="text-xs text-muted-foreground">
                    More steps = better quality but slower generation (10-50)
                  </p>
                </div>

                {/* Guidance Scale */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium">Guidance Scale</label>
                    <span className="text-sm text-muted-foreground">{guidanceScale.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    disabled={isGenerating}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer dark:bg-slate-700"
                  />
                  <p className="text-xs text-muted-foreground">
                    Higher = follows prompt more closely (1-20, recommended: 7-10)
                  </p>
                </div>

                {/* Noise Interaction */}
                <div className="space-y-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                  <h4 className="text-sm font-medium text-purple-600 dark:text-purple-400">🎨 Noise Interaction</h4>
                  
                  {/* Start Step */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">Start from Step</label>
                      <span className="text-sm text-muted-foreground">{startStep}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={Math.max(numSteps - 1, 0)}
                      step="1"
                      value={startStep}
                      onChange={(e) => setStartStep(Number(e.target.value))}
                      disabled={isGenerating}
                      className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer dark:bg-purple-900"
                    />
                    <p className="text-xs text-muted-foreground">
                      Skip initial steps (0 = start from noise, {numSteps-1} = almost final)
                    </p>
                  </div>

                  {/* Use Current Image Checkbox */}
                  {currentImage && startStep > 0 && (
                    <div className="flex items-center space-x-2 bg-purple-50 dark:bg-purple-950 p-3 rounded-lg">
                      <input
                        type="checkbox"
                        id="use-current-image"
                        checked={useCurrentImage}
                        onChange={(e) => setUseCurrentImage(e.target.checked)}
                        disabled={isGenerating}
                        className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                      />
                      <label htmlFor="use-current-image" className="text-sm font-medium cursor-pointer">
                        Start from current image (add noise at step {startStep})
                      </label>
                    </div>
                  )}

                  {startStep > 0 && (
                    <div className="space-y-2">
                      <div className="bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 p-2 rounded text-xs">
                        <p className="font-medium text-amber-900 dark:text-amber-200">
                          💡 Tip: {useCurrentImage && currentImage 
                            ? `Will add noise to current image and denoise from step ${startStep}`
                            : `Will skip first ${startStep} step${startStep > 1 ? 's' : ''} of denoising`}
                        </p>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={handleResetNoiseSettings}
                        className="w-full text-xs"
                      >
                        Reset to Default (Start from Noise)
                      </Button>
                    </div>
                  )}
                </div>

                {/* Latent Noise Disruption */}
                <div className="space-y-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                  <h4 className="text-sm font-medium text-indigo-600 dark:text-indigo-400">🔬 Latent Noise Disruption</h4>
                  <p className="text-xs text-muted-foreground">
                    Inject noise to specific regions at any timestep to study diffusion behavior
                  </p>

                  {/* Noise Mask Type */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Noise Pattern</label>
                    <select
                      value={noiseMaskType}
                      onChange={(e) => setNoiseMaskType(e.target.value)}
                      disabled={isGenerating}
                      className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-sm"
                    >
                      <option value="none">None (disabled)</option>
                      <option value="center_circle">Center Circle</option>
                      <option value="center_square">Center Square</option>
                      <option value="edges">Edges/Border</option>
                      <option value="corners">Four Corners</option>
                      <option value="left_half">Left Half</option>
                      <option value="right_half">Right Half</option>
                      <option value="top_half">Top Half</option>
                      <option value="bottom_half">Bottom Half</option>
                      <option value="checkerboard">Checkerboard</option>
                    </select>
                  </div>

                  {noiseMaskType !== 'none' && (
                    <>
                      {/* Inject At Step */}
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <label className="text-sm font-medium">Inject at Step</label>
                          <span className="text-sm text-muted-foreground">{injectAtStep >= 0 ? injectAtStep : 'Not set'}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max={Math.max(numSteps - 1, 0)}
                          step="1"
                          value={injectAtStep >= 0 ? injectAtStep : 0}
                          onChange={(e) => setInjectAtStep(Number(e.target.value))}
                          disabled={isGenerating}
                          className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer dark:bg-indigo-900"
                        />
                        <p className="text-xs text-muted-foreground">
                          Step when noise will be injected (0 = early, {numSteps-1} = late)
                        </p>
                      </div>

                      {/* Noise Strength */}
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <label className="text-sm font-medium">Noise Strength</label>
                          <span className="text-sm text-muted-foreground">{noiseStrength.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={noiseStrength}
                          onChange={(e) => setNoiseStrength(Number(e.target.value))}
                          disabled={isGenerating}
                          className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer dark:bg-indigo-900"
                        />
                        <p className="text-xs text-muted-foreground">
                          Intensity of injected noise (0 = no effect, 1 = normal, 2 = strong)
                        </p>
                      </div>

                      {/* Info Box */}
                      <div className="bg-indigo-50 dark:bg-indigo-950 border border-indigo-200 dark:border-indigo-800 p-3 rounded text-xs space-y-1">
                        <p className="font-medium text-indigo-900 dark:text-indigo-200">
                          🔬 Experiment: Does model highlight the disrupted area?
                        </p>
                        <p className="text-indigo-700 dark:text-indigo-300">
                          Pattern: <strong>{noiseMaskType.replace(/_/g, ' ')}</strong> at step <strong>{injectAtStep}</strong>
                        </p>
                      </div>

                      {/* Reset Button */}
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => {
                          setNoiseMaskType('none')
                          setInjectAtStep(-1)
                          setNoiseStrength(1.0)
                        }}
                        className="w-full text-xs"
                      >
                        Disable Noise Disruption
                      </Button>
                    </>
                  )}
                </div>
              </CardContent>
            )}
          </Card>

          {/* Progress */}
          {(isGenerating || currentImage) && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Progress</span>
                  <span className="text-sm font-normal text-muted-foreground">
                    Step {currentStep} / {totalSteps}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Progress value={progress} className="h-2" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{progress.toFixed(1)}% Complete</span>
                  {currentModel && (
                    <span>Using: {AVAILABLE_MODELS.find(m => m.value === currentModel.replace('stable-diffusion-', ''))?.label}</span>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Info */}
          <Card className="border-dashed">
            <CardContent className="pt-6">
              <div className="text-xs text-muted-foreground space-y-2">
                <p className="flex items-start gap-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary mt-1"></span>
                  <span>All 4 DDPO models pre-loaded in VRAM for instant switching</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary mt-1"></span>
                  <span>Progressive rendering shows each denoising step in real-time</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-500 mt-1"></span>
                  <span>Models: Aesthetic, Alignment, Compressibility, Incompressibility</span>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Right Panel - Image Display */}
      <div className="w-1/2 flex flex-col bg-slate-100 dark:bg-slate-900">
        {currentImage ? (
          <>
            {/* Image Header */}
            <div className="p-6 border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
              <h2 className="text-lg font-semibold mb-1">Generated Image</h2>
              <p className="text-sm text-muted-foreground line-clamp-2">{prompt}</p>
            </div>

            {/* Image Container */}
            <div className="flex-1 flex items-center justify-center p-6">
              <div className="relative w-full h-full flex items-center justify-center">
                <img
                  src={currentImage}
                  alt="Generated"
                  className={`max-w-full max-h-full object-contain rounded-lg shadow-2xl transition-opacity duration-300 ${
                    isGenerating ? 'opacity-70' : 'opacity-100'
                  }`}
                />
                {isGenerating && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-white/90 dark:bg-slate-900/90 px-6 py-3 rounded-full shadow-lg backdrop-blur-sm">
                      <p className="text-sm font-medium flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Processing...
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          // Placeholder
          <div className="flex-1 flex items-center justify-center p-12">
            <div className="text-center space-y-4">
              <div className="w-24 h-24 mx-auto rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center">
                <Sparkles className="w-12 h-12 text-slate-400" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-semibold text-slate-600 dark:text-slate-400">
                  No Image Yet
                </h3>
                <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                  Select a model, enter a prompt, and click generate to see your creation appear here
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
