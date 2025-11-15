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
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(20)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)

  const handleGenerate = () => {
    if (!prompt.trim() || isGenerating) return

    setIsGenerating(true)
    setCurrentImage(null)
    setProgress(0)
    setCurrentStep(0)
    setCurrentModel(null)

    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    // Create new EventSource connection with model parameter
    let url = `/api/generate?prompt=${encodeURIComponent(prompt)}&steps=${numSteps}&model=${selectedModel}&guidance_scale=${guidanceScale}`
    
    // Add negative prompt if provided
    if (negativePrompt.trim()) {
      url += `&negative_prompt=${encodeURIComponent(negativePrompt)}`
    }
    
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      if (event.data === '[DONE]') {
        eventSource.close()
        setIsGenerating(false)
        return
      }

      try {
        const data: StreamData = JSON.parse(event.data)
        
        // Update progress
        setProgress(data.progress)
        setCurrentStep(data.step)
        setTotalSteps(data.total_steps)

        // Update model if provided
        if (data.model) {
          setCurrentModel(data.model)
        }

        // Update image
        const imageUrl = `data:image/jpeg;base64,${data.image}`
        setCurrentImage(imageUrl)

        // If done, close connection
        if (data.done) {
          eventSource.close()
          setIsGenerating(false)
        }
      } catch (error) {
        console.error('Error parsing stream data:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error)
      eventSource.close()
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

          {/* Prompt Input */}
          <Card>
            <CardHeader>
              <CardTitle>Prompt</CardTitle>
              <CardDescription>
                Describe the image you want to generate
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
                    disabled={isGenerating}
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
                  disabled={isGenerating || !prompt.trim()}
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
                      Generate Image
                    </>
                  )}
                </Button>
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
