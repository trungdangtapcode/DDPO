import { useState, useRef, useEffect } from 'react'
import { Sparkles, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface StreamData {
  step: number
  total_steps: number
  image: string
  progress: number
  done: boolean
}

function App() {
  const [prompt, setPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(20)
  const eventSourceRef = useRef<EventSource | null>(null)

  const handleGenerate = () => {
    if (!prompt.trim() || isGenerating) return

    setIsGenerating(true)
    setCurrentImage(null)
    setProgress(0)
    setCurrentStep(0)

    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    // Create new EventSource connection
    const url = `/api/generate?prompt=${encodeURIComponent(prompt)}&steps=${totalSteps}`
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600">
              Mock Diffusion Studio
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Watch AI generate images in real-time with streaming diffusion
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Input Card */}
          <Card>
            <CardHeader>
              <CardTitle>Generate Image</CardTitle>
              <CardDescription>
                Enter a prompt and watch the diffusion process in real-time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="A beautiful sunset over mountains..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={isGenerating}
                  className="flex-1"
                />
                <Button 
                  onClick={handleGenerate} 
                  disabled={isGenerating || !prompt.trim()}
                  className="min-w-[120px]"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Generate
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Progress Card */}
          {(isGenerating || currentImage) && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Generation Progress</span>
                  <span className="text-sm font-normal text-muted-foreground">
                    Step {currentStep} / {totalSteps}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-muted-foreground mt-2 text-center">
                  {progress.toFixed(1)}% Complete
                </p>
              </CardContent>
            </Card>
          )}

          {/* Image Preview Card */}
          {currentImage && (
            <Card>
              <CardHeader>
                <CardTitle>Generated Image</CardTitle>
                <CardDescription className="line-clamp-1">
                  {prompt}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-square rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-800">
                  <img
                    src={currentImage}
                    alt="Generated"
                    className={`w-full h-full object-contain transition-opacity duration-300 ${
                      isGenerating ? 'opacity-70' : 'opacity-100'
                    }`}
                  />
                  {isGenerating && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/10 backdrop-blur-[1px]">
                      <div className="bg-white/90 dark:bg-slate-900/90 px-4 py-2 rounded-full shadow-lg">
                        <p className="text-sm font-medium">Processing...</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Info Card */}
          <Card className="border-dashed">
            <CardContent className="pt-6">
              <div className="text-sm text-muted-foreground space-y-2">
                <p className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-primary"></span>
                  This is a mock diffusion model that simulates the image generation process
                </p>
                <p className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-primary"></span>
                  Watch as the image progressively becomes clearer with each denoising step
                </p>
                <p className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-primary"></span>
                  Architecture: ViteJS (Frontend) → Node.js (Backend) → Python FastAPI (Service)
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default App
