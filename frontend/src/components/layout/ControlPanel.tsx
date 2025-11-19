/**
 * Control panel with all generation settings.
 */

import { ChevronDown, ChevronUp, Sparkles } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ModelSelector } from '@/components/generation/ModelSelector'
import { PromptInput } from '@/components/generation/PromptInput'
import { PromptImageUpload } from '@/components/generation/PromptImageUpload'
import { InitImageUpload } from '@/components/generation/InitImageUpload'
import { BasicSettings } from '@/components/generation/BasicSettings'
import { NoiseControls } from '@/components/generation/NoiseControls'
import { InfoCard } from '@/components/generation/InfoCard'

interface ControlPanelProps {
  // Model
  selectedModel: string
  setSelectedModel: (model: string) => void
  
  // Prompts
  prompt: string
  setPrompt: (prompt: string) => void
  negativePrompt: string
  setNegativePrompt: (prompt: string) => void
  
  // Images
  promptImage: string | null
  setPromptImage: (image: string | null) => void
  useImageAsPrompt: boolean
  setUseImageAsPrompt: (use: boolean) => void
  initImage: string | null
  setInitImage: (image: string | null) => void
  strength: number
  setStrength: (strength: number) => void
  
  // Basic Settings
  numSteps: number
  setNumSteps: (steps: number) => void
  guidanceScale: number
  setGuidanceScale: (scale: number) => void
  
  // Advanced Settings
  showAdvanced: boolean
  setShowAdvanced: (show: boolean) => void
  noiseMaskType: string
  setNoiseMaskType: (type: string) => void
  noiseStrength: number
  setNoiseStrength: (strength: number) => void
  injectAtStep: number
  setInjectAtStep: (step: number) => void
  startStep: number
  setStartStep: (step: number) => void
  useCurrentImage: boolean
  setUseCurrentImage: (use: boolean) => void
  currentImage: string | null
  
  // Actions
  isGenerating: boolean
  onGenerate: () => void
  onResetNoiseSettings: () => void
}

export function ControlPanel(props: ControlPanelProps) {
  const {
    selectedModel,
    setSelectedModel,
    prompt,
    setPrompt,
    negativePrompt,
    setNegativePrompt,
    promptImage,
    setPromptImage,
    useImageAsPrompt,
    setUseImageAsPrompt,
    initImage,
    setInitImage,
    strength,
    setStrength,
    numSteps,
    setNumSteps,
    guidanceScale,
    setGuidanceScale,
    showAdvanced,
    setShowAdvanced,
    noiseMaskType,
    setNoiseMaskType,
    noiseStrength,
    setNoiseStrength,
    injectAtStep,
    setInjectAtStep,
    startStep,
    setStartStep,
    useCurrentImage,
    setUseCurrentImage,
    currentImage,
    isGenerating,
    onGenerate,
    onResetNoiseSettings,
  } = props

  const canGenerate = (prompt.trim() || promptImage) && !isGenerating

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-6 space-y-6">
          <ModelSelector
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            isGenerating={isGenerating}
          />

          <PromptInput
            prompt={prompt}
            setPrompt={setPrompt}
            negativePrompt={negativePrompt}
            setNegativePrompt={setNegativePrompt}
            promptImage={promptImage}
            isGenerating={isGenerating}
          />

          <PromptImageUpload
            promptImage={promptImage}
            setPromptImage={setPromptImage}
            useImageAsPrompt={useImageAsPrompt}
            setUseImageAsPrompt={setUseImageAsPrompt}
            isGenerating={isGenerating}
          />

          <InitImageUpload
            initImage={initImage}
            setInitImage={setInitImage}
            strength={strength}
            setStrength={setStrength}
            isGenerating={isGenerating}
          />

          <BasicSettings
            numSteps={numSteps}
            setNumSteps={setNumSteps}
            guidanceScale={guidanceScale}
            setGuidanceScale={setGuidanceScale}
            isGenerating={isGenerating}
          />

          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center justify-between w-full text-sm font-semibold text-gray-700 hover:text-gray-900"
            >
              <span>Advanced Settings</span>
              {showAdvanced ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </button>
          </div>

          {showAdvanced && (
            <NoiseControls
              noiseMaskType={noiseMaskType}
              setNoiseMaskType={setNoiseMaskType}
              noiseStrength={noiseStrength}
              setNoiseStrength={setNoiseStrength}
              injectAtStep={injectAtStep}
              setInjectAtStep={setInjectAtStep}
              startStep={startStep}
              setStartStep={setStartStep}
              useCurrentImage={useCurrentImage}
              setUseCurrentImage={setUseCurrentImage}
              currentImage={currentImage}
              numSteps={numSteps}
              isGenerating={isGenerating}
              onResetNoiseSettings={onResetNoiseSettings}
            />
          )}

          <Button
            onClick={onGenerate}
            disabled={!canGenerate}
            className="w-full"
            size="lg"
          >
            <Sparkles className="h-5 w-5 mr-2" />
            {isGenerating ? 'Generating...' : 'Generate Image'}
          </Button>
        </CardContent>
      </Card>

      <InfoCard />
    </div>
  )
}
