/**
 * Type definitions for image generation.
 */

export interface StreamData {
  step: number
  total_steps: number
  image: string
  progress: number
  done: boolean
  model?: string
}

export interface ModelOption {
  value: string
  label: string
}

export interface GenerationState {
  prompt: string
  negativePrompt: string
  selectedModel: string
  numSteps: number
  guidanceScale: number
  startStep: number
  useCurrentImage: boolean
  noiseMaskType: string
  noiseStrength: number
  injectAtStep: number
  initImage: string | null
  strength: number
  promptImage: string | null
  useImageAsPrompt: boolean
  isGenerating: boolean
  currentImage: string | null
  currentModel: string | null
  progress: number
  currentStep: number
  totalSteps: number
  showAdvanced: boolean
}

export interface GenerationActions {
  setPrompt: (prompt: string) => void
  setNegativePrompt: (prompt: string) => void
  setSelectedModel: (model: string) => void
  setNumSteps: (steps: number) => void
  setGuidanceScale: (scale: number) => void
  setStartStep: (step: number) => void
  setUseCurrentImage: (use: boolean) => void
  setNoiseMaskType: (type: string) => void
  setNoiseStrength: (strength: number) => void
  setInjectAtStep: (step: number) => void
  setInitImage: (image: string | null) => void
  setStrength: (strength: number) => void
  setPromptImage: (image: string | null) => void
  setUseImageAsPrompt: (use: boolean) => void
  setShowAdvanced: (show: boolean) => void
  handleGenerate: () => Promise<void>
  handleResetNoiseSettings: () => void
}
