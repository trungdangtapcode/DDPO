/**
 * Custom hook for image generation logic.
 */

import { useState, useRef } from 'react'
import { StreamData } from '@/types/generation'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001'

export function useImageGeneration() {
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
  const [promptImage, setPromptImage] = useState<string | null>(null)
  const [useImageAsPrompt, setUseImageAsPrompt] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(20)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const eventSourceRef = useRef<EventSource | null>(null)

  const buildRequestBody = () => {
    const requestBody: any = {
      prompt: promptImage ? "" : prompt,
      steps: numSteps,
      model: selectedModel,
      guidance_scale: guidanceScale,
    }

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

    if (promptImage) {
      const base64Image = promptImage.split(',')[1] || promptImage
      requestBody.prompt_image = base64Image
    }

    return requestBody
  }

  const handleGenerate = async () => {
    if ((!prompt.trim() && !promptImage) || isGenerating) return

    setIsGenerating(true)
    setCurrentImage(null)
    setProgress(0)
    setCurrentStep(0)
    setCurrentModel(null)

    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    try {
      const requestBody = buildRequestBody()

      const response = await fetch(`${API_URL}/api/generate`, {
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

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk

        const lines = buffer.split('\n')
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
                console.error('Error parsing stream data:', error)
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

  const handleResetNoiseSettings = () => {
    setStartStep(0)
    setUseCurrentImage(false)
  }

  return {
    // State
    prompt,
    negativePrompt,
    selectedModel,
    numSteps,
    guidanceScale,
    startStep,
    useCurrentImage,
    noiseMaskType,
    noiseStrength,
    injectAtStep,
    initImage,
    strength,
    promptImage,
    useImageAsPrompt,
    isGenerating,
    currentImage,
    currentModel,
    progress,
    currentStep,
    totalSteps,
    showAdvanced,
    
    // Actions
    setPrompt,
    setNegativePrompt,
    setSelectedModel,
    setNumSteps,
    setGuidanceScale,
    setStartStep,
    setUseCurrentImage,
    setNoiseMaskType,
    setNoiseStrength,
    setInjectAtStep,
    setInitImage,
    setStrength,
    setPromptImage,
    setUseImageAsPrompt,
    setShowAdvanced,
    handleGenerate,
    handleResetNoiseSettings,
  }
}
