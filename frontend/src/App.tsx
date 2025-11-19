/**
 * Main application component - refactored and modular.
 */

import { Header } from '@/components/layout/Header'
import { ControlPanel } from '@/components/layout/ControlPanel'
import { ImageDisplay } from '@/components/generation/ImageDisplay'
import { useImageGeneration } from '@/hooks/useImageGeneration'

function App() {
  const generation = useImageGeneration()

  return (
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      <Header />
      
      <main className="flex-1 flex overflow-hidden">
        {/* LEFT SIDE - CONTROLS */}
        <div className="w-1/2 border-r border-gray-300 bg-white overflow-y-auto">
          <div className="p-6">
            <h2 className="text-2xl font-bold mb-4 text-gray-800">🎨 Generation Controls</h2>
            <ControlPanel
              selectedModel={generation.selectedModel}
              setSelectedModel={generation.setSelectedModel}
              prompt={generation.prompt}
              setPrompt={generation.setPrompt}
              negativePrompt={generation.negativePrompt}
              setNegativePrompt={generation.setNegativePrompt}
              promptImage={generation.promptImage}
              setPromptImage={generation.setPromptImage}
              useImageAsPrompt={generation.useImageAsPrompt}
              setUseImageAsPrompt={generation.setUseImageAsPrompt}
              initImage={generation.initImage}
              setInitImage={generation.setInitImage}
              strength={generation.strength}
              setStrength={generation.setStrength}
              numSteps={generation.numSteps}
              setNumSteps={generation.setNumSteps}
              guidanceScale={generation.guidanceScale}
              setGuidanceScale={generation.setGuidanceScale}
              showAdvanced={generation.showAdvanced}
              setShowAdvanced={generation.setShowAdvanced}
              noiseMaskType={generation.noiseMaskType}
              setNoiseMaskType={generation.setNoiseMaskType}
              noiseStrength={generation.noiseStrength}
              setNoiseStrength={generation.setNoiseStrength}
              injectAtStep={generation.injectAtStep}
              setInjectAtStep={generation.setInjectAtStep}
              startStep={generation.startStep}
              setStartStep={generation.setStartStep}
              useCurrentImage={generation.useCurrentImage}
              setUseCurrentImage={generation.setUseCurrentImage}
              currentImage={generation.currentImage}
              isGenerating={generation.isGenerating}
              onGenerate={generation.handleGenerate}
              onResetNoiseSettings={generation.handleResetNoiseSettings}
            />
          </div>
        </div>

        {/* RIGHT SIDE - IMAGE DISPLAY */}
        <div className="w-1/2 bg-gray-100 p-6 flex flex-col overflow-hidden">
          <h2 className="text-2xl font-bold mb-4 text-gray-800 flex-shrink-0">🖼️ Generated Image</h2>
          <div className="flex-1 overflow-hidden">
            <ImageDisplay
              currentImage={generation.currentImage}
              currentModel={generation.currentModel}
              isGenerating={generation.isGenerating}
              progress={generation.progress}
              currentStep={generation.currentStep}
              totalSteps={generation.totalSteps}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
