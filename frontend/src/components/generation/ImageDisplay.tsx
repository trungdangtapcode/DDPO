/**
 * Component for displaying generated image and progress.
 */

import { Progress } from '@/components/ui/progress'

interface ImageDisplayProps {
  currentImage: string | null
  currentModel: string | null
  isGenerating: boolean
  progress: number
  currentStep: number
  totalSteps: number
}

export function ImageDisplay({
  currentImage,
  currentModel,
  isGenerating,
  progress,
  currentStep,
  totalSteps,
}: ImageDisplayProps) {
  return (
    <div className="h-full flex flex-col">
      {isGenerating && (
        <div className="mb-4 flex-shrink-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">
              Generating... Step {currentStep} of {totalSteps}
            </span>
            <span className="text-sm font-semibold text-blue-600">
              {Math.round(progress)}%
            </span>
          </div>
          <Progress value={progress} />
        </div>
      )}

      <div className="flex-1 bg-white rounded-lg border-2 border-gray-300 overflow-hidden flex items-center justify-center min-h-0">
        {currentImage ? (
          <img
            src={currentImage}
            alt="Generated"
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <div className="text-center text-gray-400">
            <p className="text-lg">No image generated yet</p>
            <p className="text-sm mt-2">
              Enter a prompt and click Generate to start
            </p>
          </div>
        )}
      </div>

      {currentModel && (
        <p className="mt-4 text-sm text-gray-600 flex-shrink-0">
          Model: <span className="font-semibold">{currentModel}</span>
        </p>
      )}
    </div>
  )
}
