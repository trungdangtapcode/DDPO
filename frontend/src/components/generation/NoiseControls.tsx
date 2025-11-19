/**
 * Component for advanced noise control settings.
 */

import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'
import { NOISE_MASK_OPTIONS } from '@/constants/models'

interface NoiseControlsProps {
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
  numSteps: number
  isGenerating: boolean
  onResetNoiseSettings: () => void
}

export function NoiseControls({
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
  numSteps,
  isGenerating,
  onResetNoiseSettings,
}: NoiseControlsProps) {
  return (
    <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
      <h3 className="font-semibold text-sm">Noise Injection Controls</h3>

      <div className="space-y-2">
        <Label>Noise Mask Type</Label>
        <select
          value={noiseMaskType}
          onChange={(e) => setNoiseMaskType(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-md"
          disabled={isGenerating}
        >
          {NOISE_MASK_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {noiseMaskType !== 'none' && (
        <>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Noise Strength</Label>
              <span className="text-sm text-gray-500">
                {noiseStrength.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[noiseStrength]}
              onValueChange={(v: number[]) => setNoiseStrength(v[0])}
              min={0.1}
              max={2.0}
              step={0.1}
              disabled={isGenerating}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Inject at Step</Label>
              <span className="text-sm text-gray-500">{injectAtStep}</span>
            </div>
            <Slider
              value={[injectAtStep]}
              onValueChange={(v: number[]) => setInjectAtStep(v[0])}
              min={0}
              max={numSteps - 1}
              step={1}
              disabled={isGenerating}
            />
          </div>
        </>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Start Step</Label>
          <span className="text-sm text-gray-500">{startStep}</span>
        </div>
        <Slider
          value={[startStep]}
          onValueChange={(v: number[]) => setStartStep(v[0])}
          min={0}
          max={numSteps - 1}
          step={1}
          disabled={isGenerating}
        />
      </div>

      {startStep > 0 && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="use-current"
            checked={useCurrentImage}
            onCheckedChange={setUseCurrentImage}
            disabled={!currentImage || isGenerating}
          />
          <Label htmlFor="use-current" className="text-sm">
            Use current image as start image
          </Label>
        </div>
      )}

      {(startStep > 0 || noiseMaskType !== 'none') && (
        <button
          onClick={onResetNoiseSettings}
          className="text-sm text-blue-600 hover:underline"
          disabled={isGenerating}
        >
          Reset Noise Settings
        </button>
      )}
    </div>
  )
}
