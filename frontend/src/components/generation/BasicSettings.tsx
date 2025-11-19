/**
 * Component for basic generation settings.
 */

import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'

interface BasicSettingsProps {
  numSteps: number
  setNumSteps: (steps: number) => void
  guidanceScale: number
  setGuidanceScale: (scale: number) => void
  isGenerating: boolean
}

export function BasicSettings({
  numSteps,
  setNumSteps,
  guidanceScale,
  setGuidanceScale,
  isGenerating,
}: BasicSettingsProps) {
  return (
    <>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Steps</Label>
          <span className="text-sm text-gray-500">{numSteps}</span>
        </div>
        <Slider
          value={[numSteps]}
          onValueChange={(v: number[]) => setNumSteps(v[0])}
          min={1}
          max={100}
          step={1}
          disabled={isGenerating}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Guidance Scale</Label>
          <span className="text-sm text-gray-500">{guidanceScale.toFixed(1)}</span>
        </div>
        <Slider
          value={[guidanceScale]}
          onValueChange={(v: number[]) => setGuidanceScale(v[0])}
          min={1}
          max={20}
          step={0.5}
          disabled={isGenerating}
        />
      </div>
    </>
  )
}
