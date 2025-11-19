/**
 * Component for selecting the diffusion model.
 */

import { Label } from '@/components/ui/label'
import { AVAILABLE_MODELS, MODEL_DESCRIPTIONS } from '@/constants/models'

interface ModelSelectorProps {
  selectedModel: string
  setSelectedModel: (model: string) => void
  isGenerating: boolean
}

export function ModelSelector({
  selectedModel,
  setSelectedModel,
  isGenerating,
}: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <Label>Model</Label>
      <select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        disabled={isGenerating}
      >
        {AVAILABLE_MODELS.map((model) => (
          <option key={model.value} value={model.value}>
            {model.label}
          </option>
        ))}
      </select>
      <p className="text-sm text-gray-600">
        {MODEL_DESCRIPTIONS[selectedModel] || 'Select a model'}
      </p>
    </div>
  )
}
