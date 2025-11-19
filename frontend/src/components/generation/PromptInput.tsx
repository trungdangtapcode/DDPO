/**
 * Component for text prompt input.
 */

import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'

interface PromptInputProps {
  prompt: string
  setPrompt: (prompt: string) => void
  negativePrompt: string
  setNegativePrompt: (prompt: string) => void
  promptImage: string | null
  isGenerating: boolean
}

export function PromptInput({
  prompt,
  setPrompt,
  negativePrompt,
  setNegativePrompt,
  promptImage,
  isGenerating,
}: PromptInputProps) {
  return (
    <>
      <div className="space-y-2">
        <Label htmlFor="prompt">
          {promptImage ? 'Prompt (Optional)' : 'Prompt'}
        </Label>
        <Textarea
          id="prompt"
          placeholder={
            promptImage
              ? 'Optional text prompt (image will be used for CLIP embeddings)'
              : 'Enter your prompt...'
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="min-h-[100px]"
          disabled={isGenerating}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="negative-prompt">Negative Prompt (Optional)</Label>
        <Textarea
          id="negative-prompt"
          placeholder="What you don't want to see..."
          value={negativePrompt}
          onChange={(e) => setNegativePrompt(e.target.value)}
          className="min-h-[80px]"
          disabled={isGenerating}
        />
      </div>
    </>
  )
}
