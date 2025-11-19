/**
 * Component for uploading prompt image (CLIP-based image-as-prompt).
 */

import { Upload, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { MAX_FILE_SIZE } from '@/constants/models'

interface PromptImageUploadProps {
  promptImage: string | null
  setPromptImage: (image: string | null) => void
  useImageAsPrompt: boolean
  setUseImageAsPrompt: (use: boolean) => void
  isGenerating: boolean
}

export function PromptImageUpload({
  promptImage,
  setPromptImage,
  setUseImageAsPrompt,
  isGenerating,
}: PromptImageUploadProps) {
  const handlePromptImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.size > MAX_FILE_SIZE) {
        alert(`File size must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB`)
        return
      }
      const reader = new FileReader()
      reader.onloadend = () => {
        setPromptImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>Prompt Image (CLIP)</Label>
        {promptImage && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setPromptImage(null)
              setUseImageAsPrompt(false)
            }}
            disabled={isGenerating}
          >
            <X className="h-4 w-4 mr-1" />
            Remove
          </Button>
        )}
      </div>

      {!promptImage ? (
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors">
          <input
            type="file"
            accept="image/*"
            onChange={handlePromptImageUpload}
            className="hidden"
            id="prompt-image-upload"
            disabled={isGenerating}
          />
          <label
            htmlFor="prompt-image-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            <Upload className="h-8 w-8 text-gray-400 mb-2" />
            <span className="text-sm text-gray-600">
              Upload image for CLIP-based prompt
            </span>
          </label>
        </div>
      ) : (
        <div className="relative">
          <img
            src={promptImage}
            alt="Prompt Image"
            className="w-full max-h-64 object-contain rounded-lg border border-gray-300 bg-gray-50"
          />
        </div>
      )}

      <p className="text-xs text-gray-500">
        Use image instead of text prompt (CLIP embeddings)
      </p>
    </div>
  )
}
