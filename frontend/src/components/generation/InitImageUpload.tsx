/**
 * Component for uploading init image (img2img).
 */

import { Upload, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { MAX_FILE_SIZE } from '@/constants/models'

interface InitImageUploadProps {
  initImage: string | null
  setInitImage: (image: string | null) => void
  strength: number
  setStrength: (strength: number) => void
  isGenerating: boolean
}

export function InitImageUpload({
  initImage,
  setInitImage,
  strength,
  setStrength,
  isGenerating,
}: InitImageUploadProps) {
  const handleInitImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.size > MAX_FILE_SIZE) {
        alert(`File size must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB`)
        return
      }
      const reader = new FileReader()
      reader.onloadend = () => {
        setInitImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>Init Image (img2img)</Label>
        {initImage && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setInitImage(null)}
            disabled={isGenerating}
          >
            <X className="h-4 w-4 mr-1" />
            Remove
          </Button>
        )}
      </div>

      {!initImage ? (
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors">
          <input
            type="file"
            accept="image/*"
            onChange={handleInitImageUpload}
            className="hidden"
            id="init-image-upload"
            disabled={isGenerating}
          />
          <label
            htmlFor="init-image-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            <Upload className="h-8 w-8 text-gray-400 mb-2" />
            <span className="text-sm text-gray-600">
              Upload starting image
            </span>
          </label>
        </div>
      ) : (
        <div className="relative">
          <img
            src={initImage}
            alt="Init Image"
            className="w-full max-h-64 object-contain rounded-lg border border-gray-300 bg-gray-50"
          />
        </div>
      )}

      {initImage && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Strength</Label>
            <span className="text-sm text-gray-500">{strength.toFixed(2)}</span>
          </div>
          <Slider
            value={[strength]}
            onValueChange={(v) => setStrength(v[0])}
            min={0.1}
            max={1.0}
            step={0.05}
            disabled={isGenerating}
          />
          <p className="text-xs text-gray-500">
            Higher = more changes from init image
          </p>
        </div>
      )}
    </div>
  )
}
