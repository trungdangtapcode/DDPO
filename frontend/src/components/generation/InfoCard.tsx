/**
 * Component for displaying information and tips.
 */

import { Card, CardContent } from '@/components/ui/card'
import { Info } from 'lucide-react'

export function InfoCard() {
  return (
    <Card className="mt-6">
      <CardContent className="p-6">
        <div className="flex items-start space-x-3">
          <Info className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-gray-600 space-y-2">
            <p>
              <strong>CLIP Image Prompts:</strong> Upload an image instead of using text.
              The model will use CLIP embeddings to understand the image content.
            </p>
            <p>
              <strong>Init Image:</strong> Start from an existing image and transform it.
              Strength controls how much it changes.
            </p>
            <p>
              <strong>Noise Controls:</strong> Advanced features to inject noise at specific
              steps or start from a partially denoised image.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
