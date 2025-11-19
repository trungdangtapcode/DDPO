/**
 * Application header component.
 */

import { Wand2 } from 'lucide-react'

export function Header() {
  return (
    <header className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 shadow-lg">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center space-x-3">
          <Wand2 className="h-8 w-8" />
          <div>
            <h1 className="text-3xl font-bold">AI Image Generator</h1>
            <p className="text-sm opacity-90">
              Powered by Stable Diffusion with DDPO optimization
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}
