/**
 * Application constants.
 */

import { ModelOption } from '@/types/generation'

export const AVAILABLE_MODELS: ModelOption[] = [
  { value: 'aesthetic', label: 'Aesthetic Quality' },
  { value: 'alignment', label: 'Text Alignment' },
  { value: 'compressibility', label: 'Compressibility' },
  { value: 'incompressibility', label: 'Incompressibility' },
]

export const MODEL_DESCRIPTIONS: Record<string, string> = {
  aesthetic: '📸 Optimized for visual appeal and beauty',
  alignment: '🎯 Optimized for accurate prompt matching',
  compressibility: '💾 Optimized for smaller file sizes',
  incompressibility: '✨ Optimized for maximum detail',
}

export const NOISE_MASK_OPTIONS = [
  { value: 'none', label: 'None (disabled)' },
  { value: 'center_circle', label: 'Center Circle' },
  { value: 'center_square', label: 'Center Square' },
  { value: 'edges', label: 'Edges/Border' },
  { value: 'corners', label: 'Four Corners' },
  { value: 'left_half', label: 'Left Half' },
  { value: 'right_half', label: 'Right Half' },
  { value: 'top_half', label: 'Top Half' },
  { value: 'bottom_half', label: 'Bottom Half' },
  { value: 'checkerboard', label: 'Checkerboard' },
]

export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
