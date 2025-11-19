import * as React from "react"

interface SliderProps {
  value: number[]
  onValueChange: (value: number[]) => void
  min: number
  max: number
  step: number
  disabled?: boolean
  className?: string
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ value, onValueChange, min, max, step, disabled, className }, ref) => {
    return (
      <input
        ref={ref}
        type="range"
        value={value[0]}
        onChange={(e) => onValueChange([parseFloat(e.target.value)])}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={`w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${className || ''}`}
      />
    )
  }
)
Slider.displayName = "Slider"

export { Slider }
