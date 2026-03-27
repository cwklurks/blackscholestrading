interface ParamInputProps {
  label: string;
  value: number;
  step: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
}

export function ParamInput({ label, value, step, min, max, onChange }: ParamInputProps) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-muted-foreground">
        {label}
      </label>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const parsed = parseFloat(e.target.value);
          if (!Number.isNaN(parsed)) onChange(parsed);
        }}
        className="h-10 w-full rounded-[var(--radius)] border border-input bg-input px-2.5 font-mono text-sm tabular-nums text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
      />
    </div>
  );
}
