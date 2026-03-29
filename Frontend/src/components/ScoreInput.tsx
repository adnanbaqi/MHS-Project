import { motion } from "framer-motion";

const OPTIONS = [0, 1, 2, 3] as const;
const LABELS = ["Not at all", "Several days", "More than half", "Nearly every day"];

interface ScoreInputProps {
  label: string;
  index: number;
  value: number;
  onChange: (value: number) => void;
}

const ScoreInput = ({ label, index, value, onChange }: ScoreInputProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className="glass-card rounded-lg p-4"
    >
      <p className="text-sm font-medium text-foreground mb-3">{label}</p>
      <div className="flex gap-2">
        {OPTIONS.map((opt) => (
          <motion.button
            key={opt}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onChange(opt)}
            className={`flex-1 rounded-md py-2 text-xs font-medium transition-colors ${
              value === opt
                ? "bg-primary text-primary-foreground glow-primary"
                : "bg-secondary text-secondary-foreground hover:bg-muted"
            }`}
            title={LABELS[opt]}
          >
            {opt}
          </motion.button>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground mt-1.5 text-center">{LABELS[value]}</p>
    </motion.div>
  );
};

export default ScoreInput;
