import { motion } from "framer-motion";
import { Shield, Brain, MessageSquare, AlertTriangle, RefreshCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { AssessmentOutput } from "@/lib/api";

interface Props {
  data: AssessmentOutput;
  mode: "full" | "quick";
  onReset: () => void;
}

const riskColor = (level: string) => {
  const l = level.toLowerCase();
  if (l.includes("low") || l.includes("minimal")) return "text-risk-low";
  if (l.includes("moderate") || l.includes("mild")) return "text-risk-moderate";
  if (l.includes("high") || l.includes("moderately")) return "text-risk-high";
  return "text-risk-critical";
};

const riskBg = (level: string) => {
  const l = level.toLowerCase();
  if (l.includes("low") || l.includes("minimal")) return "bg-risk-low/10 border-risk-low/20";
  if (l.includes("moderate") || l.includes("mild")) return "bg-risk-moderate/10 border-risk-moderate/20";
  if (l.includes("high") || l.includes("moderately")) return "bg-risk-high/10 border-risk-high/20";
  return "bg-risk-critical/10 border-risk-critical/20";
};

const ResultsDisplay = ({ data, mode, onReset }: Props) => {
  const clinical = data.clinical;
  const prediction = data.prediction;
  const textAnalysis = data.text_analysis;

  const container = {
    hidden: {},
    show: { transition: { staggerChildren: 0.12 } },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-4">
      {/* Risk prediction card */}
      {prediction && (
        <motion.div
          variants={item}
          className={`rounded-xl border p-6 ${riskBg(prediction.risk_level)}`}
        >
          <div className="flex items-center gap-3 mb-4">
            <Shield className={`w-6 h-6 ${riskColor(prediction.risk_level)}`} />
            <div>
              <h3 className="font-heading font-bold text-foreground">Risk Assessment</h3>
              <p className={`text-sm font-semibold ${riskColor(prediction.risk_level)}`}>
                {prediction.risk_level}
              </p>
            </div>
            <div className="ml-auto text-right">
              <motion.p
                className={`text-3xl font-mono font-bold ${riskColor(prediction.risk_level)}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.3 }}
              >
                {(prediction.risk_score * 100).toFixed(0)}%
              </motion.p>
              <p className="text-xs text-muted-foreground">
                {(prediction.confidence * 100).toFixed(0)}% confidence
              </p>
            </div>
          </div>
          <p className="text-sm text-foreground/80">{prediction.recommendation}</p>
        </motion.div>
      )}

      {/* Clinical scores */}
      {clinical && (
        <motion.div variants={item} className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-5 h-5 text-primary" />
            <h3 className="font-heading font-bold text-foreground">Clinical Scores</h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg bg-secondary p-4">
              <p className="text-xs text-muted-foreground mb-1">GAD-7 Score</p>
              <p className="text-2xl font-mono font-bold text-foreground">{clinical.gad_score}</p>
              <p className="text-xs font-medium text-primary mt-1">{clinical.gad_severity}</p>
              <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(clinical.gad_score / 21) * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                />
              </div>
            </div>
            <div className="rounded-lg bg-secondary p-4">
              <p className="text-xs text-muted-foreground mb-1">PHQ-9 Score</p>
              <p className="text-2xl font-mono font-bold text-foreground">{clinical.phq_score}</p>
              <p className="text-xs font-medium text-primary mt-1">{clinical.phq_severity}</p>
              <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(clinical.phq_score / 27) * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.5 }}
                />
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Text analysis */}
      {mode === "full" && textAnalysis && (
        <motion.div variants={item} className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <MessageSquare className="w-5 h-5 text-primary" />
            <h3 className="font-heading font-bold text-foreground">Text Analysis</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Sentiment</span>
              <span className="font-medium text-foreground">{textAnalysis.sentiment_label}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Word count</span>
              <span className="font-mono text-foreground">{textAnalysis.word_count}</span>
            </div>
            {textAnalysis.negative_keywords_found.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <AlertTriangle className="w-3.5 h-3.5 text-warning" />
                  <span className="text-xs text-muted-foreground">Flagged keywords</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {textAnalysis.negative_keywords_found.map((kw, i) => (
                    <motion.span
                      key={kw}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.6 + i * 0.05 }}
                      className="text-xs px-2 py-1 rounded-md bg-warning/10 text-warning border border-warning/20"
                    >
                      {kw}
                    </motion.span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      <motion.div variants={item} className="flex justify-center pt-4">
        <Button variant="outline" onClick={onReset} className="gap-2">
          <RefreshCcw className="w-4 h-4" /> Start Over
        </Button>
      </motion.div>
    </motion.div>
  );
};

export default ResultsDisplay;
