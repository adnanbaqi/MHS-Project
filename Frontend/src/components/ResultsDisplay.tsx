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
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-4 w-full">
      {/* Risk prediction card */}
      {prediction && (
        <motion.div variants={item} className={`rounded-xl border p-5 ${riskBg(prediction.risk_level)}`}>
          <div className="flex items-center gap-3 mb-3">
            <Shield className={`w-5 h-5 ${riskColor(prediction.risk_level)}`} />
            <div>
              <h3 className="font-heading font-bold text-foreground text-sm">Risk Assessment</h3>
              <p className={`text-xs font-semibold ${riskColor(prediction.risk_level)}`}>
                {prediction.risk_level}
              </p>
            </div>
            <div className="ml-auto text-right">
              <motion.p
                className={`text-2xl font-mono font-bold ${riskColor(prediction.risk_level)}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.3 }}
              >
                {(prediction.risk_score * 100).toFixed(0)}%
              </motion.p>
              <p className="text-[10px] text-muted-foreground">
                {(prediction.confidence * 100).toFixed(0)}% confidence
              </p>
            </div>
          </div>
          <p className="text-sm text-foreground/80 leading-relaxed">{prediction.recommendation}</p>
        </motion.div>
      )}

      {/* Clinical scores */}
      {clinical && (
        <motion.div variants={item} className="bg-card border rounded-xl p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-primary" />
            <h3 className="font-heading font-bold text-foreground text-sm">Clinical Scores</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-secondary/50 p-3">
              <p className="text-[11px] text-muted-foreground mb-1">GAD-7 Score</p>
              <p className="text-xl font-mono font-bold text-foreground">{clinical.gad_score}</p>
              <p className="text-[11px] font-medium text-primary mt-1">{clinical.gad_severity}</p>
              <div className="mt-2 h-1 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(clinical.gad_score / 21) * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                />
              </div>
            </div>
            <div className="rounded-lg bg-secondary/50 p-3">
              <p className="text-[11px] text-muted-foreground mb-1">PHQ-9 Score</p>
              <p className="text-xl font-mono font-bold text-foreground">{clinical.phq_score}</p>
              <p className="text-[11px] font-medium text-primary mt-1">{clinical.phq_severity}</p>
              <div className="mt-2 h-1 rounded-full bg-muted overflow-hidden">
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
        <motion.div variants={item} className="bg-card border rounded-xl p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare className="w-4 h-4 text-primary" />
            <h3 className="font-heading font-bold text-foreground text-sm">Text Analysis</h3>
          </div>
          <div className="space-y-2.5">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground text-xs">Sentiment</span>
              <span className="font-medium text-foreground text-xs">{textAnalysis.sentiment_label}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground text-xs">Word count</span>
              <span className="font-mono text-foreground text-xs">{textAnalysis.word_count}</span>
            </div>
            {textAnalysis.negative_keywords_found.length > 0 && (
              <div className="pt-2 border-t mt-2">
                <div className="flex items-center gap-1.5 mb-2">
                  <AlertTriangle className="w-3 h-3 text-warning" />
                  <span className="text-[11px] text-muted-foreground">Flagged keywords</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {textAnalysis.negative_keywords_found.map((kw, i) => (
                    <motion.span
                      key={kw}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.6 + i * 0.05 }}
                      className="text-[10px] px-2 py-1 rounded-md bg-warning/10 text-warning border border-warning/20"
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

      <motion.div variants={item} className="flex justify-center pt-2">
        <Button variant="outline" size="sm" onClick={onReset} className="gap-2">
          <RefreshCcw className="w-3 h-3" /> Start Over
        </Button>
      </motion.div>
    </motion.div>
  );
};

export default ResultsDisplay;