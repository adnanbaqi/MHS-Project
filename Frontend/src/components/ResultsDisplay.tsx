import { motion, Variants } from "framer-motion";
import {
  Shield,
  Brain,
  MessageSquare,
  AlertTriangle,
  RefreshCcw,
  CheckCircle2,
  ArrowRight,
  Info
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { AssessmentOutput } from "@/lib/api";

interface Props {
  data: AssessmentOutput;
  mode: "full" | "quick";
  onReset: () => void;
}

// --- Helper Functions for Theming ---

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

const getClinicalContext = (score: number, type: "GAD" | "PHQ") => {
  if (score <= 4) return "Scores in this range suggest minimal symptoms.";
  if (score <= 9) return "Scores in this range suggest mild symptoms that may benefit from monitoring.";
  if (score <= 14) return "Scores in this range indicate moderate symptoms. Consider speaking with a professional.";
  return "Scores in this range indicate severe symptoms. We strongly recommend reaching out for support.";
};

// --- Main Component ---

const ResultsDisplay = ({ data, mode, onReset }: Props) => {
  const clinical = data.clinical;
  const prediction = data.prediction;
  const textAnalysis = data.text_analysis;

  // Animation variants correctly typed for strict TypeScript
  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.15, delayChildren: 0.1 }
    },
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-6 w-full pb-8">

      {/* --- Risk Prediction Card (The Headline) --- */}
      {prediction && (
        <motion.div variants={item} className={`rounded-2xl border p-6 shadow-sm ${riskBg(prediction.risk_level)}`}>
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`p-2.5 rounded-xl bg-background shadow-sm ${riskColor(prediction.risk_level)}`}>
                <Shield className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-heading font-bold text-foreground text-base">Overall Assessment</h3>
                <p className={`text-sm font-semibold mt-0.5 ${riskColor(prediction.risk_level)}`}>
                  {prediction.risk_level} Risk Profile
                </p>
              </div>
            </div>

            <div className="text-right">
              <motion.div
                className={`text-3xl font-mono font-bold tracking-tight ${riskColor(prediction.risk_level)}`}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.4 }}
              >
                {(prediction.risk_score * 100).toFixed(0)}<span className="text-xl">%</span>
              </motion.div>
              <p className="text-xs text-muted-foreground mt-1 flex items-center justify-end gap-1">
                <Info className="w-3 h-3" />
                {(prediction.confidence * 100).toFixed(0)}% AI confidence
              </p>
            </div>
          </div>

          <div className="mt-5 pt-5 border-t border-foreground/10">
            <h4 className="text-xs font-semibold text-foreground/70 uppercase tracking-wider mb-2 flex items-center gap-2">
              <ArrowRight className="w-3.5 h-3.5" /> Next Steps
            </h4>
            <p className="text-sm text-foreground/90 leading-relaxed font-medium">
              {prediction.recommendation}
            </p>
          </div>
        </motion.div>
      )}

      {/* --- Clinical Scores Breakdown --- */}
      {clinical && (
        <motion.div variants={item} className="bg-card border rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2.5 mb-5">
            <div className="p-2 rounded-lg bg-primary/10">
              <Brain className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-foreground text-sm">Clinical Breakdown</h3>
              <p className="text-xs text-muted-foreground">Standardized questionnaire results</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {/* GAD-7 Card */}
            <div className="rounded-xl border border-muted/50 bg-secondary/30 p-4 transition-all hover:bg-secondary/50">
              <div className="flex justify-between items-start mb-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Anxiety (GAD-7)</p>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-background shadow-sm text-primary">
                  {clinical.gad_severity}
                </span>
              </div>
              <p className="text-3xl font-mono font-bold text-foreground my-2">{clinical.gad_score}<span className="text-sm text-muted-foreground font-normal">/21</span></p>

              <div className="mt-3 h-1.5 rounded-full bg-muted overflow-hidden" role="progressbar" aria-valuenow={clinical.gad_score} aria-valuemin={0} aria-valuemax={21}>
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(clinical.gad_score / 21) * 100}%` }}
                  transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
                />
              </div>
              <p className="text-[11px] text-muted-foreground mt-3 leading-snug">
                {getClinicalContext(clinical.gad_score, "GAD")}
              </p>
            </div>

            {/* PHQ-9 Card */}
            <div className="rounded-xl border border-muted/50 bg-secondary/30 p-4 transition-all hover:bg-secondary/50">
              <div className="flex justify-between items-start mb-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Depression (PHQ-9)</p>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-background shadow-sm text-primary">
                  {clinical.phq_severity}
                </span>
              </div>
              <p className="text-3xl font-mono font-bold text-foreground my-2">{clinical.phq_score}<span className="text-sm text-muted-foreground font-normal">/27</span></p>

              <div className="mt-3 h-1.5 rounded-full bg-muted overflow-hidden" role="progressbar" aria-valuenow={clinical.phq_score} aria-valuemin={0} aria-valuemax={27}>
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(clinical.phq_score / 27) * 100}%` }}
                  transition={{ duration: 1, delay: 0.6, ease: "easeOut" }}
                />
              </div>
              <p className="text-[11px] text-muted-foreground mt-3 leading-snug">
                {getClinicalContext(clinical.phq_score, "PHQ")}
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* --- NLP Text Analysis (Deep Dive Only) --- */}
      {mode === "full" && textAnalysis && (
        <motion.div variants={item} className="bg-card border rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2.5 mb-5">
            <div className="p-2 rounded-lg bg-primary/10">
              <MessageSquare className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-foreground text-sm">Journal Analysis</h3>
              <p className="text-xs text-muted-foreground">AI sentiment and keyword extraction</p>
            </div>
          </div>

          <div className="bg-secondary/30 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-muted/50 pb-3">
              <span className="text-sm text-muted-foreground">Overall Sentiment</span>
              <span className="font-medium text-foreground px-3 py-1 bg-background rounded-full text-xs shadow-sm">
                {textAnalysis.sentiment_label}
              </span>
            </div>

            <div className="flex items-center justify-between border-b border-muted/50 pb-3">
              <span className="text-sm text-muted-foreground">Word Count Analyzed</span>
              <span className="font-mono text-foreground text-sm">{textAnalysis.word_count} words</span>
            </div>

            <div className="pt-1">
              <span className="text-sm text-muted-foreground block mb-3">Flagged Indicators</span>

              {textAnalysis.negative_keywords_found.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {textAnalysis.negative_keywords_found.map((kw, i) => (
                    <motion.span
                      key={kw}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.8 + i * 0.05 }}
                      className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md bg-warning/10 text-warning border border-warning/20 font-medium"
                    >
                      <AlertTriangle className="w-3 h-3" />
                      {kw}
                    </motion.span>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm text-success bg-success/10 px-3 py-2 rounded-lg border border-success/20">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>No high-risk keywords detected in your text.</span>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* --- Action Buttons --- */}
      <motion.div variants={item} className="flex flex-col sm:flex-row justify-center gap-3 pt-4">
        <Button onClick={onReset} className="gap-2 w-full sm:w-auto px-8" size="lg">
          <RefreshCcw className="w-4 h-4" /> Start New Assessment
        </Button>
      </motion.div>

    </motion.div>
  );
};

export default ResultsDisplay;