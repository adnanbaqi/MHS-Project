import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, FileText, Zap, ArrowRight, ArrowLeft, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import ScoreInput from "./ScoreInput";
import ResultsDisplay from "./ResultsDisplay";
import type { AssessmentOutput } from "@/lib/api";

const GAD7_QUESTIONS = [
  "Feeling nervous, anxious, or on edge",
  "Not being able to stop or control worrying",
  "Worrying too much about different things",
  "Trouble relaxing",
  "Being so restless that it's hard to sit still",
  "Becoming easily annoyed or irritable",
  "Feeling afraid as if something awful might happen",
];

const PHQ9_QUESTIONS = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling/staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself or that you're a failure",
  "Trouble concentrating on things",
  "Moving or speaking slowly, or being fidgety/restless",
  "Thoughts that you would be better off dead or of hurting yourself",
];

type Step = "gad7" | "phq9" | "text" | "results";

interface AssessmentFormProps {
  apiBaseUrl: string;
  mode: "full" | "quick";
}

const AssessmentForm = ({ apiBaseUrl, mode }: AssessmentFormProps) => {
  const [gad7, setGad7] = useState<number[]>(Array(7).fill(0));
  const [phq9, setPhq9] = useState<number[]>(Array(9).fill(0));
  const [text, setText] = useState("");
  const [step, setStep] = useState<Step>("gad7");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AssessmentOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const steps: Step[] = mode === "full" ? ["gad7", "phq9", "text", "results"] : ["gad7", "phq9", "results"];
  const currentIdx = steps.indexOf(step);

  const updateGad7 = (i: number, v: number) => setGad7((p) => p.map((x, j) => (j === i ? v : x)));
  const updatePhq9 = (i: number, v: number) => setPhq9((p) => p.map((x, j) => (j === i ? v : x)));

  const gadTotal = gad7.reduce((a, b) => a + b, 0);
  const phqTotal = phq9.reduce((a, b) => a + b, 0);

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = mode === "full" ? `${apiBaseUrl}/api/analyze` : `${apiBaseUrl}/api/quick-screen`;

      // Unify the payload. Both endpoints should receive clean JSON.
      const payload = mode === "full"
        ? { gad7, phq9, text }
        : { gad7, phq9 };

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        // Try to grab the exact FastAPI validation error for easier debugging
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail?.[0]?.msg || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
      setStep("results");
    } catch (e: any) {
      setError(e.message || "Failed to connect to the server.");
    } finally {
      setLoading(false);
    }
  };

  const canProceed = () => {
    if (step === "text" && mode === "full") return text.trim().length > 0;
    return true;
  };

  const next = () => {
    const nextIdx = currentIdx + 1;
    if (steps[nextIdx] === "results") {
      submit();
    } else {
      setStep(steps[nextIdx]);
    }
  };

  const prev = () => {
    if (currentIdx > 0) setStep(steps[currentIdx - 1]);
  };

  const reset = () => {
    setGad7(Array(7).fill(0));
    setPhq9(Array(9).fill(0));
    setText("");
    setStep("gad7");
    setResults(null);
    setError(null);
  };

  const stepInfo = {
    gad7: { icon: Brain, title: "GAD-7 Assessment", desc: "Generalized Anxiety Disorder scale" },
    phq9: { icon: Brain, title: "PHQ-9 Assessment", desc: "Patient Health Questionnaire" },
    text: { icon: FileText, title: "Journal Entry", desc: "Share your thoughts for NLP analysis" },
    results: { icon: Zap, title: "Results", desc: "Your assessment results" },
  };

  const info = stepInfo[step];
  const Icon = info.icon;

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Progress bar */}
      <div className="flex gap-2 mb-8">
        {steps.map((s, i) => (
          <motion.div
            key={s}
            className="flex-1 h-1.5 rounded-full overflow-hidden bg-secondary"
          >
            <motion.div
              className="h-full bg-primary rounded-full"
              initial={{ width: "0%" }}
              animate={{ width: i <= currentIdx ? "100%" : "0%" }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            />
          </motion.div>
        ))}
      </div>

      {/* Step header */}
      <motion.div
        key={step}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-3 mb-6"
      >
        <div className="p-2.5 rounded-xl bg-primary/10">
          <Icon className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-heading font-bold text-foreground">{info.title}</h2>
          <p className="text-sm text-muted-foreground">{info.desc}</p>
        </div>
        {(step === "gad7" || step === "phq9") && (
          <motion.div
            className="ml-auto font-mono text-2xl font-bold text-primary"
            key={step === "gad7" ? gadTotal : phqTotal}
            initial={{ scale: 1.3 }}
            animate={{ scale: 1 }}
          >
            {step === "gad7" ? gadTotal : phqTotal}
            <span className="text-xs text-muted-foreground font-body ml-1">
              /{step === "gad7" ? 21 : 27}
            </span>
          </motion.div>
        )}
      </motion.div>

      {/* Content */}
      <AnimatePresence mode="wait">
        {step === "gad7" && (
          <motion.div
            key="gad7"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
            className="space-y-3"
          >
            {GAD7_QUESTIONS.map((q, i) => (
              <ScoreInput key={i} label={`${i + 1}. ${q}`} index={i} value={gad7[i]} onChange={(v) => updateGad7(i, v)} />
            ))}
          </motion.div>
        )}

        {step === "phq9" && (
          <motion.div
            key="phq9"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
            className="space-y-3"
          >
            {PHQ9_QUESTIONS.map((q, i) => (
              <ScoreInput key={i} label={`${i + 1}. ${q}`} index={i} value={phq9[i]} onChange={(v) => updatePhq9(i, v)} />
            ))}
          </motion.div>
        )}

        {step === "text" && (
          <motion.div
            key="text"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
          >
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Write about how you've been feeling recently..."
              className="w-full h-48 rounded-lg glass-card p-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none"
              maxLength={2000}
            />
            <p className="text-xs text-muted-foreground mt-2 text-right">{text.length}/2000</p>
          </motion.div>
        )}

        {step === "results" && results && (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <ResultsDisplay data={results} mode={mode} onReset={reset} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm"
        >
          {error}
        </motion.div>
      )}

      {/* Navigation */}
      {step !== "results" && (
        <div className="flex justify-between mt-8">
          <Button
            variant="ghost"
            onClick={prev}
            disabled={currentIdx === 0}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" /> Back
          </Button>

          <Button
            onClick={next}
            disabled={!canProceed() || loading}
            className="gap-2"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : steps[currentIdx + 1] === "results" ? (
              <>Analyze <Zap className="w-4 h-4" /></>
            ) : (
              <>Next <ArrowRight className="w-4 h-4" /></>
            )}
          </Button>
        </div>
      )}
    </div>
  );
};

export default AssessmentForm;
