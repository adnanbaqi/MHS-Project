import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, FileText, Zap, ArrowRight, ArrowLeft, Loader2, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
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

// Clinical definitions for the 0-3 scale
const RATING_OPTIONS = [
  { value: 0, label: "Not at all" },
  { value: 1, label: "Several days" },
  { value: 2, label: "More than half the days" },
  { value: 3, label: "Nearly every day" },
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
      const payload = mode === "full" ? { gad7, phq9, text } : { gad7, phq9 };

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail?.[0]?.msg || errData?.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
      setStep("results");
    } catch (e: unknown) {
      if (e instanceof Error) {
        setError(e.message);
      } else {
        setError("An unexpected error occurred.");
      }
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
      window.scrollTo({ top: 0, behavior: "smooth" });
      setStep(steps[nextIdx]);
    }
  };

  const prev = () => {
    if (currentIdx > 0) {
      window.scrollTo({ top: 0, behavior: "smooth" });
      setStep(steps[currentIdx - 1]);
    }
  };

  const reset = () => {
    setGad7(Array(7).fill(0));
    setPhq9(Array(9).fill(0));
    setText("");
    setStep("gad7");
    setResults(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const stepInfo = {
    gad7: { icon: Brain, title: "GAD-7 Assessment", desc: "Over the last 2 weeks, how often have you been bothered by..." },
    phq9: { icon: Brain, title: "PHQ-9 Assessment", desc: "Over the last 2 weeks, how often have you been bothered by..." },
    text: { icon: FileText, title: "Journal Entry", desc: "Share your thoughts for NLP analysis" },
    results: { icon: Zap, title: "Assessment Complete", desc: "Processing your clinical and text data..." },
  };

  const info = stepInfo[step];
  const Icon = info.icon;

  // Reusable Question Block Component to replace the confusing ScoreInput
  const QuestionBlock = ({
    question,
    index,
    currentValue,
    onChange
  }: {
    question: string,
    index: number,
    currentValue: number,
    onChange: (val: number) => void
  }) => (
    <div className="mb-8 p-6 rounded-xl border bg-card text-card-foreground shadow-sm">
      <h3 className="text-base font-semibold mb-4 leading-tight">
        <span className="text-muted-foreground mr-2">{index + 1}.</span>
        {question}
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {RATING_OPTIONS.map((opt) => {
          const isSelected = currentValue === opt.value;
          return (
            <button
              key={opt.value}
              onClick={() => onChange(opt.value)}
              className={`relative flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all duration-200 text-sm font-medium ${isSelected
                  ? "border-primary bg-primary/5 text-primary"
                  : "border-muted hover:border-primary/50 hover:bg-muted/50 text-muted-foreground"
                }`}
            >
              {isSelected && (
                <CheckCircle2 className="absolute top-2 right-2 w-4 h-4 text-primary" />
              )}
              <span className="text-center">{opt.label}</span>
              <span className="mt-1 text-xs opacity-70">(Score: {opt.value})</span>
            </button>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="w-full max-w-3xl mx-auto pb-12">

      {/* Hide Progress & Headers on Results Step for cleaner UI */}
      {step !== "results" && (
        <>
          {/* Progress bar */}
          <div className="flex gap-2 mb-8">
            {steps.filter(s => s !== "results").map((s, i) => (
              <div key={s} className="flex-1 h-2 rounded-full overflow-hidden bg-secondary">
                <motion.div
                  className="h-full bg-primary rounded-full"
                  initial={{ width: "0%" }}
                  animate={{ width: i < currentIdx ? "100%" : i === currentIdx ? "50%" : "0%" }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                />
              </div>
            ))}
          </div>

          {/* Step header */}
          <motion.div
            key={`header-${step}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-4 mb-8 p-4 rounded-2xl bg-secondary/30"
          >
            <div className="p-3 rounded-xl bg-primary/10">
              <Icon className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-foreground">{info.title}</h2>
              <p className="text-sm text-muted-foreground mt-1">{info.desc}</p>
            </div>
            {(step === "gad7" || step === "phq9") && (
              <div className="ml-auto text-right">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Current Score</div>
                <motion.div
                  className="font-mono text-3xl font-bold text-primary"
                  key={step === "gad7" ? gadTotal : phqTotal}
                  initial={{ scale: 1.2, color: "hsl(var(--primary))" }}
                  animate={{ scale: 1, color: "hsl(var(--primary))" }}
                >
                  {step === "gad7" ? gadTotal : phqTotal}
                  <span className="text-sm text-muted-foreground font-sans ml-1">
                    /{step === "gad7" ? 21 : 27}
                  </span>
                </motion.div>
              </div>
            )}
          </motion.div>
        </>
      )}

      {/* Main Content Area */}
      <AnimatePresence mode="wait">
        {step === "gad7" && (
          <motion.div
            key="gad7"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            {GAD7_QUESTIONS.map((q, i) => (
              <QuestionBlock key={`gad-${i}`} question={q} index={i} currentValue={gad7[i]} onChange={(v) => updateGad7(i, v)} />
            ))}
          </motion.div>
        )}

        {step === "phq9" && (
          <motion.div
            key="phq9"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            {PHQ9_QUESTIONS.map((q, i) => (
              <QuestionBlock key={`phq-${i}`} question={q} index={i} currentValue={phq9[i]} onChange={(v) => updatePhq9(i, v)} />
            ))}
          </motion.div>
        )}

        {step === "text" && (
          <motion.div
            key="text"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="p-6 rounded-xl border bg-card shadow-sm"
          >
            <h3 className="text-lg font-semibold mb-4">How have you been feeling recently?</h3>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Take a moment to write down your thoughts, feelings, or anything that has been bothering you lately. (Minimum a few words required)"
              className="w-full h-56 rounded-lg border-2 border-muted bg-background p-4 text-base text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary resize-none transition-colors"
              maxLength={2000}
            />
            <div className="flex justify-between items-center mt-3">
              <span className="text-xs text-muted-foreground">
                {text.trim().length === 0 ? "Required for analysis" : ""}
              </span>
              <span className={`text-xs ${text.length > 1900 ? "text-destructive" : "text-muted-foreground"}`}>
                {text.length} / 2000 characters
              </span>
            </div>
          </motion.div>
        )}

        {step === "results" && results && (
          <motion.div
            key="results"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <ResultsDisplay data={results} mode={mode} onReset={reset} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Toast/Banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-6 p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm font-medium flex items-center">
              <Zap className="w-4 h-4 mr-2 flex-shrink-0" />
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navigation Footer */}
      {step !== "results" && (
        <div className="flex justify-between items-center mt-8 pt-6 border-t border-border">
          <Button
            variant="outline"
            onClick={prev}
            disabled={currentIdx === 0}
            className="gap-2 px-6"
          >
            <ArrowLeft className="w-4 h-4" /> Back
          </Button>

          <Button
            onClick={next}
            disabled={!canProceed() || loading}
            size="lg"
            className="gap-2 px-8"
          >
            {loading ? (
              <>Processing <Loader2 className="w-4 h-4 animate-spin ml-2" /></>
            ) : steps[currentIdx + 1] === "results" ? (
              <>Analyze Results <Zap className="w-4 h-4 ml-1" /></>
            ) : (
              <>Continue <ArrowRight className="w-4 h-4 ml-1" /></>
            )}
          </Button>
        </div>
      )}
    </div>
  );
};

export default AssessmentForm;