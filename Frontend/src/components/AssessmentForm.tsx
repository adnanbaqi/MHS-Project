import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Paperclip, Send, Loader2, Bot, User, AlertCircle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import ResultsDisplay from "./ResultsDisplay";
import type { AssessmentOutput } from "@/lib/api";

const GAD7_QUESTIONS = [
  "Feeling nervous, anxious, or on edge?",
  "Not being able to stop or control worrying?",
  "Worrying too much about different things?",
  "Trouble relaxing?",
  "Being so restless that it's hard to sit still?",
  "Becoming easily annoyed or irritable?",
  "Feeling afraid as if something awful might happen?",
];

const PHQ9_QUESTIONS = [
  "Little interest or pleasure in doing things?",
  "Feeling down, depressed, or hopeless?",
  "Trouble falling/staying asleep, or sleeping too much?",
  "Feeling tired or having little energy?",
  "Poor appetite or overeating?",
  "Feeling bad about yourself or that you're a failure?",
  "Trouble concentrating on things?",
  "Moving or speaking slowly, or being fidgety/restless?",
  "Thoughts that you would be better off dead or of hurting yourself?",
];

const RATING_OPTIONS = [
  { value: 0, label: "Not at all" },
  { value: 1, label: "Several days" },
  { value: 2, label: "More than half the days" },
  { value: 3, label: "Nearly every day" },
];

type Phase = "gad7" | "phq9" | "context" | "processing" | "results";

interface Message {
  id: string;
  role: "bot" | "user";
  content: string | React.ReactNode;
  isQuickReply?: boolean;
}

interface AssessmentFormProps {
  apiBaseUrl: string;
  mode: "full" | "quick";
}

const AssessmentForm = ({ apiBaseUrl, mode }: AssessmentFormProps) => {
  const [gad7, setGad7] = useState<number[]>([]);
  const [phq9, setPhq9] = useState<number[]>([]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const [phase, setPhase] = useState<Phase>("gad7");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [results, setResults] = useState<AssessmentOutput | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, phase]);

  // Helper to simulate bot typing delay
  const pushBotMessage = (content: string | React.ReactNode, isQuickReply = false, delay = 800) => {
    setIsTyping(true);
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString() + Math.random(), role: "bot", content, isQuickReply },
      ]);
      setIsTyping(false);
    }, delay);
  };

  const pushUserMessage = (content: string | React.ReactNode) => {
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString() + Math.random(), role: "user", content },
    ]);
  };

  // Initialize chat
  useEffect(() => {
    pushBotMessage("Hi there. I'm here to help assess your well-being. Let's start with a few quick questions about how you've been feeling over the last 2 weeks.", false, 400);
    setTimeout(() => {
      pushBotMessage(`Over the last 2 weeks, how often have you been bothered by: **${GAD7_QUESTIONS[0]}**`, true, 600);
    }, 1000);
  }, []);

  const handleOptionSelect = (value: number, label: string) => {
    // Disable current quick replies immediately to prevent double clicks
    setMessages((prev) => prev.map((m, i) => i === prev.length - 1 ? { ...m, isQuickReply: false } : m));

    pushUserMessage(label);

    if (phase === "gad7") {
      const newGad7 = [...gad7, value];
      setGad7(newGad7);

      if (newGad7.length < GAD7_QUESTIONS.length) {
        pushBotMessage(`**${GAD7_QUESTIONS[newGad7.length]}**`, true);
      } else {
        setPhase("phq9");
        pushBotMessage("Thank you. Now let's move on to the next set. Over the last 2 weeks, how often have you been bothered by:");
        setTimeout(() => {
          pushBotMessage(`**${PHQ9_QUESTIONS[0]}**`, true);
        }, 1200);
      }
    } else if (phase === "phq9") {
      const newPhq9 = [...phq9, value];
      setPhq9(newPhq9);

      if (newPhq9.length < PHQ9_QUESTIONS.length) {
        pushBotMessage(`**${PHQ9_QUESTIONS[newPhq9.length]}**`, true);
      } else {
        if (mode === "full") {
          setPhase("context");
          pushBotMessage(
            "Thanks for answering those. Finally, how have you been feeling recently? Feel free to write a short journal entry or share what's on your mind. You can also upload any relevant health files if needed."
          );
        } else {
          submitAssessment(gad7, newPhq9, "");
        }
      }
    }
  };

  const handleContextSubmit = () => {
    if (!text.trim() && !file) return;

    let userMsg = text;
    if (file) userMsg += `\n[Attached File: ${file.name}]`;

    pushUserMessage(userMsg);
    submitAssessment(gad7, phq9, text);
  };

  const submitAssessment = async (finalGad7: number[], finalPhq9: number[], finalText: string) => {
    setPhase("processing");
    setLoading(true);
    pushBotMessage("Processing your clinical and text data... Please wait.");

    try {
      const url = mode === "full" ? `${apiBaseUrl}/api/analyze` : `${apiBaseUrl}/api/quick-screen`;
      const payload = mode === "full" ? { gad7: finalGad7, phq9: finalPhq9, text: finalText } : { gad7: finalGad7, phq9: finalPhq9 };

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
      // Wait a moment for UX before snapping to results
      setTimeout(() => setPhase("results"), 1000);
    } catch (e: unknown) {
      let errorMsg = "An unexpected error occurred.";
      if (e instanceof Error) errorMsg = e.message;

      pushBotMessage((
        <div className="flex items-center text-destructive">
          <AlertCircle className="w-4 h-4 mr-2" />
          {errorMsg}
        </div>
      ));
      setPhase("context");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const reset = () => {
    setGad7([]);
    setPhq9([]);
    setText("");
    setFile(null);
    setPhase("gad7");
    setResults(null);
    setMessages([]);
    pushBotMessage("Let's start over. Over the last 2 weeks, how often have you been bothered by:", false, 400);
    setTimeout(() => {
      pushBotMessage(`**${GAD7_QUESTIONS[0]}**`, true, 600);
    }, 1000);
  };

  // If we have results, completely replace the chat with the ResultsDisplay
  if (phase === "results" && results) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.98, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        className="w-full max-w-3xl mx-auto py-8"
      >
        <div className="text-center mb-8">
          <div className="inline-flex justify-center items-center p-3 rounded-full bg-success/10 text-success mb-4">
            <CheckCircle2 className="w-8 h-8" />
          </div>
          <h2 className="text-2xl font-bold font-heading text-foreground">Assessment Complete</h2>
          <p className="text-muted-foreground mt-2 text-sm">Thank you for sharing. Here is your personalized analysis.</p>
        </div>
        <ResultsDisplay data={results} mode={mode} onReset={reset} />
      </motion.div>
    );
  }

  // Normal Chat Interface
  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col h-[calc(100vh-180px)] min-h-[500px]">

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-6">
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => {
            const isBot = msg.role === "bot";
            const isLastMessage = idx === messages.length - 1;

            return (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`flex gap-3 ${isBot ? "flex-row" : "flex-row-reverse"}`}
              >
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 ${isBot ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
                  {isBot ? <Bot className="w-4 h-4" /> : <User className="w-4 h-4" />}
                </div>

                {/* Message Bubble */}
                <div className={`max-w-[85%] flex flex-col gap-2 ${isBot ? "items-start" : "items-end"}`}>
                  <div className={`px-4 py-3 rounded-2xl ${isBot ? "bg-secondary/60 text-secondary-foreground rounded-tl-sm" : "bg-primary text-primary-foreground rounded-tr-sm"}`}>
                    {typeof msg.content === "string" ? (
                      <span className="leading-relaxed" dangerouslySetInnerHTML={{ __html: msg.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                    ) : (
                      msg.content
                    )}
                  </div>

                  {/* Render Quick Replies */}
                  {isBot && msg.isQuickReply && isLastMessage && !isTyping && (phase === "gad7" || phase === "phq9") && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 w-full max-w-[400px]"
                    >
                      {RATING_OPTIONS.map((opt) => (
                        <button
                          key={opt.value}
                          onClick={() => handleOptionSelect(opt.value, opt.label)}
                          className="p-3 text-sm text-center border bg-background hover:border-primary/50 hover:bg-primary/5 rounded-xl transition-all font-medium text-foreground shadow-sm"
                        >
                          {opt.label} <span className="opacity-50 text-xs ml-1 font-normal">({opt.value})</span>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center mt-1">
              <Bot className="w-4 h-4" />
            </div>
            <div className="px-4 py-4 rounded-2xl bg-secondary/60 rounded-tl-sm flex items-center gap-1.5 w-16 justify-center">
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '0ms' }}></span>
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '150ms' }}></span>
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '300ms' }}></span>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} className="h-1" />
      </div>

      {/* Input Area (Only visible during context phase) */}
      {phase === "context" && mode === "full" && !loading && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="p-4 bg-background/80 backdrop-blur-sm border-t">
          <div className="max-w-3xl mx-auto">
            {file && (
              <div className="mb-2 text-xs flex items-center text-primary bg-primary/10 px-3 py-1.5 rounded-md inline-flex">
                <Paperclip className="w-3 h-3 mr-2" /> {file.name}
                <button onClick={() => setFile(null)} className="ml-2 font-bold hover:text-destructive">×</button>
              </div>
            )}

            <div className="flex items-end gap-2 relative">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Share your thoughts, feelings, or attach a file..."
                className="w-full max-h-32 min-h-[56px] bg-secondary/30 border border-muted-foreground/20 rounded-2xl py-3 px-4 pr-12 resize-none focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all"
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleContextSubmit();
                  }
                }}
              />
              <input
                type="file"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileChange}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="absolute bottom-3.5 right-[70px] text-muted-foreground hover:text-primary transition-colors"
                title="Attach file"
              >
                <Paperclip className="w-5 h-5" />
              </button>

              <Button
                onClick={handleContextSubmit}
                disabled={!text.trim() && !file}
                className="rounded-2xl h-[56px] w-[56px] flex-shrink-0 shadow-md"
              >
                <Send className="w-5 h-5 ml-1" />
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AssessmentForm;
