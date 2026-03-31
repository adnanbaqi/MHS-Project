import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Zap, Activity, ArrowLeft } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import AssessmentForm from "@/components/AssessmentForm";
import PageTransition from "@/components/PageTransition";

const API_BASE_URL = "http://localhost:8000";

type Tab = "full" | "quick";

const Assess = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>((searchParams.get("mode") as Tab) || "full");

  return (
    <PageTransition>
      <div className="min-h-screen flex flex-col bg-background">

        {/* Sleek App Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50"
        >
          <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button onClick={() => navigate("/")} className="p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="p-2 rounded-xl bg-primary/10">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="text-base font-heading font-bold text-foreground leading-tight">MindScan Chat</h1>
                <p className="text-[11px] text-muted-foreground">Conversational Assessment</p>
              </div>
            </div>

            {/* Minimalist Tab Switcher in Header */}
            <div className="hidden sm:flex p-1 bg-secondary rounded-lg">
              <button
                onClick={() => setTab("full")}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${tab === "full" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                <Brain className="w-3.5 h-3.5" /> Deep Dive
              </button>
              <button
                onClick={() => setTab("quick")}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${tab === "quick" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                <Zap className="w-3.5 h-3.5" /> Quick
              </button>
            </div>
          </div>
        </motion.header>

        {/* Mobile Tab Switcher (Visible only on small screens) */}
        <div className="sm:hidden flex p-2 bg-background border-b justify-center gap-2">
          <button
            onClick={() => setTab("full")}
            className={`flex-1 flex justify-center items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border transition-all ${tab === "full" ? "bg-primary/5 border-primary/20 text-primary" : "bg-secondary/50 border-transparent text-muted-foreground"}`}
          >
            <Brain className="w-4 h-4" /> Deep Dive
          </button>
          <button
            onClick={() => setTab("quick")}
            className={`flex-1 flex justify-center items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border transition-all ${tab === "quick" ? "bg-primary/5 border-primary/20 text-primary" : "bg-secondary/50 border-transparent text-muted-foreground"}`}
          >
            <Zap className="w-4 h-4" /> Quick
          </button>
        </div>

        {/* Chat / Results Container */}
        <div className="flex-1 w-full flex flex-col pt-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="flex-1 w-full flex flex-col"
            >
              <AssessmentForm apiBaseUrl={API_BASE_URL} mode={tab} />
            </motion.div>
          </AnimatePresence>
        </div>

      </div>
    </PageTransition>
  );
};

export default Assess;