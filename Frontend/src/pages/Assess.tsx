import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Zap, Settings, Activity, ArrowLeft } from "lucide-react";
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
      <div className="min-h-screen">
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50"
        >
          <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button onClick={() => navigate("/")} className="p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground">
                <ArrowLeft className="w-4 h-4" />
              </button>
              <div className="p-2 rounded-xl bg-primary/10">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="text-base font-heading font-bold text-foreground">MindScan</h1>
                <p className="text-xs text-muted-foreground">AI Mental Health Assessment</p>
              </div>
            </div>
            <motion.div
              className="w-2 h-2 rounded-full bg-success"
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ repeat: Infinity, duration: 2 }}
            />
          </div>
        </motion.header>

        <div className="max-w-3xl mx-auto px-6 pt-10 pb-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="flex gap-2 justify-center mb-10"
          >
            {[
              { id: "full" as Tab, label: "Full Assessment", icon: Brain, desc: "GAD-7 + PHQ-9 + Text" },
              { id: "quick" as Tab, label: "Quick Screen", icon: Zap, desc: "GAD-7 + PHQ-9 only" },
            ].map((t) => (
              <motion.button
                key={t.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-medium transition-all ${tab === t.id
                  ? "glass-elevated text-foreground glow-primary"
                  : "bg-secondary text-muted-foreground hover:text-foreground"
                  }`}
              >
                <t.icon className="w-4 h-4" />
                <div className="text-left">
                  <p className="font-semibold">{t.label}</p>
                  <p className="text-[10px] opacity-70">{t.desc}</p>
                </div>
              </motion.button>
            ))}
          </motion.div>

          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
            >
              <AssessmentForm apiBaseUrl={API_BASE_URL} mode={tab} />
            </motion.div>
          </AnimatePresence>
        </div>

        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="border-t border-border mt-16 py-6"
        >
          <div className="max-w-3xl mx-auto px-6 flex items-center justify-between text-xs text-muted-foreground">
            <p>For clinical screening purposes only</p>
            <div className="flex items-center gap-1.5">
              <Settings className="w-3 h-3" />
              <span>v1.0.0</span>
            </div>
          </div>
        </motion.footer>
      </div>
    </PageTransition>
  );
};

export default Assess;
