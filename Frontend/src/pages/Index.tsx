import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Zap, Settings, Activity, MessageCircle } from "lucide-react";
import ChatAssessment from "@/components/AssessmentForm";

const API_BASE_URL = "http://localhost:8000";

type Tab = "full" | "quick";

const Index = () => {
  const [tab, setTab] = useState<Tab>("full");

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50"
      >
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-primary/10">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-base font-heading font-bold text-foreground">MindScan Chat</h1>
              <p className="text-xs text-muted-foreground">Conversational Assessment</p>
            </div>
          </div>
          <motion.div
            className="w-2 h-2 rounded-full bg-success"
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ repeat: Infinity, duration: 2 }}
          />
        </div>
      </motion.header>

      {/* Main Content Area */}
      <div className="flex-1 max-w-4xl w-full mx-auto px-4 md:px-6 pt-8 pb-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="text-center mb-8"
        >
          <motion.div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-4"
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
          >
            <MessageCircle className="w-3.5 h-3.5" />
            AI Chatbot Interface
          </motion.div>
          <h2 className="text-2xl md:text-3xl font-heading font-extrabold text-foreground tracking-tight">
            Let's Check In On Your
            <br />
            <span className="text-primary">Mental Well-being</span>
          </h2>
        </motion.div>

        {/* Tab switcher */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="flex gap-2 justify-center mb-8"
        >
          {[
            { id: "full" as Tab, label: "Deep Dive", icon: Brain, desc: "Chat + Journal Analysis" },
            { id: "quick" as Tab, label: "Quick Chat", icon: Zap, desc: "Fast Clinical Screen" },
          ].map((t) => (
            <motion.button
              key={t.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-medium transition-all ${tab === t.id
                ? "bg-primary/10 border-primary/20 border text-primary"
                : "bg-secondary border border-transparent text-muted-foreground hover:text-foreground"
                }`}
            >
              <t.icon className="w-4 h-4" />
              <div className="text-left hidden sm:block">
                <p className="font-semibold">{t.label}</p>
                <p className="text-[10px] opacity-70">{t.desc}</p>
              </div>
            </motion.button>
          ))}
        </motion.div>

        {/* Chat Interface */}
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{ duration: 0.3 }}
          >
            <ChatAssessment apiBaseUrl={API_BASE_URL} mode={tab} />
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="border-t border-border py-6 mt-auto"
      >
        <div className="max-w-4xl mx-auto px-6 flex items-center justify-between text-xs text-muted-foreground">
          <p>For clinical screening purposes only</p>
          <div className="flex items-center gap-1.5">
            <Settings className="w-3 h-3" />
            <span>v2.0.0</span>
          </div>
        </div>
      </motion.footer>
    </div>
  );
};

export default Index;