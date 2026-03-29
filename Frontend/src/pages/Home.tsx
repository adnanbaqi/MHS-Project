import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import PageTransition from "@/components/PageTransition";
import { Brain, Shield, Zap, Activity, ArrowRight, FileText, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Brain,
    title: "Clinical Questionnaires",
    desc: "Standardized GAD-7 and PHQ-9 assessments used by healthcare professionals worldwide.",
  },
  {
    icon: FileText,
    title: "NLP Text Analysis",
    desc: "AI-powered sentiment analysis identifies risk signals from free-form journal entries.",
  },
  {
    icon: BarChart3,
    title: "Risk Prediction",
    desc: "Machine learning model combines clinical and text features for accurate risk scoring.",
  },
];

const Home = () => {
  const navigate = useNavigate();

  return (
    <PageTransition>
      <div className="min-h-screen overflow-hidden">
        {/* Nav */}
        <motion.nav
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50"
        >
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-primary/10">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <span className="font-heading font-bold text-foreground text-lg">MindScan</span>
            </div>
          </div>
        </motion.nav>

        {/* Hero */}
        <section className="relative max-w-5xl mx-auto px-6 pt-20 pb-24">
          {/* Decorative blobs */}
          <div className="absolute -top-20 -right-40 w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl pointer-events-none" />
          <div className="absolute -bottom-20 -left-40 w-[400px] h-[400px] rounded-full bg-accent/5 blur-3xl pointer-events-none" />

          <div className="relative grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.15 }}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-6"
              >
                <Shield className="w-3.5 h-3.5" />
                AI-Powered Clinical Screening
              </motion.div>

              <h1 className="text-4xl md:text-5xl font-heading font-extrabold text-foreground tracking-tight leading-[1.1]">
                Mental Health
                <br />
                <span className="text-primary">Risk Assessment</span>
              </h1>

              <p className="mt-5 text-muted-foreground max-w-md leading-relaxed">
                Combining standardized clinical questionnaires with AI-powered natural language processing
                for comprehensive, evidence-based mental health screening.
              </p>

              <div className="flex gap-3 mt-8">
                <Button size="lg" onClick={() => navigate("/assess")} className="gap-2 glow-primary">
                  <Brain className="w-4 h-4" /> Full Assessment
                </Button>
                <Button size="lg" variant="outline" onClick={() => navigate("/assess?mode=quick")} className="gap-2">
                  <Zap className="w-4 h-4" /> Quick Screen
                </Button>
              </div>
            </motion.div>

            {/* Hero visual */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="hidden md:flex justify-center"
            >
              <div className="relative w-72 h-72">
                {/* Animated rings */}
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="absolute inset-0 rounded-full border border-primary/15"
                    style={{ inset: `${i * 28}px` }}
                    animate={{ rotate: i % 2 === 0 ? 360 : -360 }}
                    transition={{ duration: 20 + i * 8, repeat: Infinity, ease: "linear" }}
                  />
                ))}
                <div className="absolute inset-0 flex items-center justify-center">
                  <motion.div
                    animate={{ scale: [1, 1.08, 1] }}
                    transition={{ duration: 3, repeat: Infinity }}
                    className="p-6 rounded-2xl bg-primary/10 glow-primary"
                  >
                    <Brain className="w-12 h-12 text-primary" />
                  </motion.div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Features */}
        <section className="max-w-5xl mx-auto px-6 pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-2xl font-heading font-bold text-foreground">How It Works</h2>
            <p className="text-sm text-muted-foreground mt-2">Three layers of analysis for accurate risk assessment</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-5">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 25 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.12, duration: 0.4 }}
                whileHover={{ y: -4 }}
                className="glass-card rounded-xl p-6 group cursor-default"
              >
                <div className="p-3 rounded-xl bg-primary/10 w-fit mb-4 group-hover:glow-primary transition-shadow">
                  <f.icon className="w-5 h-5 text-primary" />
                </div>
                <h3 className="font-heading font-bold text-foreground mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="max-w-5xl mx-auto px-6 pb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-elevated rounded-2xl p-10 text-center"
          >
            <h2 className="text-2xl font-heading font-bold text-foreground mb-3">Ready to Begin?</h2>
            <p className="text-sm text-muted-foreground mb-6 max-w-md mx-auto">
              The assessment takes approximately 5 minutes and provides immediate results with clinical recommendations.
            </p>
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border py-6">
          <div className="max-w-5xl mx-auto px-6 flex items-center justify-between text-xs text-muted-foreground">
            <p>For clinical screening purposes only — not a diagnostic tool</p>
            <p>MindScan v1.0.0</p>
          </div>
        </footer>
      </div>
    </PageTransition>
  );
};

export default Home;
