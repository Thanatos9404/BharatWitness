"use client";

import { motion } from "framer-motion";
import { Button } from "./ui/Button";
import { ArrowRight, Scale, FileText, Shield } from "lucide-react";

interface HeroProps {
  onGetStarted: () => void;
}

export const Hero = ({ onGetStarted }: HeroProps) => {
  return (
    <div className="relative min-h-screen flex items-center justify-center px-4 py-20">
      <div className="max-w-7xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/10 mb-8"
          >
            <Shield className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">
              NCIIPC AI Grand Challenge Winner
            </span>
          </motion.div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
            <span className="gradient-text">BharatWitness</span>
            <br />
            <span className="text-foreground">
              AI-Powered Legal Intelligence
            </span>
          </h1>

          {/* Subheading */}
          <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto mb-8 leading-relaxed">
            Production-grade RAG system for Indian government policy and legal
            documents with{" "}
            <span className="text-primary font-semibold">
              temporal reasoning
            </span>
            ,{" "}
            <span className="text-primary font-semibold">
              span-level provenance
            </span>
            , and{" "}
            <span className="text-primary font-semibold">
              multi-lingual support
            </span>
            .
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Button
              size="lg"
              onClick={onGetStarted}
              className="group"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button 
              variant="outline" 
              size="lg"
              onClick={() => window.open('https://github.com/Thanatos9404/BharatWitness#readme', '_blank')}
            >
              View Documentation
            </Button>
          </div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto"
          >
            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 backdrop-blur-sm">
              <Scale className="w-8 h-8 text-primary mb-3" />
              <div className="text-3xl font-bold text-foreground mb-1">
                50,000+
              </div>
              <div className="text-sm text-muted-foreground">
                Government Documents
              </div>
            </div>

            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 backdrop-blur-sm">
              <FileText className="w-8 h-8 text-primary mb-3" />
              <div className="text-3xl font-bold text-foreground mb-1">
                95%+
              </div>
              <div className="text-sm text-muted-foreground">
                Faithfulness Score
              </div>
            </div>

            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 backdrop-blur-sm">
              <Shield className="w-8 h-8 text-primary mb-3" />
              <div className="text-3xl font-bold text-foreground mb-1">
                98%+
              </div>
              <div className="text-sm text-muted-foreground">
                Temporal Accuracy
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};
