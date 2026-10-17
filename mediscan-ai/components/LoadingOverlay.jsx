"use client";

import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Cpu, Sparkles, Activity } from "lucide-react";

export default function LoadingOverlay({ selectedModality = "auto" }) {
  const STAGES_MAP = {
    brain: [
      "Ingesting axial cranial slice...",
      "Extracting 3x3 fine & 5x5 broad convolutional features...",
      "Evaluating TumorNet 4-class neuro-oncology classifier...",
      "Computing CBAM spatial & channel attention heatmap...",
      "Synthesizing neuroradiology report & WHO CNS grading..."
    ],
    chest: [
      "Ingesting thoracic radiograph...",
      "Analyzing lobar airspace opacification & air bronchograms...",
      "Comparing probability against calibrated 0.780 threshold...",
      "Computing PneumoNet Grad-CAM attention overlay...",
      "Compiling pulmonology report & CURB-65 criteria..."
    ],
    gastro: [
      "Ingesting endoscopic mucosal frame...",
      "Inspecting epithelial disruption & erythema margins...",
      "Comparing probability against calibrated 0.580 threshold...",
      "Computing GastroNet Grad-CAM attention heatmap...",
      "Compiling gastroenterology report & Paris classification..."
    ],
    auto: [
      "Ingesting medical scan pixels...",
      "Auto-detecting anatomical modality (Brain / Chest / Endoscopy)...",
      "Routing to optimal DualPathResNet-18 neural model...",
      "Computing live Grad-CAM visual attention heatmap...",
      "Formatting clinical decision support findings..."
    ]
  };

  const stages = STAGES_MAP[selectedModality] || STAGES_MAP.auto;
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStageIndex((prev) => (prev < stages.length - 1 ? prev + 1 : prev));
    }, 280);
    return () => clearInterval(timer);
  }, [stages.length]);

  const progressPercent = Math.min(100, Math.round(((stageIndex + 1) / stages.length) * 100));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      className="w-full rounded-2xl border border-[#1e2e42] bg-[#0b121c] p-6 sm:p-8 my-4 shadow-2xl relative overflow-hidden backdrop-blur-md"
    >
      {/* Background ambient pulse */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-indigo-500/5 to-sky-500/5 animate-pulse pointer-events-none" />

      <div className="max-w-md mx-auto space-y-4 relative z-10">
        <div className="flex items-center justify-between text-xs text-[#8b99a9]">
          <div className="flex items-center gap-2.5 min-w-0">
            <Loader2 className="w-4 h-4 animate-spin text-[#60a5fa] shrink-0" />
            <AnimatePresence mode="wait">
              <motion.span
                key={stageIndex}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.2 }}
                className="font-medium text-[#e6edf5] truncate block"
              >
                {stages[stageIndex]}
              </motion.span>
            </AnimatePresence>
          </div>
          <span className="font-mono text-[#60a5fa] font-bold text-xs shrink-0 pl-2">
            {progressPercent}%
          </span>
        </div>

        {/* Progress Bar with Motion Spring */}
        <div className="w-full bg-[#05080c] rounded-full h-2.5 overflow-hidden border border-[#162537] p-0.5">
          <motion.div
            initial={{ width: "5%" }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ type: "spring", stiffness: 200, damping: 25 }}
            className="bg-gradient-to-r from-blue-600 via-sky-500 to-indigo-500 h-full rounded-full shadow-[0_0_12px_rgba(56,189,248,0.5)]"
          />
        </div>

        <div className="flex items-center justify-between text-[11px] text-[#536173] font-mono">
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3 text-[#38bdf8]" />
            DualPathResNet Engine
          </span>
          <span className="flex items-center gap-1 text-[#10b981]">
            <Activity className="w-3 h-3" />
            GPU Accelerated
          </span>
        </div>
      </div>
    </motion.div>
  );
}
