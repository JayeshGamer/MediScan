"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  AlertOctagon,
  Activity,
  Clock,
  ShieldAlert,
  Brain,
  Stethoscope,
  Eye,
  Sparkles,
  Layers,
  FileText
} from "lucide-react";

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    prediction,
    confidence,
    severity,
    scanType,
    analysisId,
    inferenceLatencyMs,
    urgencyText,
    modality,
    decisionThreshold,
    clinicalMetrics = {},
    isPathological,
  } = result;

  const confidencePercent = Math.round(confidence * 100);

  // Modality-specific color styles and identities with Lucide icons
  const modalityStyles = {
    brain: {
      accent: "border-indigo-500/40 bg-[#0d1222] glow-brain",
      badge: "text-indigo-300 bg-indigo-950/70 border-indigo-700/60",
      indicator: "#818cf8",
      modelTag: "TumorNet Neuro-Oncology Engine",
      Icon: Brain,
    },
    chest: {
      accent: "border-sky-500/40 bg-[#0c1624] glow-chest",
      badge: "text-sky-300 bg-sky-950/70 border-sky-700/60",
      indicator: "#38bdf8",
      modelTag: "PneumoNet Pulmonary Engine",
      Icon: Stethoscope,
    },
    gastro: {
      accent: "border-amber-500/40 bg-[#17150d] glow-gastro",
      badge: "text-amber-300 bg-amber-950/70 border-amber-700/60",
      indicator: "#fbbf24",
      modelTag: "GastroNet Endoscopy Engine",
      Icon: Eye,
    },
  }[modality] || {
    accent: "border-blue-500/40 bg-[#0c1420] glow-auto",
    badge: "text-blue-300 bg-blue-950/70 border-blue-700/60",
    indicator: "#60a5fa",
    modelTag: "MediScan AI Engine",
    Icon: Sparkles,
  };

  const ModalityIcon = modalityStyles.Icon;

  // Triage priority and clinical severity styles
  const severityStyles = {
    High: {
      icon: AlertOctagon,
      dot: "bg-rose-500",
      text: "text-rose-400",
      badge: "text-rose-300 bg-rose-950/70 border-rose-800/80",
      border: "border-rose-900/50",
      label: "Critical / High Priority",
    },
    Moderate: {
      icon: AlertTriangle,
      dot: "bg-amber-500",
      text: "text-amber-400",
      badge: "text-amber-300 bg-amber-950/70 border-amber-800/80",
      border: "border-amber-900/50",
      label: "Moderate Clinical Priority",
    },
    Low: {
      icon: CheckCircle2,
      dot: "bg-emerald-500",
      text: "text-emerald-400",
      badge: "text-emerald-300 bg-emerald-950/70 border-emerald-800/80",
      border: "border-emerald-900/50",
      label: "Routine / Normal Baseline",
    },
  }[severity] || {
    icon: Activity,
    dot: "bg-blue-500",
    text: "text-blue-400",
    badge: "text-blue-300 bg-blue-950/70 border-blue-800/80",
    border: "border-blue-900/50",
    label: "Standard Review",
  };

  const SeverityIcon = severityStyles.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={`w-full rounded-2xl border p-5 sm:p-6 transition-all shadow-xl backdrop-blur-md ${modalityStyles.accent} ${severityStyles.border}`}
    >
      {/* Top Metadata Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3.5 border-b border-[#212d3d] text-xs text-[#8b99a9]">
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <span className="flex items-center gap-1.5 font-bold text-[#e6edf5]">
            <ModalityIcon className="w-4 h-4 text-[#60a5fa]" />
            <span>{scanType || "Medical Scan"}</span>
          </span>
          <span className="text-[#3b4b5e] hidden sm:inline">•</span>
          <span className="font-mono text-[#718294] text-[11px] truncate">
            ID: {analysisId}
          </span>
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono text-[#8b99a9]">
          <span className="inline-flex items-center gap-1 text-[#60a5fa] bg-[#111f35] px-2 py-0.5 rounded-md border border-[#1b345b]">
            <Clock className="w-3 h-3" />
            {inferenceLatencyMs ? `${inferenceLatencyMs}ms` : "Live GPU"}
          </span>
          <span className="text-[#3b4b5e]">•</span>
          <span className="text-[#10b981] font-semibold">DualPath+CBAM</span>
        </div>
      </div>

      {/* Main Finding & Confidence Hero */}
      <div className="mt-4 flex flex-col sm:flex-row sm:items-start justify-between gap-4">
        <div className="space-y-2.5 max-w-xl min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[#718294]">
              Primary Clinical Finding
            </span>
            <span
              className={`text-[10px] px-2.5 py-0.5 rounded-full border font-mono font-medium ${modalityStyles.badge}`}
            >
              {modalityStyles.modelTag}
            </span>
          </div>

          <h2 className="text-xl sm:text-2xl lg:text-3xl font-extrabold text-[#e6edf5] tracking-tight leading-snug">
            {prediction}
          </h2>

          {/* Model-specific key parameters */}
          <div className="pt-1 flex flex-wrap gap-2 text-xs">
            {modality === "brain" && clinicalMetrics.whoGrade && (
              <span className="px-2.5 py-1 rounded-lg bg-[#101726] border border-[#23354f] text-[#cbd5e1] font-mono font-medium shadow-sm">
                {clinicalMetrics.whoGrade}
              </span>
            )}

            {modality === "brain" && clinicalMetrics.compartment && (
              <span className="px-2.5 py-1 rounded-lg bg-[#101726] border border-[#23354f] text-[#93c5fd] font-medium shadow-sm">
                Compartment: {clinicalMetrics.compartment}
              </span>
            )}

            {modality === "chest" && decisionThreshold && (
              <span className="px-2.5 py-1 rounded-lg bg-[#0b1b2d] border border-[#1d3c5f] text-[#7dd3fc] font-mono font-medium shadow-sm">
                Threshold: {decisionThreshold.toFixed(3)}
              </span>
            )}

            {modality === "chest" && clinicalMetrics.thresholdMargin && (
              <span className="px-2.5 py-1 rounded-lg bg-[#0b1b2d] border border-[#1d3c5f] text-[#38bdf8] font-mono font-medium shadow-sm">
                Margin: {clinicalMetrics.thresholdMargin}
              </span>
            )}

            {modality === "gastro" && decisionThreshold && (
              <span className="px-2.5 py-1 rounded-lg bg-[#211a0d] border border-[#48391b] text-[#fcd34d] font-mono font-medium shadow-sm">
                Threshold: {decisionThreshold.toFixed(3)}
              </span>
            )}

            {modality === "gastro" && clinicalMetrics.parisClassification && (
              <span className="px-2.5 py-1 rounded-lg bg-[#211a0d] border border-[#48391b] text-[#fde68a] font-medium shadow-sm">
                {clinicalMetrics.parisClassification}
              </span>
            )}
          </div>
        </div>

        {/* Confidence Percentage Metric */}
        <div className="sm:text-right shrink-0 p-3.5 rounded-xl bg-[#080d14]/90 border border-[#1c2738] shadow-inner">
          <span className="text-[10px] font-bold text-[#8b99a9] uppercase tracking-wider block mb-1">
            Posterior Confidence
          </span>
          <div className="flex sm:justify-end items-baseline gap-1">
            <span className="text-3xl sm:text-4xl font-black text-[#e6edf5] font-mono tracking-tight">
              {confidencePercent}%
            </span>
          </div>
          <div className="mt-2">
            <span
              className={`inline-flex items-center gap-1.5 text-xs px-3 py-1 rounded-lg border font-semibold shadow-sm ${severityStyles.badge}`}
            >
              <span className={`w-2 h-2 rounded-full ${severityStyles.dot} animate-pulse`}></span>
              <span>{severityStyles.label}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Clinical Urgency Action Bar */}
      {urgencyText && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className={`mt-4 p-3.5 rounded-xl border flex items-start gap-3 text-xs shadow-md ${
            isPathological
              ? "bg-[#210c11] border-[#4a1824] text-[#fda4af]"
              : "bg-[#091f16] border-[#164732] text-[#6ee7b7]"
          }`}
        >
          <SeverityIcon className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="min-w-0">
            <span className="font-bold block mb-0.5">
              {isPathological ? "Clinical Attention Required:" : "Diagnostic Status:"}
            </span>
            <p className="leading-relaxed opacity-95">
              {urgencyText}
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
