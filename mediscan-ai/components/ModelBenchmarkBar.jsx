"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu,
  ShieldCheck,
  Activity,
  Target,
  Brain,
  Stethoscope,
  Eye,
  Sparkles,
  CheckCircle2,
  SlidersHorizontal
} from "lucide-react";

export const MODEL_SPECS = {
  auto: {
    id: "auto",
    title: "Tri-Modality Intelligent Router",
    architecture: "DualPathResNet (3x3 + 5x5) + CBAM Spatial/Channel Attention",
    modalityName: "Auto-Routing Multi-Model Engine",
    datasetInfo: "Cross-modality holdout validation across 2,935 independent scans",
    primaryMetric: "3 Dedicated Models",
    metricValue: ">= 90% Acc",
    aucMetric: "0.98+ Avg AUC",
    calibratedThreshold: "Dynamic Modality Calibrated",
    themeColor: "from-blue-500/10 via-indigo-500/10 to-sky-500/10 border-blue-500/40 glow-auto",
    badgeColor: "text-blue-400 bg-blue-950/50 border-blue-800",
    modelsIncluded: [
      { name: "TumorNet", target: "Brain MRI", acc: "94.94%", auc: "0.987" },
      { name: "PneumoNet", target: "Chest X-Ray", acc: "90.06%", thresh: "0.780" },
      { name: "GastroNet", target: "GI Endoscopy", acc: "96.06%", thresh: "0.580" },
    ],
    features: [
      "Dynamic anatomical feature extraction for modality classification",
      "Live Grad-CAM attention visualizer across all target organs",
      "Calibrated decision boundary protection against false negatives",
    ],
  },
  brain: {
    id: "brain",
    title: "TumorNet — Neuro-Oncology Radiomics Engine",
    architecture: "DualPathResNet-18 Backbone + CBAM Attention + Dual Pooling",
    modalityName: "Cranial Magnetic Resonance Imaging (Axial T1/T2/FLAIR)",
    datasetInfo: "Evaluated on 1,600 independent holdout brain MRI scans",
    primaryMetric: "Overall Accuracy",
    metricValue: "94.94%",
    aucMetric: "0.9874 AUC-ROC",
    calibratedThreshold: "Softmax Multi-Class argmax with logit calibration",
    themeColor: "from-indigo-500/15 via-purple-500/10 to-violet-500/15 border-indigo-500/40 glow-brain",
    badgeColor: "text-indigo-400 bg-indigo-950/50 border-indigo-800",
    classesBreakdown: [
      { label: "Glioma", desc: "Intra-axial parenchymal infiltrative mass", acc: "94.63%" },
      { label: "Meningioma", desc: "Extra-axial dural-based circumscribed mass", acc: "95.56%" },
      { label: "Pituitary Adenoma", desc: "Sellar / suprasellar hypophyseal mass", acc: "99.25%" },
      { label: "No Tumor", desc: "Unremarkable cranial neuroimaging baseline", acc: "98.12%" },
    ],
    features: [
      "Dual-path fine (3x3) and broad (5x5) receptive field convolution",
      "WHO CNS Grade III/IV differentiation support",
      "Sub-40ms GPU inference with INT8 dynamic quantization readiness",
    ],
  },
  chest: {
    id: "chest",
    title: "PneumoNet — Thoracic Radiography Engine",
    architecture: "DualPathResNet-18 + CBAM Attention + Calibrated Sigmoid",
    modalityName: "Chest Radiograph (PA & AP Projection Radiographs)",
    datasetInfo: "Evaluated on 624 holdout clinical pediatric/adult chest radiographs",
    primaryMetric: "Test Accuracy",
    metricValue: "90.06%",
    aucMetric: "0.9605 AUC-ROC",
    calibratedThreshold: "Calibrated 0.780 Decision Cutoff (90% Sens / 90.2% Spec)",
    themeColor: "from-sky-500/15 via-cyan-500/10 to-blue-500/15 border-sky-500/40 glow-chest",
    badgeColor: "text-sky-400 bg-sky-950/50 border-sky-800",
    classesBreakdown: [
      { label: "Pneumonia Infiltrate", desc: "Alveolar consolidation & air bronchograms", acc: "90.00% Recall" },
      { label: "Normal Lung Fields", desc: "Sharp costophrenic angles & clear parenchyma", acc: "90.17% Recall" },
    ],
    features: [
      "Calibrated threshold (0.780) eliminates high false-positive false alarms",
      "Airspace consolidation vs broncho-vascular prominence localization",
      "Integrated CURB-65 severity score & empiric antibiotic guidance",
    ],
  },
  gastro: {
    id: "gastro",
    title: "GastroNet — Endoscopic Mucosal Diagnostics Engine",
    architecture: "DualPathResNet-18 + CBAM Spatial/Channel Attention + Classifier",
    modalityName: "Digestive Endoscopic Imaging (Upper & Lower GI Frames)",
    datasetInfo: "Evaluated on 711 holdout clinical endoscopic mucosal images",
    primaryMetric: "Test Accuracy",
    metricValue: "96.06%",
    aucMetric: "0.9906 AUC-ROC",
    calibratedThreshold: "Calibrated 0.580 Decision Cutoff (94.5% Ulcer Recall / 96.8% Normal)",
    themeColor: "from-amber-500/15 via-emerald-500/10 to-teal-500/15 border-amber-500/40 glow-gastro",
    badgeColor: "text-amber-400 bg-amber-950/50 border-amber-800",
    classesBreakdown: [
      { label: "Mucosal Ulceration", desc: "Epithelial crater, erythema & active margins", acc: "94.49% Recall" },
      { label: "Normal Mucosa", desc: "Intact foveolar lining & healthy capillary network", acc: "96.84% Recall" },
    ],
    features: [
      "Trained on 4,736 high-resolution endoscopic mucosal frames",
      "Paris & Forrest classification for mucosal lesion severity tiering",
      "Sub-25ms real-time latency supporting live endoscopy video streams",
    ],
  },
};

export default function ModelBenchmarkBar({ selectedModality, onSelectModality, isAnalyzing }) {
  const spec = MODEL_SPECS[selectedModality] || MODEL_SPECS.auto;

  const MODALITY_BUTTONS = [
    { id: "auto", label: "Auto", Icon: Sparkles },
    { id: "brain", label: "Brain", Icon: Brain },
    { id: "chest", label: "Chest", Icon: Stethoscope },
    { id: "gastro", label: "Gastro", Icon: Eye },
  ];

  return (
    <motion.div
      layout
      className={`w-full rounded-2xl border bg-gradient-to-br ${spec.themeColor} bg-[#0a0f16]/95 p-4 sm:p-5 transition-all shadow-xl backdrop-blur-md`}
    >
      {/* Top Header: Title + Modality Selector Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3.5 border-b border-[#1f2c3d]">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-[#141e2b] border border-[#26374d] flex items-center justify-center shrink-0 text-[#60a5fa] shadow-sm">
            <Cpu className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm sm:text-base font-bold text-[#e6edf5] truncate tracking-tight">
              {spec.title}
            </h2>
            <p className="text-[11px] text-[#8b99a9] truncate font-medium">
              {spec.modalityName}
            </p>
          </div>
        </div>

        {/* Quick Modality Switcher Pills */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto overflow-x-auto max-w-full pb-1 sm:pb-0">
          {MODALITY_BUTTONS.map((item) => {
            const isActive = selectedModality === item.id;
            const Icon = item.Icon;
            return (
              <motion.button
                key={item.id}
                type="button"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                disabled={isAnalyzing}
                onClick={() => onSelectModality(item.id)}
                aria-label={`Switch to ${item.label} mode`}
                className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-all cursor-pointer shrink-0 flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
                  isActive
                    ? "bg-[#2563eb] text-white border-[#3b82f6] shadow-md shadow-blue-500/20"
                    : "bg-[#0f1722] text-[#8b99a9] border-[#1c2838] hover:text-[#e6edf5] hover:border-[#2f425a]"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Grid: 3 Responsive Benchmark Stat Cards with Animated Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={spec.id}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className="mt-3.5 grid grid-cols-1 sm:grid-cols-3 gap-3"
        >
          {/* Metric 1: Accuracy */}
          <div className="p-3.5 rounded-xl bg-[#0a111a] border border-[#1c2838] flex items-center gap-3 shadow-sm hover:border-[#2c3d52] transition-colors">
            <div className="w-10 h-10 rounded-lg bg-[#132030] border border-[#1e344e] flex items-center justify-center shrink-0 text-[#10b981]">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-semibold">
                {spec.primaryMetric}
              </span>
              <span className="text-base sm:text-lg font-bold text-[#e6edf5] font-mono block tracking-tight">
                {spec.metricValue}
              </span>
              <span className="text-[10px] text-[#10b981] font-mono flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                Validated on raw data
              </span>
            </div>
          </div>

          {/* Metric 2: AUC-ROC */}
          <div className="p-3.5 rounded-xl bg-[#0a111a] border border-[#1c2838] flex items-center gap-3 shadow-sm hover:border-[#2c3d52] transition-colors">
            <div className="w-10 h-10 rounded-lg bg-[#132030] border border-[#1e344e] flex items-center justify-center shrink-0 text-[#60a5fa]">
              <Activity className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-semibold">
                Discriminative Power
              </span>
              <span className="text-base sm:text-lg font-bold text-[#e6edf5] font-mono block tracking-tight">
                {spec.aucMetric}
              </span>
              <span className="text-[10px] text-[#60a5fa] font-mono">
                High class separation
              </span>
            </div>
          </div>

          {/* Metric 3: Decision Boundary / Threshold */}
          <div className="p-3.5 rounded-xl bg-[#0a111a] border border-[#1c2838] flex items-center gap-3 shadow-sm hover:border-[#2c3d52] transition-colors">
            <div className="w-10 h-10 rounded-lg bg-[#132030] border border-[#1e344e] flex items-center justify-center shrink-0 text-[#f59e0b]">
              <Target className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-semibold">
                Decision Boundary
              </span>
              <span className="text-xs sm:text-sm font-bold text-[#e6edf5] font-mono truncate block">
                {spec.calibratedThreshold}
              </span>
              <span className="text-[10px] text-[#8b99a9]">
                Optimized Youden index
              </span>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Class Sensitivity List (For Brain, Chest, or Gastro) */}
      {spec.classesBreakdown && (
        <div className="mt-3 pt-3 border-t border-[#182332] grid grid-cols-2 sm:grid-cols-4 gap-2">
          {spec.classesBreakdown.map((item, idx) => (
            <div
              key={idx}
              className="px-3 py-2 rounded-lg bg-[#0b131e] border border-[#1c2838] text-[11px]"
            >
              <div className="flex items-center justify-between gap-1 mb-0.5">
                <span className="font-bold text-[#e6edf5] truncate">
                  {item.label}
                </span>
                <span className="font-mono text-[#10b981] font-bold shrink-0">
                  {item.acc}
                </span>
              </div>
              <p className="text-[10px] text-[#6d7e91] truncate">
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Auto-Mode Model Cards (When in Auto mode) */}
      {spec.modelsIncluded && (
        <div className="mt-3 pt-3 border-t border-[#182332] grid grid-cols-1 sm:grid-cols-3 gap-2">
          {spec.modelsIncluded.map((m, idx) => (
            <div
              key={idx}
              className="p-2.5 rounded-lg bg-[#0b131e] border border-[#1c2838] flex items-center justify-between text-xs"
            >
              <div>
                <span className="font-bold text-[#e6edf5]">{m.name}</span>
                <span className="text-[#8b99a9] text-[10px] block">{m.target}</span>
              </div>
              <div className="text-right">
                <span className="font-mono text-[#10b981] font-semibold">{m.acc}</span>
                <span className="text-[#64748b] text-[10px] block">
                  {m.auc ? `AUC: ${m.auc}` : `Thresh: ${m.thresh}`}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
