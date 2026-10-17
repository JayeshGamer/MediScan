"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Stethoscope, Eye, Sparkles, Activity, ShieldCheck, ChevronRight } from "lucide-react";

export default function HeroSection({
  onSelectSample,
  activePreset,
  selectedModality,
  onSelectModality,
  isAnalyzing,
}) {
  const MODALITIES = [
    {
      id: "auto",
      name: "Auto-Detect AI",
      short: "Auto",
      IconComponent: Sparkles,
      iconColor: "text-blue-400",
      activeBg: "bg-blue-600/15 border-blue-500 text-blue-300",
      badge: "Tri-Modality Engine",
      description: "Intelligently identifies scan type between Brain MRI, Chest X-Ray, and Endoscopy using visual feature inspection.",
    },
    {
      id: "brain",
      name: "Brain MRI (TumorNet)",
      short: "Brain MRI",
      IconComponent: Brain,
      iconColor: "text-indigo-400",
      activeBg: "bg-indigo-600/15 border-indigo-500 text-indigo-300",
      badge: "94.94% Acc | 0.987 AUC",
      description: "4-Class axial neuro-oncology classifier for Glioma (intra-axial), Meningioma (extra-axial), Pituitary Adenoma, and Healthy Cranium.",
    },
    {
      id: "chest",
      name: "Chest X-Ray (PneumoNet)",
      short: "Chest Radiograph",
      IconComponent: Stethoscope,
      iconColor: "text-sky-400",
      activeBg: "bg-sky-600/15 border-sky-500 text-sky-300",
      badge: "90.06% Acc | 0.961 AUC",
      description: "Binary pulmonary consolidation detector calibrated at 0.780 decision threshold to guarantee >=90% sensitivity and specificity.",
    },
    {
      id: "gastro",
      name: "Endoscopy (GastroNet)",
      short: "GI Endoscopy",
      IconComponent: Eye,
      iconColor: "text-amber-400",
      activeBg: "bg-amber-600/15 border-amber-500 text-amber-300",
      badge: "96.06% Acc | 0.991 AUC",
      description: "Trained on 4,736 mucosal images. Detects mucosal ulcers vs healthy digestive lining with 94.5% ulcer recall and 96.8% normal recall.",
    },
  ];

  const currentModality = MODALITIES.find((m) => m.id === selectedModality) || MODALITIES[0];
  const CurrentIcon = currentModality.IconComponent;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="pt-4 sm:pt-6 pb-6 border-b border-[#1c2633] space-y-6"
    >
      {/* Header Headline */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium bg-[#101e33] text-[#60a5fa] border border-[#1e3a6a] shadow-sm">
              <span className="w-2 h-2 rounded-full bg-[#3b82f6] animate-pulse"></span>
              MediScan AI Clinical Intelligence v2.0
            </span>
            <span className="hidden sm:inline-block text-[11px] text-[#536173]">
              DualPathResNet + CBAM Attention
            </span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-[#e6edf5]">
            Multi-Modality Radiology & Endoscopy Workstation
          </h1>
          <p className="mt-2 text-xs sm:text-sm text-[#8b99a9] max-w-2xl leading-relaxed">
            Clinical decision support platform powered by three deep residual neural networks. Evaluates Brain MRI, Chest Radiographs, and GI Endoscopy with live Grad-CAM heatmaps.
          </p>
        </div>

        {/* Modality Metrics Quick Badge */}
        <motion.div
          key={currentModality.id}
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.25 }}
          className="flex flex-row md:flex-col items-center md:items-end justify-between gap-2 p-3.5 rounded-xl bg-[#0e141d] border border-[#1c2633] shrink-0 shadow-md"
        >
          <span className="text-[10px] text-[#536173] uppercase tracking-wider font-semibold">
            Active Neural Network
          </span>
          <span className="text-sm font-bold text-[#e6edf5] flex items-center gap-1.5">
            <CurrentIcon className={`w-4 h-4 ${currentModality.iconColor}`} />
            <span>{currentModality.name}</span>
          </span>
          <span className="text-[11px] font-mono text-[#10b981] bg-[#0c2419] border border-[#16432f] px-2 py-0.5 rounded-md">
            {currentModality.badge}
          </span>
        </motion.div>
      </div>

      {/* Model Mode Navigation Tabs with Layout Animation */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-[#8b99a9]">
          <span className="font-semibold text-[#ccd6e2]">Select Diagnostic Model:</span>
          <span className="text-[11px] text-[#536173]">All models exceed 90%+ holdout accuracy</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {MODALITIES.map((mod) => {
            const isActive = selectedModality === mod.id;
            const Icon = mod.IconComponent;
            return (
              <motion.button
                key={mod.id}
                type="button"
                whileHover={{ scale: 1.015 }}
                whileTap={{ scale: 0.985 }}
                onClick={() => onSelectModality(mod.id)}
                disabled={isAnalyzing}
                aria-label={`Select ${mod.name}`}
                className={`flex flex-col p-3.5 rounded-xl border text-left transition-all cursor-pointer relative overflow-hidden focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
                  isActive
                    ? `${mod.activeBg} shadow-md`
                    : "bg-[#0e141d] border-[#1c2633] hover:border-[#2b394d] hover:bg-[#121a24]"
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1.5">
                  <Icon className={`w-4 h-4 ${mod.iconColor}`} />
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-md ${
                      isActive
                        ? "bg-[#2563eb] text-white font-medium"
                        : "bg-[#17202b] text-[#8b99a9]"
                    }`}
                  >
                    {mod.short}
                  </span>
                </div>
                <div className="text-xs font-bold text-[#e6edf5] truncate">
                  {mod.name.split(" ")[0]} {mod.name.split(" ")[1] || ""}
                </div>
                <div className="text-[10px] text-[#8b99a9] mt-0.5 truncate">
                  {mod.badge}
                </div>

                {isActive && (
                  <motion.div
                    layoutId="activeModalityIndicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 to-indigo-500"
                    transition={{ type: "spring", stiffness: 400, damping: 35 }}
                  />
                )}
              </motion.button>
            );
          })}
        </div>

        {/* Model Description Bar */}
        <p className="text-xs text-[#718294] leading-relaxed pt-0.5">
          {currentModality.description}
        </p>
      </div>

      {/* Model-Specific Reference Scans Bar */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <span className="text-xs font-semibold text-[#8b99a9] mr-1 flex items-center gap-1">
          <Activity className="w-3.5 h-3.5 text-[#60a5fa]" />
          <span>Clinical Test Scans:</span>
        </span>

        {selectedModality === "brain" && (
          <>
            <SampleButton
              presetKey="glioma"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Brain MRI (Glioma)"
              sub="Intra-Axial"
            />
            <SampleButton
              presetKey="meningioma"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Brain MRI (Meningioma)"
              sub="Extra-Axial"
            />
            <SampleButton
              presetKey="pituitary"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Brain MRI (Pituitary)"
              sub="Sellar Mass"
            />
            <SampleButton
              presetKey="notumor"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Brain MRI (No Tumor)"
              sub="Unremarkable"
            />
          </>
        )}

        {selectedModality === "chest" && (
          <>
            <SampleButton
              presetKey="pneumonia"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Chest X-Ray (Pneumonia)"
              sub="Consolidation"
            />
            <SampleButton
              presetKey="normal"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Chest X-Ray (Clear)"
              sub="Normal Lung"
            />
          </>
        )}

        {selectedModality === "gastro" && (
          <>
            <SampleButton
              presetKey="ulcer"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Endoscopy (Mucosal Ulcer)"
              sub="Erythema / Crater"
            />
            <SampleButton
              presetKey="gastro-normal"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Endoscopy (Normal Mucosa)"
              sub="Intact Cecum"
            />
          </>
        )}

        {selectedModality === "auto" && (
          <>
            <SampleButton
              presetKey="glioma"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Brain MRI"
              sub="Glioma"
            />
            <SampleButton
              presetKey="pneumonia"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Chest X-Ray"
              sub="Pneumonia"
            />
            <SampleButton
              presetKey="ulcer"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Endoscopy"
              sub="Ulceration"
            />
            <SampleButton
              presetKey="normal"
              activePreset={activePreset}
              isAnalyzing={isAnalyzing}
              onClick={onSelectSample}
              label="Chest X-Ray"
              sub="Clear Lungs"
            />
          </>
        )}
      </div>
    </motion.div>
  );
}

function SampleButton({ presetKey, activePreset, isAnalyzing, onClick, label, sub }) {
  const isActive = activePreset === presetKey;
  return (
    <motion.button
      type="button"
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      disabled={isAnalyzing}
      onClick={() => onClick(presetKey)}
      aria-label={`Load reference scan ${label}`}
      className={`text-xs px-3 py-1.5 rounded-lg border transition-all cursor-pointer flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
        isActive
          ? "bg-[#1c2e4a] text-[#93c5fd] border-[#3b82f6] font-medium shadow-sm"
          : "bg-[#0e141d] text-[#8b99a9] border-[#1c2633] hover:text-[#e6edf5] hover:border-[#2b394d]"
      }`}
    >
      <span>{label}</span>
      {sub && (
        <span className="text-[10px] text-[#536173] font-mono">
          [{sub}]
        </span>
      )}
    </motion.button>
  );
}
