"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Copy,
  Check,
  Download,
  Stethoscope,
  Layers,
  FileCheck2,
  ChevronRight,
  ShieldCheck,
  AlertCircle
} from "lucide-react";

export default function DiagnosisDetail({ result }) {
  const [activeTab, setActiveTab] = useState("findings");
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  if (!result) return null;

  const {
    description,
    findings = [],
    recommendations = [],
    prediction,
    confidence,
    analysisId,
    modality,
    clinicalMetrics = {},
    decisionThreshold,
    severity,
    scanType,
    timestamp,
  } = result;

  const confidencePercent = Math.round(confidence * 100);

  // Generate full clinical transcript for copying
  const handleCopySummary = () => {
    const text = `=====================================================
MEDISCAN AI CLINICAL RADIOLOGY / ENDOSCOPY REPORT
=====================================================
Scan ID:          ${analysisId || "N/A"}
Date & Time:      ${timestamp || new Date().toISOString()}
Examination:      ${scanType || "Medical Scan"}
Neural Engine:    DualPathResNet-18 + CBAM Attention (GPU Accelerated)
Primary Finding:  ${prediction} (${confidencePercent}% Posterior Confidence)
Clinical Urgency: ${severity} Priority
Calibrated Cutoff:${decisionThreshold ? `${decisionThreshold.toFixed(3)}` : "N/A"}

-----------------------------------------------------
1. RADIOGRAPHIC / ENDOSCOPIC INTERPRETATION:
-----------------------------------------------------
${description}

-----------------------------------------------------
2. OBSERVED ANATOMICAL FEATURES:
-----------------------------------------------------
${findings.map((f, i) => `[${i + 1}] ${f}`).join("\n")}

-----------------------------------------------------
3. CLINICAL METRICS & DIFFERENTIAL DIAGNOSIS:
-----------------------------------------------------
${clinicalMetrics.compartment ? `Compartment:       ${clinicalMetrics.compartment}\n` : ""}${clinicalMetrics.whoGrade ? `WHO Classification:${clinicalMetrics.whoGrade}\n` : ""}${clinicalMetrics.edemaIndex ? `Edema / Mass Effect:${clinicalMetrics.edemaIndex}\n` : ""}${clinicalMetrics.pattern ? `Opacification Pattern:${clinicalMetrics.pattern}\n` : ""}${clinicalMetrics.curb65Risk ? `CURB-65 Risk Tier: ${clinicalMetrics.curb65Risk}\n` : ""}${clinicalMetrics.parisClassification ? `Paris Classification:${clinicalMetrics.parisClassification}\n` : ""}${clinicalMetrics.forrestStigmata ? `Forrest Stigmata:  ${clinicalMetrics.forrestStigmata}\n` : ""}${clinicalMetrics.differential ? `Differential:      ${clinicalMetrics.differential}\n` : ""}
-----------------------------------------------------
4. ACTIONABLE CLINICAL PROTOCOLS:
-----------------------------------------------------
${recommendations.map((r, i) => `• ${r}`).join("\n")}
${clinicalMetrics.antibioticRegimen ? `\nEmpiric Antimicrobial Regimen:\n• ${clinicalMetrics.antibioticRegimen}\n` : ""}${clinicalMetrics.pharmacotherapy ? `\nRecommended Pharmacotherapy:\n• ${clinicalMetrics.pharmacotherapy}\n` : ""}
=====================================================
DISCLAIMER: AI-Assisted Decision Support. Must be reviewed by board-certified clinician.
=====================================================`;

    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  };

  const handleDownloadJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `mediscan_${analysisId || "report"}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2200);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.12, ease: "easeOut" }}
      className="w-full rounded-2xl border border-[#1c2633] bg-[#0e141d] overflow-hidden shadow-xl backdrop-blur-md"
    >
      {/* Workstation Top Action Bar */}
      <div className="px-4 sm:px-6 py-3.5 border-b border-[#1c2633] bg-[#0a0f16] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 text-xs">
          <FileText className="w-4 h-4 text-[#60a5fa]" />
          <h3 className="font-bold text-[#e6edf5] text-xs sm:text-sm tracking-tight">
            Clinical Diagnostic Dossier
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <motion.button
            type="button"
            id="copy-report-summary-btn"
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={handleCopySummary}
            aria-label="Copy clinical summary to clipboard"
            className="text-xs text-[#8b99a9] hover:text-[#e6edf5] bg-[#141b24] hover:bg-[#1e2938] border border-[#232f3e] px-3.5 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8]"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-[#10b981]" />
                <span className="text-[#10b981] font-semibold">Copied to Clipboard</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Report</span>
              </>
            )}
          </motion.button>

          <motion.button
            type="button"
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={handleDownloadJSON}
            aria-label="Download JSON diagnostic payload"
            className="text-xs text-[#8b99a9] hover:text-[#e6edf5] bg-[#141b24] hover:bg-[#1e2938] border border-[#232f3e] px-3.5 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8]"
            title="Download JSON Clinical Data"
          >
            {downloaded ? (
              <Check className="w-3.5 h-3.5 text-[#10b981]" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            <span className="hidden sm:inline">JSON</span>
          </motion.button>
        </div>
      </div>

      {/* Responsive Navigation Tabs with Layout Animation */}
      <div className="px-4 sm:px-6 pt-3 bg-[#0c121a] border-b border-[#1c2633] flex items-center gap-3 overflow-x-auto text-xs">
        <button
          type="button"
          onClick={() => setActiveTab("findings")}
          className={`pb-3 px-2 border-b-2 font-bold transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
            activeTab === "findings"
              ? "border-[#3b82f6] text-[#60a5fa]"
              : "border-transparent text-[#8b99a9] hover:text-[#e6edf5]"
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Radiological Findings</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("staging")}
          className={`pb-3 px-2 border-b-2 font-bold transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
            activeTab === "staging"
              ? "border-[#3b82f6] text-[#60a5fa]"
              : "border-transparent text-[#8b99a9] hover:text-[#e6edf5]"
          }`}
        >
          <Stethoscope className="w-4 h-4" />
          <span>Staging & Differential</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("protocols")}
          className={`pb-3 px-2 border-b-2 font-bold transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
            activeTab === "protocols"
              ? "border-[#3b82f6] text-[#60a5fa]"
              : "border-transparent text-[#8b99a9] hover:text-[#e6edf5]"
          }`}
        >
          <FileCheck2 className="w-4 h-4" />
          <span>Clinical Protocol</span>
        </button>
      </div>

      {/* Animated Tab Content with AnimatePresence */}
      <AnimatePresence mode="wait">
        {activeTab === "findings" && (
          <motion.div
            key="findings"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.2 }}
            className="p-5 sm:p-6 space-y-5"
          >
            {/* Narrative Summary */}
            <div>
              <h4 className="text-xs font-bold text-[#8b99a9] uppercase tracking-wider mb-2">
                Radiological Impression
              </h4>
              <p className="text-xs sm:text-sm text-[#ccd6e2] leading-relaxed bg-[#0a0f16] p-4 rounded-xl border border-[#1a2533] shadow-inner font-medium">
                {description}
              </p>
            </div>

            {/* Observed Specific Anatomical Features */}
            {findings.length > 0 && (
              <div>
                <h4 className="text-xs font-bold text-[#8b99a9] uppercase tracking-wider mb-2.5">
                  Key Observed Radiographic Features
                </h4>
                <div className="space-y-2">
                  {findings.map((finding, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="p-3.5 rounded-xl bg-[#0a0f16] border border-[#1a2533] text-xs text-[#ccd6e2] flex items-start gap-3 shadow-sm"
                    >
                      <span className="w-5 h-5 rounded-full bg-[#142337] border border-[#213854] text-[#60a5fa] font-mono font-bold flex items-center justify-center shrink-0 text-[11px] mt-0.5">
                        {idx + 1}
                      </span>
                      <span className="leading-relaxed font-medium">{finding}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {activeTab === "staging" && (
          <motion.div
            key="staging"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.2 }}
            className="p-5 sm:p-6 space-y-4"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Brain-specific metrics */}
              {modality === "brain" && (
                <>
                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Anatomical Compartment
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.compartment || "Cranial Parenchyma"}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      WHO CNS Histological Grade Tier
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.whoGrade || "Clinical Review Required"}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Perilesional Edema & Mass Effect
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.edemaIndex || "Evaluated by Dual-Path Broad Kernels"}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Surgical Intervention Feasibility
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.resectability || "Multidisciplinary Tumor Board"}
                    </span>
                  </div>
                </>
              )}

              {/* Chest-specific metrics */}
              {modality === "chest" && (
                <>
                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Parenchymal Opacification Pattern
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.pattern || "Alveolar Infiltrate"}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      CURB-65 Pneumonia Severity Score
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.curb65Risk || "Stratification based on vitals"}
                    </span>
                  </div>
                </>
              )}

              {/* Gastro-specific metrics */}
              {modality === "gastro" && (
                <>
                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Paris Endoscopic Classification
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.parisClassification || "Paris Type 0-III"}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-[#0a0f16] border border-[#1a2533] shadow-sm">
                    <span className="text-[10px] text-[#718294] uppercase tracking-wider block font-bold">
                      Forrest Bleeding Stigmata Risk
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-[#e6edf5] mt-1 block">
                      {clinicalMetrics.forrestStigmata || "Forrest IIb/IIc"}
                    </span>
                  </div>
                </>
              )}
            </div>

            {/* Differential Diagnosis Card */}
            {clinicalMetrics.differential && (
              <div className="p-4 rounded-xl bg-[#111924] border border-[#1f2d3d] mt-3 shadow-md">
                <span className="text-[10px] font-bold text-[#8b99a9] uppercase tracking-wider block mb-1">
                  Differential Considerations
                </span>
                <p className="text-xs sm:text-sm text-[#cbd5e1] font-semibold leading-relaxed">
                  {clinicalMetrics.differential}
                </p>
              </div>
            )}
          </motion.div>
        )}

        {activeTab === "protocols" && (
          <motion.div
            key="protocols"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.2 }}
            className="p-5 sm:p-6 space-y-4"
          >
            {/* Pharmacotherapy guidance */}
            {clinicalMetrics.antibioticRegimen && (
              <div className="p-4 rounded-xl bg-[#102035] border border-[#1e3c63] text-xs text-[#bae6fd] shadow-md">
                <span className="font-bold block text-sm mb-1 text-[#e0f2fe] flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-[#38bdf8]" />
                  Targeted Antimicrobial Regimen
                </span>
                <p className="leading-relaxed opacity-95 font-medium">
                  {clinicalMetrics.antibioticRegimen}
                </p>
              </div>
            )}

            {clinicalMetrics.pharmacotherapy && (
              <div className="p-4 rounded-xl bg-[#261f0d] border border-[#524119] text-xs text-[#fef08a] shadow-md">
                <span className="font-bold block text-sm mb-1 text-[#fef9c3] flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-[#facc15]" />
                  Recommended Mucosal Pharmacotherapy
                </span>
                <p className="leading-relaxed opacity-95 font-medium">
                  {clinicalMetrics.pharmacotherapy}
                </p>
              </div>
            )}

            {/* Recommendations List */}
            {recommendations.length > 0 && (
              <div>
                <h4 className="text-xs font-bold text-[#8b99a9] uppercase tracking-wider mb-2.5">
                  Standard Care Pathways & Follow-Up
                </h4>
                <div className="space-y-2">
                  {recommendations.map((rec, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded-xl bg-[#0a0f16] border border-[#1a2533] text-xs text-[#ccd6e2] flex items-start gap-3 shadow-sm font-medium"
                    >
                      <ChevronRight className="w-4 h-4 text-[#3b82f6] shrink-0 mt-0.5" />
                      <span className="leading-relaxed">{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
