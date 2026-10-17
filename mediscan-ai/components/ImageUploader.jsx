"use client";

import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, AlertCircle, Brain, Stethoscope, Eye, Sparkles, FileImage } from "lucide-react";

export default function ImageUploader({ onImageSelected, disabled, selectedModality = "auto" }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);

  const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];

  const modalityHints = {
    brain: {
      title: "Upload Cranial MRI Examination",
      subtitle: "Axial, Coronal, or Sagittal T1 / T1-Contrast / T2 / FLAIR brain scan",
      badge: "TumorNet Neuro Mode",
      badgeColor: "text-indigo-300 bg-indigo-950/70 border-indigo-700/60",
      Icon: Brain,
    },
    chest: {
      title: "Upload Chest Radiograph (X-Ray)",
      subtitle: "Posteroanterior (PA) or Anteroposterior (AP) thoracic projection",
      badge: "PneumoNet Thoracic Mode",
      badgeColor: "text-sky-300 bg-sky-950/70 border-sky-700/60",
      Icon: Stethoscope,
    },
    gastro: {
      title: "Upload Endoscopic Mucosal Frame",
      subtitle: "Upper GI Gastroscopy or Lower GI Colonoscopy clinical frame",
      badge: "GastroNet Endoscopy Mode",
      badgeColor: "text-amber-300 bg-amber-950/70 border-amber-700/60",
      Icon: Eye,
    },
    auto: {
      title: "Upload Medical Examination Scan",
      subtitle: "Brain MRI, Chest Radiograph (PA/AP), or GI Endoscopic Mucosal scan",
      badge: "Auto-Routing Tri-Modality Engine",
      badgeColor: "text-blue-300 bg-blue-950/70 border-blue-700/60",
      Icon: Sparkles,
    },
  }[selectedModality] || {
    title: "Upload Medical Scan",
    subtitle: "Standard DICOM, JPEG, PNG, or WebP clinical image",
    badge: "Clinical Ingestion",
    badgeColor: "text-blue-300 bg-blue-950/70 border-blue-700/60",
    Icon: FileImage,
  };

  const HintIcon = modalityHints.Icon;

  const handleFile = (file) => {
    setErrorMessage(null);
    if (!file) return;

    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith(".dcm")) {
      setErrorMessage("Please upload a standard medical image format (JPEG, PNG, WebP, or DICOM).");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setErrorMessage("File exceeds 20MB. Please use an optimized image file.");
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  return (
    <div className="w-full">
      <motion.div
        id="image-drop-zone"
        tabIndex={0}
        role="button"
        whileHover={!disabled ? { scale: 1.008 } : {}}
        whileTap={!disabled ? { scale: 0.992 } : {}}
        aria-label="Upload medical scan"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            fileInputRef.current?.click();
          }
        }}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`w-full rounded-2xl border border-dashed p-8 sm:p-10 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[320px] sm:min-h-[360px] relative overflow-hidden focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
          isDragOver
            ? "border-[#3b82f6] bg-[#121c2b] shadow-2xl shadow-blue-500/20 scale-[1.01]"
            : "border-[#26374d] bg-[#0c1219] hover:border-[#3b516f] hover:bg-[#0f1722]"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          id="medical-file-input"
          accept="image/jpeg,image/png,image/webp,.dcm"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        {/* Modality Status Pill */}
        <div className="mb-4">
          <span
            className={`inline-flex items-center gap-1.5 text-[11px] px-3 py-1 rounded-full border font-mono font-medium ${modalityHints.badgeColor}`}
          >
            <HintIcon className="w-3.5 h-3.5" />
            <span>{modalityHints.badge}</span>
          </span>
        </div>

        {/* Icon Circle */}
        <div className="w-14 h-14 rounded-2xl bg-[#141f2d] border border-[#24354c] flex items-center justify-center text-[#60a5fa] mb-4 shadow-md">
          <Upload className="w-6 h-6" />
        </div>

        <h3 className="text-base sm:text-lg font-bold text-[#e6edf5] max-w-sm tracking-tight">
          {modalityHints.title}
        </h3>
        <p className="text-xs text-[#718294] mt-2 max-w-xs leading-relaxed font-medium">
          {modalityHints.subtitle}
        </p>

        <div className="mt-4 flex items-center gap-2 text-[11px] text-[#536173] font-mono">
          <span className="px-2 py-0.5 rounded-md bg-[#101722] border border-[#1b2737]">DICOM</span>
          <span className="px-2 py-0.5 rounded-md bg-[#101722] border border-[#1b2737]">JPEG</span>
          <span className="px-2 py-0.5 rounded-md bg-[#101722] border border-[#1b2737]">PNG</span>
          <span className="px-2 py-0.5 rounded-md bg-[#101722] border border-[#1b2737]">Max 20MB</span>
        </div>

        {errorMessage && (
          <div className="mt-4 flex items-center gap-2 text-xs text-[#fda4af] bg-[#290d14] border border-[#521825] px-3.5 py-2 rounded-xl shadow">
            <AlertCircle className="w-4 h-4 shrink-0 text-[#f43f5e]" />
            <span>{errorMessage}</span>
          </div>
        )}
      </motion.div>
    </div>
  );
}
