"use client";

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Microscope,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  AlertCircle
} from "lucide-react";
import HeroSection from "@/components/HeroSection";
import ModelBenchmarkBar from "@/components/ModelBenchmarkBar";
import ImageUploader from "@/components/ImageUploader";
import PreviewPanel from "@/components/PreviewPanel";
import AnalyzeButton from "@/components/AnalyzeButton";
import LoadingOverlay from "@/components/LoadingOverlay";
import ResultCard from "@/components/ResultCard";
import ConfidenceChart from "@/components/ConfidenceChart";
import DiagnosisDetail from "@/components/DiagnosisDetail";
import DisclaimerBanner from "@/components/DisclaimerBanner";
import { analyzeMedicalImage } from "@/lib/api";

export default function Home() {
  const [selectedModality, setSelectedModality] = useState("auto");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [activePreset, setActivePreset] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const resultsRef = useRef(null);

  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const handleSelectModality = (modalityId) => {
    setSelectedModality(modalityId);
    setError(null);
  };

  const handleImageSelected = (file) => {
    setError(null);
    setSelectedFile(file);
    setActivePreset(null);
    setResult(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    let typeHint = "Medical Radiograph";
    let detectedModality = selectedModality;
    const lower = file.name.toLowerCase();

    if (lower.includes("mri") || lower.includes("brain") || lower.includes("tumor") || lower.includes("glioma") || lower.includes("mening") || lower.includes("pituitary")) {
      typeHint = "Brain MRI Scan (Axial/Coronal)";
      detectedModality = "brain";
    } else if (lower.includes("xray") || lower.includes("chest") || lower.includes("pneumonia") || lower.includes("person") || lower.includes("normal") || lower.includes("lung")) {
      typeHint = "Chest Radiograph (PA/AP View)";
      detectedModality = "chest";
    } else if (lower.includes("ulcer") || lower.includes("gastro") || lower.includes("endoscop") || lower.includes("mucosa") || lower.includes("colitis")) {
      typeHint = "GI Endoscopic Frame";
      detectedModality = "gastro";
    }

    if (selectedModality === "auto" && detectedModality !== "auto") {
      setSelectedModality(detectedModality);
    }

    setFileInfo({
      name: file.name,
      sizeFormatted: formatBytes(file.size),
      typeHint: typeHint,
    });
  };

  const handleSelectSample = (presetKey) => {
    setError(null);
    setActivePreset(presetKey);
    setResult(null);

    let samplePath = "/samples/sample-pneumonia.jpeg";
    let name = "reference_chest_pneumonia.jpeg";
    let typeHint = "Chest Radiograph (AP View)";
    let estSize = 61118;
    let targetModality = "chest";

    switch (presetKey) {
      case "pneumonia":
        samplePath = "/samples/sample-pneumonia.jpeg";
        name = "reference_chest_pneumonia.jpeg";
        typeHint = "Chest Radiograph (AP View)";
        estSize = 61118;
        targetModality = "chest";
        break;

      case "normal":
        samplePath = "/samples/sample-normal.jpeg";
        name = "reference_chest_unremarkable.jpeg";
        typeHint = "Chest Radiograph (PA View)";
        estSize = 253130;
        targetModality = "chest";
        break;

      case "glioma":
        samplePath = "/samples/sample-glioma.jpg";
        name = "reference_brain_mri_glioma.jpg";
        typeHint = "Brain MRI (Axial T1/T2 View)";
        estSize = 30302;
        targetModality = "brain";
        break;

      case "meningioma":
        samplePath = "/samples/sample-meningioma.jpg";
        name = "reference_brain_mri_meningioma.jpg";
        typeHint = "Brain MRI (Axial T1-Contrast)";
        estSize = 18697;
        targetModality = "brain";
        break;

      case "pituitary":
        samplePath = "/samples/sample-pituitary.jpg";
        name = "reference_brain_mri_pituitary.jpg";
        typeHint = "Brain MRI (Sellar / Suprasellar)";
        estSize = 26572;
        targetModality = "brain";
        break;

      case "notumor":
        samplePath = "/samples/sample-notumor.jpg";
        name = "reference_brain_mri_unremarkable.jpg";
        typeHint = "Brain MRI (Axial Normal)";
        estSize = 9447;
        targetModality = "brain";
        break;

      case "ulcer":
        samplePath = "/samples/sample-ulcer.jpg";
        name = "reference_endoscopy_ulcer.jpg";
        typeHint = "GI Endoscopic Frame (Mucosal Ulcer)";
        estSize = 229285;
        targetModality = "gastro";
        break;

      case "gastro-normal":
        samplePath = "/samples/sample-gastro-normal.jpg";
        name = "reference_endoscopy_normal.jpg";
        typeHint = "GI Endoscopic Frame (Intact Mucosa)";
        estSize = 243459;
        targetModality = "gastro";
        break;

      default:
        samplePath = "/samples/sample-pneumonia.jpeg";
        name = "reference_sample.jpg";
        typeHint = "Medical Radiograph";
        estSize = 50000;
        targetModality = "chest";
    }

    if (selectedModality === "auto") {
      setSelectedModality(targetModality);
    }

    setSelectedFile(samplePath);
    setPreviewUrl(samplePath);
    setFileInfo({
      name: name,
      sizeFormatted: formatBytes(estSize),
      typeHint: typeHint,
    });
  };

  const handleAnalyze = async () => {
    if (!selectedFile && !previewUrl) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const data = await analyzeMedicalImage(selectedFile, activePreset, selectedModality);
      setResult(data);

      if (data.modality && selectedModality === "auto") {
        setSelectedModality(data.modality);
      }

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 150);
    } catch (err) {
      console.error(err);
      setError("Unable to process the scan. Please check your image format or retry.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileInfo(null);
    setActivePreset(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#070b10] text-[#e6edf5]">
      {/* Top Application Bar with Glassmorphic Style */}
      <header className="border-b border-[#1c2633] bg-[#0b0f15]/90 px-4 sm:px-8 py-3.5 flex items-center justify-between sticky top-0 z-30 backdrop-blur-md shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center text-white font-black text-sm shadow-md shadow-blue-500/25">
            M
          </div>
          <span className="font-extrabold text-sm sm:text-base tracking-tight text-[#e6edf5]">
            MediScan AI
          </span>
          <span className="text-xs text-[#536173] border-l border-[#1c2633] pl-3 hidden sm:inline font-medium">
            Clinical Decision Support Workstation
          </span>
        </div>

        <div className="flex items-center gap-2">
          {previewUrl && (
            <motion.button
              type="button"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleClear}
              aria-label="Load a new scan"
              className="text-xs text-[#8b99a9] hover:text-[#e6edf5] px-3.5 py-1.5 rounded-lg border border-[#1c2633] hover:border-[#2b394d] bg-[#0e141d] hover:bg-[#151e2b] transition-all cursor-pointer flex items-center gap-1.5 focus-visible:ring-2 focus-visible:ring-[#38bdf8]"
            >
              <RotateCcw className="w-3 h-3" />
              <span>New Scan</span>
            </motion.button>
          )}
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Model Tabs & Header */}
        <HeroSection
          onSelectSample={handleSelectSample}
          activePreset={activePreset}
          selectedModality={selectedModality}
          onSelectModality={handleSelectModality}
          isAnalyzing={isAnalyzing}
        />

        {/* Dynamic Model Benchmark Dashboard */}
        <ModelBenchmarkBar
          selectedModality={selectedModality}
          onSelectModality={handleSelectModality}
          isAnalyzing={isAnalyzing}
        />

        {/* Workstation Console (Two Column) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 sm:gap-8 items-start">
          {/* Left Column: Image Ingestion & Lightboard */}
          <div className="lg:col-span-6 space-y-4">
            <div className="flex items-center justify-between text-xs font-bold text-[#8b99a9]">
              <span>Diagnostic Radiograph View</span>
              {fileInfo && (
                <span className="font-mono text-[11px] text-[#536173]">
                  {fileInfo.typeHint}
                </span>
              )}
            </div>

            {!previewUrl ? (
              <ImageUploader
                onImageSelected={handleImageSelected}
                disabled={isAnalyzing}
                selectedModality={selectedModality}
              />
            ) : (
              <PreviewPanel
                previewUrl={previewUrl}
                fileInfo={fileInfo}
                onClear={handleClear}
                isAnalyzing={isAnalyzing}
                gradcamUrl={result?.gradcam_base64}
              />
            )}

            <AnalyzeButton
              onClick={handleAnalyze}
              isAnalyzing={isAnalyzing}
              hasImage={Boolean(previewUrl)}
              hasResult={Boolean(result)}
            />

            {error && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl bg-[#220e14] border border-[#481622] text-xs text-[#fda4af] leading-relaxed shadow"
              >
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
                  <span>{error}</span>
                </div>
              </motion.div>
            )}
          </div>

          {/* Right Column: Findings and Clinical Report */}
          <div ref={resultsRef} className="lg:col-span-6 space-y-5">
            <div className="flex items-center justify-between text-xs font-bold text-[#8b99a9]">
              <span>Clinical Intelligence Findings</span>
              {result && (
                <span className="font-mono text-[11px] text-[#10b981] font-semibold flex items-center gap-1">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Analysis Complete
                </span>
              )}
            </div>

            <AnimatePresence mode="wait">
              {isAnalyzing && (
                <LoadingOverlay
                  key="loading"
                  selectedModality={selectedModality}
                />
              )}

              {!isAnalyzing && result && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.35 }}
                  className="space-y-5"
                >
                  <ResultCard result={result} />
                  <ConfidenceChart
                    classes={result.all_classes}
                    decisionThreshold={result.decisionThreshold}
                    modality={result.modality}
                  />
                  <DiagnosisDetail result={result} />
                </motion.div>
              )}

              {!isAnalyzing && !result && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="rounded-2xl border border-[#1c2633] bg-[#0e141d] p-8 sm:p-12 text-center min-h-[380px] flex flex-col items-center justify-center shadow-inner"
                >
                  <div className="w-14 h-14 rounded-2xl bg-[#141d2a] border border-[#212f42] flex items-center justify-center text-[#60a5fa] mb-3 shadow-md">
                    <Microscope className="w-7 h-7" />
                  </div>
                  <p className="text-sm font-bold text-[#e6edf5]">
                    Diagnostic Console Ready
                  </p>
                  <p className="text-xs text-[#718294] max-w-sm mt-2 leading-relaxed font-medium">
                    Upload an image on the left or click any of the reference scans above to evaluate findings, Grad-CAM attention heatmaps, and calibrated probability scores.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Regulatory Footer */}
      <div className="px-4 sm:px-6 lg:px-8 border-t border-[#141b25] bg-[#070a0e]">
        <DisclaimerBanner />
      </div>
    </div>
  );
}
