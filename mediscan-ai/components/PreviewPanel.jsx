"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ZoomIn, Eye, Sparkles, Trash2, Maximize2, ShieldCheck, Activity } from "lucide-react";

export default function PreviewPanel({
  previewUrl,
  fileInfo,
  onClear,
  isAnalyzing,
  gradcamUrl,
}) {
  const [showGradCam, setShowGradCam] = useState(Boolean(gradcamUrl));
  const [zoomLevel, setZoomLevel] = useState(1);

  // Sync showGradCam when new gradcamUrl arrives
  React.useEffect(() => {
    if (gradcamUrl) {
      setShowGradCam(true);
    }
  }, [gradcamUrl]);

  if (!previewUrl) return null;

  const currentDisplayUrl = showGradCam && gradcamUrl ? gradcamUrl : previewUrl;

  const toggleZoom = () => {
    setZoomLevel((prev) => (prev === 1 ? 1.3 : prev === 1.3 ? 1.6 : 1));
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="w-full rounded-2xl border border-[#1c2633] bg-[#0e141d] overflow-hidden shadow-xl transition-all"
    >
      {/* Lightboard Header Controls */}
      <div className="px-3.5 sm:px-4 py-2.5 border-b border-[#1c2633] bg-[#0a0f16] flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse"></span>
          <span className="font-bold text-[#e6edf5] text-xs">
            Diagnostic Lightboard
          </span>
          {gradcamUrl && (
            <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-mono text-[#60a5fa] bg-[#111f38] px-2.5 py-0.5 rounded-md border border-[#1e3a6a]">
              <Sparkles className="w-3 h-3 text-[#38bdf8]" />
              CBAM Heatmap Ready
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Zoom Toggle Button */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleZoom}
            className="text-[11px] font-mono text-[#8b99a9] hover:text-[#e6edf5] bg-[#141d29] hover:bg-[#1c2738] border border-[#223042] px-2.5 py-1 rounded-lg transition-colors cursor-pointer flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-[#38bdf8]"
            title="Toggle Lightboard Zoom"
            aria-label="Toggle Zoom Level"
          >
            <ZoomIn className="w-3.5 h-3.5" />
            <span>{zoomLevel === 1 ? "1.0x" : `${zoomLevel}x`}</span>
          </motion.button>

          {/* Clear Scan Button */}
          <motion.button
            type="button"
            id="clear-image-btn"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            disabled={isAnalyzing}
            onClick={onClear}
            className="text-xs text-[#8b99a9] hover:text-[#fda4af] bg-[#141d29] hover:bg-[#27151a] hover:border-[#4d1f27] border border-[#223042] px-2.5 py-1 rounded-lg transition-colors cursor-pointer disabled:opacity-40 flex items-center gap-1 focus-visible:ring-2 focus-visible:ring-rose-500"
            aria-label="Clear loaded scan"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear</span>
          </motion.button>
        </div>
      </div>

      {/* Lightboard Image Frame */}
      <div className="relative w-full bg-[#030507] min-h-[300px] sm:min-h-[360px] max-h-[500px] flex items-center justify-center p-3 overflow-hidden select-none">
        {/* Active Laser Scanning Beam Overlay (when isAnalyzing) */}
        {isAnalyzing && (
          <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
            <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-[#38bdf8] to-transparent shadow-[0_0_15px_#38bdf8] animate-scan-beam" />
            <div className="absolute inset-0 bg-blue-500/5 backdrop-blur-[1px]" />
          </div>
        )}

        {/* View Toggle Bar (Floating Top Left) */}
        {gradcamUrl && (
          <div className="absolute top-3 left-3 right-3 sm:right-auto z-10 flex items-center justify-center sm:justify-start bg-[#0a0f16]/95 backdrop-blur-md border border-[#263548] rounded-xl p-1 text-[11px] shadow-lg max-w-full overflow-x-auto">
            <button
              type="button"
              onClick={() => setShowGradCam(false)}
              className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer whitespace-nowrap font-medium ${
                !showGradCam
                  ? "bg-[#2563eb] text-white shadow"
                  : "text-[#8b99a9] hover:text-[#e6edf5]"
              }`}
            >
              Original Scan
            </button>
            <button
              type="button"
              onClick={() => setShowGradCam(true)}
              className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 whitespace-nowrap font-medium ${
                showGradCam
                  ? "bg-[#e11d48] text-white shadow"
                  : "text-[#8b99a9] hover:text-[#e6edf5]"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-white animate-ping"></span>
              Grad-CAM Heatmap
            </button>
          </div>
        )}

        {/* The Diagnostic Medical Image with Framer Motion Zoom */}
        <motion.div
          animate={{ scale: zoomLevel }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
          className="flex items-center justify-center max-w-full w-full"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={currentDisplayUrl}
            alt="Loaded medical scan"
            className="max-h-[380px] sm:max-h-[440px] w-auto max-w-full object-contain select-none rounded-lg shadow-md"
          />
        </motion.div>

        {/* Heatmap Legend Overlay (When Grad-CAM active) */}
        {gradcamUrl && showGradCam && (
          <div className="absolute bottom-3 right-3 z-10 bg-[#0a0f16]/90 backdrop-blur-md border border-[#263548] rounded-lg px-3 py-1.5 flex items-center gap-2.5 text-[10px] text-[#8b99a9] shadow-lg">
            <span>Low Activation</span>
            <div className="w-14 sm:w-20 h-2.5 rounded-full bg-gradient-to-r from-blue-600 via-yellow-400 to-red-600 shadow-inner"></div>
            <span className="text-[#fda4af] font-semibold">High Focus</span>
          </div>
        )}
      </div>

      {/* Metadata Strip */}
      <div className="px-4 py-3 border-t border-[#1c2633] bg-[#0a0f16] flex flex-wrap items-center justify-between gap-2 text-xs text-[#8b99a9]">
        <div className="min-w-0 flex items-center gap-2">
          <span className="font-semibold text-[#e6edf5] truncate max-w-[180px] sm:max-w-[280px]">
            {fileInfo?.name || "clinical_scan.dcm"}
          </span>
          <span className="text-[#536173]">•</span>
          <span className="text-[#536173] font-mono">{fileInfo?.sizeFormatted}</span>
        </div>

        <div className="flex items-center gap-2">
          {gradcamUrl && showGradCam && (
            <span className="text-[10px] uppercase font-bold text-[#fda4af] bg-[#2d1217] px-2 py-0.5 rounded-md border border-[#4c1d24]">
              Attention Heatmap
            </span>
          )}
          <span className="text-[#8b99a9] font-medium">
            {fileInfo?.typeHint || "Medical Examination"}
          </span>
        </div>
      </div>
    </motion.div>
  );
}
