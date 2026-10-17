"use client";

import React from "react";
import { motion } from "framer-motion";
import { Loader2, Sparkles, RotateCcw, Play } from "lucide-react";

export default function AnalyzeButton({ onClick, isAnalyzing, hasImage, hasResult }) {
  return (
    <div className="w-full mt-4">
      <motion.button
        type="button"
        id="analyze-prediction-btn"
        whileHover={hasImage && !isAnalyzing ? { scale: 1.015 } : {}}
        whileTap={hasImage && !isAnalyzing ? { scale: 0.985 } : {}}
        disabled={!hasImage || isAnalyzing}
        onClick={onClick}
        aria-label={hasResult ? "Re-run clinical analysis" : "Run neural network analysis"}
        className={`w-full py-3.5 px-4 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 cursor-pointer focus-visible:ring-2 focus-visible:ring-[#38bdf8] ${
          !hasImage
            ? "bg-[#141b24] text-[#536173] border border-[#1e2733] cursor-not-allowed opacity-60"
            : isAnalyzing
            ? "bg-[#14233c] text-[#93c5fd] border border-[#2b4c8a] cursor-wait shadow-lg"
            : "bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600 text-white hover:opacity-95 shadow-lg shadow-blue-600/25 active:shadow-sm"
        }`}
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-[#93c5fd]" />
            <span>Evaluating Neural Layers...</span>
          </>
        ) : hasResult ? (
          <>
            <RotateCcw className="w-4 h-4" />
            <span>Re-Analyze Scan</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-blue-200" />
            <span>Run Clinical Diagnostics</span>
          </>
        )}
      </motion.button>

      {!hasImage && (
        <p className="text-center text-xs text-[#536173] mt-2 font-medium">
          Select or drop a scan above to initiate inference
        </p>
      )}
    </div>
  );
}
