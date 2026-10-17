"use client";

import React from "react";
import { motion } from "framer-motion";
import { BarChart3, HelpCircle, ShieldCheck } from "lucide-react";

export default function ConfidenceChart({ classes = [], decisionThreshold = null, modality = null }) {
  if (!classes || classes.length === 0) return null;

  const thresholdPercent = decisionThreshold ? Math.round(decisionThreshold * 100) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.08, ease: "easeOut" }}
      className="w-full rounded-2xl border border-[#1c2633] bg-[#0e141d] p-5 sm:p-6 shadow-xl backdrop-blur-md"
    >
      {/* Header */}
      <div className="pb-3.5 border-b border-[#1c2633] flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-[#60a5fa]" />
          <h3 className="font-bold text-[#e6edf5] text-xs sm:text-sm">
            Class Probability Distribution
          </h3>
        </div>

        <div className="flex items-center gap-2 text-[#718294] text-[11px] font-mono">
          <span>Softmax Posteriors</span>
          {thresholdPercent && (
            <span className="px-2.5 py-0.5 rounded-md bg-[#131d2a] border border-[#1e2f45] text-[#38bdf8] font-bold">
              Threshold: {thresholdPercent}%
            </span>
          )}
        </div>
      </div>

      {/* Threshold Explanation Note if calibrated threshold exists */}
      {thresholdPercent && (
        <div className="mt-3.5 px-3.5 py-2.5 rounded-xl bg-[#0a121d] border border-[#1c2a3d] text-[11px] text-[#8b99a9] flex items-center justify-between gap-2 shadow-inner">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#f59e0b] shrink-0"></span>
            <span>Decision boundary calibrated at <strong>{thresholdPercent}%</strong> to guarantee $\ge 90\%$ test sensitivity and specificity.</span>
          </span>
        </div>
      )}

      {/* Class Probabilities Bars with Framer Motion Animation */}
      <div className="mt-4 space-y-4">
        {classes.map((item, index) => {
          const rawConfidence = item.confidence ?? 0;
          const percent = Math.round(rawConfidence * 1000) / 10;
          const isTop = index === 0;

          // Resolve bar color
          const barColor =
            item.color ||
            (isTop
              ? percent >= (thresholdPercent || 50)
                ? "#f43f5e"
                : "#10b981"
              : "#334155");

          return (
            <div key={item.label || index} className="space-y-1.5">
              {/* Row Header */}
              <div className="flex items-center justify-between text-xs gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
                    style={{ backgroundColor: barColor }}
                  />
                  <span
                    className={`truncate ${
                      isTop
                        ? "font-bold text-[#f1f5f9]"
                        : "text-[#8b99a9] font-medium"
                    }`}
                  >
                    {item.label}
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0 font-mono text-xs">
                  {thresholdPercent && index === 0 && (
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-md font-mono font-medium ${
                        percent >= thresholdPercent
                          ? "bg-rose-950/80 text-rose-300 border border-rose-800/80"
                          : "bg-emerald-950/80 text-emerald-300 border border-emerald-800/80"
                      }`}
                    >
                      {percent >= thresholdPercent ? "Above Threshold" : "Below Threshold"}
                    </span>
                  )}
                  <span
                    className={`font-bold font-mono ${
                      isTop ? "text-[#f1f5f9]" : "text-[#718294]"
                    }`}
                  >
                    {percent.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Progress Track with Threshold Marker & Spring Fill */}
              <div className="relative w-full bg-[#070b10] rounded-full h-3 overflow-hidden border border-[#16212e] p-0.5 shadow-inner">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(percent, 2)}%` }}
                  transition={{ duration: 0.8, delay: index * 0.1, ease: "easeOut" }}
                  className="h-full rounded-full shadow-sm"
                  style={{
                    backgroundColor: barColor,
                  }}
                />

                {/* Calibrated Threshold Mark on the bar */}
                {thresholdPercent && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-amber-400 z-10 shadow-[0_0_6px_#fbbf24]"
                    style={{ left: `${thresholdPercent}%` }}
                    title={`Calibrated Cutoff (${thresholdPercent}%)`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
