"use client";

import React from "react";

export default function DisclaimerBanner() {
  return (
    <footer className="w-full mt-16 pt-6 pb-12 border-t border-[#1c2633] text-xs text-[#536173]">
      <div className="max-w-6xl mx-auto space-y-2">
        <p className="leading-relaxed">
          <strong className="text-[#8b99a9] font-medium">Investigational notice:</strong> This interface provides automated preliminary feature evaluation based on deep neural network models trained on public radiographic datasets. It does not provide medical diagnoses, treatment recommendations, or prescriptions. All outputs must be evaluated by a qualified healthcare provider.
        </p>
        <p className="text-[11px] text-[#3d4b5c]">
          Images uploaded in this browser session are processed locally or in-memory without persistent user profiling.
        </p>
      </div>
    </footer>
  );
}
