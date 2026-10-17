import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata = {
  title: "MediScan — Medical Scan Analysis",
  description: "Automated preliminary analysis for chest radiographs and brain MRI scans.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={sans.variable}>
      <body className="min-h-screen bg-[#090d12] text-[#e6edf5] font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
