import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Northstar — Product intelligence workspace',
  description: 'Premium SaaS landing page built with Next.js, TypeScript, Tailwind, Motion, and GSAP.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
