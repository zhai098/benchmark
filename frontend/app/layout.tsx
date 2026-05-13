import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Annotation Platform',
  description: 'Annotator-focused workflow with reviewer-only controls and editable instructions.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
