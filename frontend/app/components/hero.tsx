'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import Link from 'next/link';

export function Hero() {
  return (
    <section className="section-shell pt-16 md:pt-24">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="grid items-center gap-12 lg:grid-cols-[1.08fr_0.92fr]"
      >
        <div className="space-y-6">
          <p className="inline-flex rounded-full border border-indigo-300/25 bg-indigo-400/10 px-4 py-1 text-xs font-medium uppercase tracking-[0.18em] text-indigo-100">
            Northstar Platform
          </p>
          <h1 className="text-balance text-4xl font-semibold leading-tight text-white sm:text-5xl lg:text-6xl">
            Orchestrate product decisions with faster context, cleaner workflows, and quieter execution.
          </h1>
          <p className="max-w-2xl text-pretty text-base leading-7 text-slate-300 sm:text-lg">
            Northstar combines collaborative planning, operational telemetry, and AI-guided prioritization in a single
            workspace designed for product organizations that ship at enterprise velocity.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <Link
              href="/annotator"
              className="rounded-xl bg-white px-6 py-3 text-sm font-semibold text-slate-900 transition hover:translate-y-[-1px] hover:bg-indigo-50"
            >
              Start free trial
            </Link>
            <Link
              href="/review"
              className="rounded-xl border border-white/20 px-6 py-3 text-sm font-semibold text-white transition hover:border-white/40 hover:bg-white/5"
            >
              Open review panel
            </Link>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.12, ease: 'easeOut' }}
          className="card-surface overflow-hidden"
        >
          <Image
            src="/dashboard-shot.svg"
            alt="Northstar dashboard preview"
            width={1120}
            height={820}
            priority
            className="h-auto w-full"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}
