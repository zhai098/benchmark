'use client';

import { motion } from 'framer-motion';

export function FooterCta() {
  return (
    <section className="section-shell pb-16">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="card-surface flex flex-col items-start justify-between gap-6 p-8 md:flex-row md:items-center"
      >
        <div>
          <h3 className="text-2xl font-semibold text-white">Ready to modernize product operations?</h3>
          <p className="mt-2 text-sm leading-7 text-slate-300">Deploy in days, not quarters, with a stack built for performance and governance.</p>
        </div>
        <button className="rounded-xl bg-indigo-500 px-6 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400">
          Request access
        </button>
      </motion.div>
    </section>
  );
}
