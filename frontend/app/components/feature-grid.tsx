'use client';

import { motion } from 'framer-motion';

const features = [
  {
    title: 'Decision inbox',
    body: 'Centralize initiative updates, unresolved blockers, and dependency risks in one high-signal queue.'
  },
  {
    title: 'Planning intelligence',
    body: 'Auto-surface impacted teams, release windows, and technical constraints before scope commits.'
  },
  {
    title: 'Execution analytics',
    body: 'Track cycle time, handoff latency, and roadmap confidence with executive-ready visual clarity.'
  },
  {
    title: 'Governance guardrails',
    body: 'Use policy-aware workflows and approval layers without slowing product delivery tempo.'
  }
];

export function FeatureGrid() {
  return (
    <section className="section-shell py-20 md:py-28">
      <div className="mb-10 max-w-2xl space-y-3">
        <p className="text-xs font-medium uppercase tracking-[0.16em] text-cyan-200/80">Capabilities</p>
        <h2 className="text-3xl font-semibold leading-tight text-white md:text-4xl">Built for disciplined teams that move quickly.</h2>
      </div>
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '-80px' }}
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.08 } }
        }}
        className="grid gap-5 sm:grid-cols-2"
      >
        {features.map((item) => (
          <motion.article
            key={item.title}
            variants={{ hidden: { opacity: 0, y: 18 }, visible: { opacity: 1, y: 0 } }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
            whileHover={{ y: -4 }}
            className="card-surface p-6"
          >
            <h3 className="mb-3 text-lg font-semibold text-white">{item.title}</h3>
            <p className="text-sm leading-7 text-slate-300">{item.body}</p>
          </motion.article>
        ))}
      </motion.div>
    </section>
  );
}
