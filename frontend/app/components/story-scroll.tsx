'use client';

import { useEffect, useRef } from 'react';

export function StoryScroll() {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let cleanup = () => {};

    (async () => {
      const gsap = (await import('gsap')).default;
      const { ScrollTrigger } = await import('gsap/ScrollTrigger');
      gsap.registerPlugin(ScrollTrigger);

      const ctx = gsap.context(() => {
        gsap.fromTo(
          '.story-panel',
          { opacity: 0.5, y: 24 },
          {
            opacity: 1,
            y: 0,
            stagger: 0.18,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: rootRef.current,
              start: 'top 70%',
              end: 'bottom 30%',
              scrub: 0.6
            }
          }
        );
      }, rootRef);

      cleanup = () => ctx.revert();
    })();

    return () => cleanup();
  }, []);

  return (
    <section ref={rootRef} className="section-shell pb-20 md:pb-28">
      <div className="card-surface overflow-hidden p-8 md:p-10">
        <div className="grid gap-6 md:grid-cols-3">
          {[
            'Consolidate product context across roadmap, design, and engineering.',
            'Automate decision handoffs with approval-ready briefing packets.',
            'Measure execution quality continuously with leadership-level confidence.'
          ].map((text) => (
            <article key={text} className="story-panel rounded-xl border border-white/10 bg-white/[0.02] p-5">
              <p className="text-sm leading-7 text-slate-200">{text}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
