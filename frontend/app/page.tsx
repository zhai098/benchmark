import { FeatureGrid } from './components/feature-grid';
import { FooterCta } from './components/footer-cta';
import { Hero } from './components/hero';
import { StoryScroll } from './components/story-scroll';

export default function HomePage() {
  return (
    <main>
      <Hero />
      <FeatureGrid />
      <StoryScroll />
      <FooterCta />
    </main>
  );
}
