/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    formats: ['image/avif', 'image/webp']
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://127.0.0.1:5000';
    return [
      { source: '/annotator', destination: `${backend}/annotator` },
      { source: '/review', destination: `${backend}/review` },
      { source: '/static/:path*', destination: `${backend}/static/:path*` },
      { source: '/api/:path*', destination: `${backend}/api/:path*` }
    ];
  }
};

export default nextConfig;
