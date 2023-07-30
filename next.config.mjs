const getCorsHeaders = () => {
  const headers = [
    { key: "Access-Control-Allow-Credentials", value: "true" },
    { key: "Access-Control-Allow-Origin", value: "*" },
    { key: "Access-Control-Allow-Methods", value: "GET,OPTIONS,PATCH,DELETE,POST,PUT" },
    { key: "Access-Control-Allow-Headers", value: "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version" },
  ];

  return headers;
};

const nextConfig = {
  reactStrictMode: false,
  swcMinify: true,
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
  webpack(config) {
    config.experiments = { ...config.experiments, topLevelAwait: true };
    return config;
  },
  headers: async () => {
    return [
      {
        source: "/api/:path*",
        headers: getCorsHeaders(),
      },
    ];
  },
};

export default nextConfig;
