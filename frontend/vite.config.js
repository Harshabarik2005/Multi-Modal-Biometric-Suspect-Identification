import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // 4173, not vite's default 5173, which is taken by another project
    // on this machine. package.json passes it too, so `npx vite` and
    // `npm run dev` land on the same port.
    port: 4173,
    // The API runs separately (scripts/serve.py). Proxying keeps the browser
    // on one origin so there is no CORS configuration to get wrong.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
