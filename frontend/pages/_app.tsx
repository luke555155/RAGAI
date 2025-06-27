import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { DocProvider } from '@/components/DocContext'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <DocProvider>
      <Component {...pageProps} />
    </DocProvider>
  )
}
