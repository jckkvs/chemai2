// frontend_next/src/app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ChemAI Nexus',
  description: 'Chemoinformatics Machine Learning Platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body className={`${inter.className} min-h-screen bg-slate-50 text-slate-900 antialiased`}>
        <Providers>
          <div className="relative flex min-h-screen flex-col">
            {/* Global Header */}
            <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md">
              <div className="container mx-auto flex h-16 items-center justify-between px-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-200">
                    <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a2 2 0 00-1.96 1.414l-.718 2.154a2 2 0 01-3.04 1.172l-1.403-1.052a2 2 0 00-1.802-.276l-2.292.655a2 2 0 01-2.48-2.48l.655-2.292a2 2 0 00-.276-1.802L1.65 10.403a2 2 0 011.172-3.04l2.154-.718a2 2 0 001.414-1.96l.477-2.387a2 2 0 00.547-1.022M15 15a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold tracking-tight text-slate-900">ChemAI Nexus</h1>
                    <p className="text-[10px] font-medium uppercase tracking-widest text-slate-500">Intelligent Chemistry Lab</p>
                  </div>
                </div>
                
                <nav className="hidden md:flex items-center gap-1">
                  {[
                    { label: 'Dashboard', href: '/' },
                    { label: 'Data', href: '/data' },
                    { label: 'Pipeline', href: '/pipeline' },
                    { label: 'Results', href: '/results' },
                  ].map((item) => (
                    <a
                      key={item.label}
                      href={item.href}
                      className="rounded-lg px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900"
                    >
                      {item.label}
                    </a>
                  ))}
                </nav>
                
                <div className="flex items-center gap-4">
                  <div className="hidden h-8 w-[1px] bg-slate-200 lg:block" />
                  <button className="flex h-10 items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition-all hover:bg-slate-800 active:scale-95">
                    <span>Export Project</span>
                  </button>
                </div>
              </div>
            </header>
            
            {/* Main Content */}
            <main className="flex-1">
              {children}
            </main>
            
            {/* Footer */}
            <footer className="border-t border-slate-200 bg-white py-8 text-center">
              <div className="container mx-auto px-4">
                <p className="text-sm text-slate-500">© 2026 ChemAI Nexus. Advanced Chemoinformatics Platform.</p>
              </div>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  );
}
