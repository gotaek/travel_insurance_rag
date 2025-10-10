import React from 'react';

/**
 * 상단 헤더. 서비스명을 제공합니다.
 */
export function Header(): JSX.Element {
  return (
    <header 
      className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm"
      role="banner"
    >
      <div className="mx-auto max-w-3xl px-4 py-4 flex items-center justify-center">
        <h1 
          className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent cursor-pointer hover:opacity-80 transition-opacity duration-200"
          onClick={() => window.location.reload()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              window.location.reload();
            }
          }}
          aria-label="페이지 새로고침"
        >
          Travel Shield
        </h1>
      </div>
    </header>
  );
}


