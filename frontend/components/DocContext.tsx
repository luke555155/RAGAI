import { createContext, useContext, useState } from 'react'

interface DocState {
  selectedDocId: string | null
  setSelectedDocId: (id: string | null) => void
}

const DocContext = createContext<DocState>({
  selectedDocId: null,
  setSelectedDocId: () => {},
})

export function useDoc() {
  return useContext(DocContext)
}

export function DocProvider({ children }: { children: React.ReactNode }) {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  return (
    <DocContext.Provider value={{ selectedDocId, setSelectedDocId }}>
      {children}
    </DocContext.Provider>
  )
}
