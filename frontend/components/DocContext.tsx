import { createContext, useContext, useState } from 'react'

interface DocState {
  selectedDocIds: string[]
  setSelectedDocIds: (ids: string[]) => void
}

const DocContext = createContext<DocState>({
  selectedDocIds: [],
  setSelectedDocIds: () => {},
})

export function useDoc() {
  return useContext(DocContext)
}

export function DocProvider({ children }: { children: React.ReactNode }) {
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  return (
    <DocContext.Provider value={{ selectedDocIds, setSelectedDocIds }}>
      {children}
    </DocContext.Provider>
  )
}
