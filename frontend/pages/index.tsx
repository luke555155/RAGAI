import UploadForm from '@/components/UploadForm'
import QnAForm from '@/components/QnAForm'
import { useEffect, useState } from 'react'
import { fetchDocs, fetchDocumentSegments } from '@/utils/api'

interface Doc {
  document_id: string
  file_name: string
  upload_time: string
}

interface Segment {
  text: string
  chunk_index?: number
}

export default function Home() {
  const [docs, setDocs] = useState<Doc[]>([])
  const [segments, setSegments] = useState<Segment[]>([])
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => {
    fetchDocs().then(data => setDocs(data.documents || []))
  }, [])

  async function handleSelect(id: string) {
    const data = await fetchDocumentSegments(id)
    setSelected(id)
    setSegments(data.segments || [])
  }

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-8">
      <h1 className="text-2xl font-bold text-center">Document Q&A</h1>
      <UploadForm />
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">檔案紀錄</h2>
        <ul className="list-disc list-inside space-y-1">
          {docs.map(doc => (
            <li key={doc.document_id}>
              <button
                className="text-blue-600 underline"
                onClick={() => handleSelect(doc.document_id)}
              >
                {doc.document_id} (
                {new Date(doc.upload_time).toLocaleString()})
              </button>
            </li>
          ))}
        </ul>
        {selected && (
          <div className="space-y-1 mt-2">
            <h3 className="font-semibold">Segments of {selected}</h3>
            <ul className="list-decimal list-inside space-y-1">
              {segments.map((s, idx) => (
                <li key={idx}>{s.text}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <QnAForm />
    </div>
  )
}
