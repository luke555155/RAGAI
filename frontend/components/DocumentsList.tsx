import { useEffect, useState } from 'react'
import Link from 'next/link'
import { fetchDocs, deleteDoc } from '@/utils/api'
import { useDoc } from './DocContext'

interface Doc {
  document_id: string
  file_name: string
  upload_time: string
  summary?: string
}

export default function DocumentsList() {
  const [docs, setDocs] = useState<Doc[]>([])
  const { selectedDocId, setSelectedDocId } = useDoc()

  useEffect(() => {
    fetchDocs().then(data => setDocs(data.documents || []))
  }, [])

  async function handleDelete(id: string) {
    if (!confirm('Delete document?')) return
    await deleteDoc(id)
    setDocs(docs.filter(d => d.document_id !== id))
    if (selectedDocId === id) {
      setSelectedDocId(null)
    }
  }

  return (
    <div className="space-y-2">
      <h2 className="text-xl font-semibold">檔案紀錄</h2>
      <ul className="space-y-2">
        {docs.map(doc => (
          <li key={doc.document_id} className="border p-2 rounded space-y-1">
            <div className="flex justify-between items-center">
              <div>
                <Link
                  href={`/docs/${doc.document_id}`}
                  className="text-blue-600 underline font-semibold"
                >
                  {doc.file_name}
                </Link>
                {doc.summary && (
                  <p className="text-sm text-gray-500">{doc.summary}</p>
                )}
                <p className="text-xs text-gray-600">
                  {new Date(doc.upload_time).toLocaleString()} | ID: {doc.document_id}
                </p>
              </div>
              <div className="space-x-2 whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => setSelectedDocId(doc.document_id)}
                  className="px-2 py-1 text-sm bg-green-500 text-white rounded"
                >
                  選擇
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(doc.document_id)}
                  className="px-2 py-1 text-sm bg-red-600 text-white rounded"
                >
                  刪除
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
