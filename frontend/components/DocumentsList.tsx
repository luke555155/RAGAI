import { useEffect, useState } from 'react'
import Link from 'next/link'
import { fetchDocs, deleteDoc, resummarizeDoc } from '@/utils/api'
import { useDoc } from './DocContext'

interface Doc {
  document_id: string
  file_name: string
  upload_time: string
  summary?: string
  tags?: string[]
}

export default function DocumentsList() {
  const [docs, setDocs] = useState<Doc[]>([])
  const [filterTag, setFilterTag] = useState('')
  const { selectedDocIds, setSelectedDocIds } = useDoc()

  useEffect(() => {
    fetchDocs().then(data => setDocs(data.documents || []))
  }, [])

  async function handleDelete(id: string) {
    if (!confirm('Delete document?')) return
    await deleteDoc(id)
    setDocs(docs.filter(d => d.document_id !== id))
    if (selectedDocIds.includes(id)) {
      setSelectedDocIds(selectedDocIds.filter(d => d !== id))
    }
  }

  const tags = Array.from(new Set(docs.flatMap(d => d.tags || [])))
  const filtered = filterTag
    ? docs.filter(d => (d.tags || []).includes(filterTag))
    : docs

  return (
    <div className="space-y-2">
      <h2 className="text-xl font-semibold">檔案紀錄</h2>
      {tags.length > 0 && (
        <select
          value={filterTag}
          onChange={e => setFilterTag(e.target.value)}
          className="border p-1 rounded text-sm"
        >
          <option value="">全部</option>
          {tags.map(t => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      )}
      <ul className="space-y-2">
        {filtered.map(doc => (
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
                {doc.tags && doc.tags.length > 0 && (
                  <p className="text-xs text-gray-600">Tags: {doc.tags.join(', ')}</p>
                )}
                <p className="text-xs text-gray-600">
                  {new Date(doc.upload_time).toLocaleString()} | ID: {doc.document_id}
                </p>
              </div>
              <div className="space-x-2 whitespace-nowrap">
                <label className="px-2 py-1 text-sm bg-green-500 text-white rounded cursor-pointer">
                  <input
                    type="checkbox"
                    className="mr-1"
                    checked={selectedDocIds.includes(doc.document_id)}
                    onChange={() => {
                      if (selectedDocIds.includes(doc.document_id)) {
                        setSelectedDocIds(selectedDocIds.filter(d => d !== doc.document_id))
                      } else {
                        setSelectedDocIds([...selectedDocIds, doc.document_id])
                      }
                    }}
                  />
                  選擇
                </label>
                <button
                  type="button"
                  onClick={() => handleDelete(doc.document_id)}
                  className="px-2 py-1 text-sm bg-red-600 text-white rounded"
                >
                  刪除
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    const res = await resummarizeDoc(doc.document_id)
                    setDocs(
                      docs.map(d =>
                        d.document_id === doc.document_id
                          ? { ...d, summary: res.summary }
                          : d
                      )
                    )
                  }}
                  className="px-2 py-1 text-sm bg-blue-600 text-white rounded"
                >
                  重新摘要
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
