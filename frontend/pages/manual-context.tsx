import { useEffect, useState } from 'react'
import Link from 'next/link'
import {
  fetchDocs,
  fetchDocumentSegments,
  manualAsk,
} from '@/utils/api'

interface Doc {
  document_id: string
  file_name: string
}

interface Segment {
  text: string
  chunk_index?: number
}

export default function ManualContext() {
  const [docs, setDocs] = useState<Doc[]>([])
  const [segments, setSegments] = useState<Record<string, Segment[]>>({})
  const [showDoc, setShowDoc] = useState<Record<string, boolean>>({})
  const [selected, setSelected] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [style, setStyle] = useState('')
  const [answer, setAnswer] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchDocs().then(data => setDocs(data.documents || []))
  }, [])

  async function toggleDoc(id: string) {
    if (!showDoc[id]) {
      const res = await fetchDocumentSegments(id)
      setSegments(prev => ({ ...prev, [id]: res.segments || [] }))
    }
    setShowDoc(prev => ({ ...prev, [id]: !prev[id] }))
  }

  function toggleSeg(text: string) {
    if (selected.includes(text)) {
      setSelected(selected.filter(t => t !== text))
    } else {
      setSelected([...selected, text])
    }
  }

  async function handleAsk() {
    if (!question || selected.length === 0) return
    setLoading(true)
    try {
      const res = await manualAsk(question, selected, style)
      setAnswer(res.answer)
      setElapsed(res.elapsed || 0)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      <Link href="/" className="text-blue-600 underline">
        返回
      </Link>
      <h1 className="text-2xl font-bold">自選段落模式</h1>
      <div className="space-y-2">
        {docs.map(doc => (
          <div key={doc.document_id} className="border p-2 rounded">
            <button
              type="button"
              onClick={() => toggleDoc(doc.document_id)}
              className="text-blue-600 underline"
            >
              {showDoc[doc.document_id] ? '隱藏' : '顯示'} {doc.file_name}
            </button>
            {showDoc[doc.document_id] && segments[doc.document_id] && (
              <ul className="space-y-1 mt-1">
                {segments[doc.document_id].map((s, idx) => (
                  <li key={idx}>
                    <label>
                      <input
                        type="checkbox"
                        className="mr-1"
                        checked={selected.includes(s.text)}
                        onChange={() => toggleSeg(s.text)}
                      />
                      [{s.chunk_index}] {s.text}
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={e => { e.preventDefault(); handleAsk() }} className="space-y-2">
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          className="w-full border p-2 rounded"
          placeholder="Ask a question"
        />
        <select
          value={style}
          onChange={e => setStyle(e.target.value)}
          className="border rounded p-2"
        >
          <option value="">預設</option>
          <option value="專業">專業</option>
          <option value="口語">口語</option>
          <option value="摘要">摘要</option>
          <option value="條列式">條列式</option>
        </select>
        <button
          type="submit"
          className="px-4 py-2 bg-green-600 text-white rounded"
          disabled={loading}
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>
      {answer && (
        <div className="space-y-2 bg-gray-100 p-4 rounded">
          <p className="text-sm text-gray-600">Time: {elapsed.toFixed(2)}s</p>
          <p>{answer}</p>
        </div>
      )}
    </div>
  )
}
