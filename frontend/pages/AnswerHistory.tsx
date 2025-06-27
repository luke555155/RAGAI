import { useEffect, useState } from 'react'
import Link from 'next/link'
import { fetchAskLog, askQuestion } from '@/utils/api'
import { useDoc } from '@/components/DocContext'

interface LogEntry {
  timestamp: string
  question: string
  answer: string
  document_ids?: string[]
  rerank_mode?: string
  style?: string
  references: { text: string }[]
}

export default function AnswerHistory() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState('')
  const { setSelectedDocIds } = useDoc()

  useEffect(() => {
    fetchAskLog().then(data => setLogs(data.logs || []))
  }, [])

  const filtered = logs.filter(l =>
    l.question.toLowerCase().includes(filter.toLowerCase())
  )

  async function handleReask(entry: LogEntry) {
    setSelectedDocIds(entry.document_ids || [])
    try {
      const res = await askQuestion(entry.question, entry.document_ids || [], entry.style)
      alert(res.answer)
    } catch {
      alert('Error')
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      <Link href="/" className="text-blue-600 underline">
        返回
      </Link>
      <h1 className="text-2xl font-bold">回答歷史</h1>
      <input
        type="text"
        value={filter}
        onChange={e => setFilter(e.target.value)}
        placeholder="Filter"
        className="border p-2 rounded w-full"
      />
      <ul className="space-y-2">
        {filtered.map((log, idx) => (
          <li key={idx} className="border p-2 rounded space-y-1">
            <p className="text-xs text-gray-600">
              {new Date(log.timestamp).toLocaleString()} | rerank: {log.rerank_mode}
            </p>
            <p className="font-semibold">Q: {log.question}</p>
            <p>A: {log.answer}</p>
            <button
              type="button"
              onClick={() => handleReask(log)}
              className="text-sm text-green-700 underline"
            >
              重新查詢
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
