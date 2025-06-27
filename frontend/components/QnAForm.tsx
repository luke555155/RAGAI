import { useState, useEffect } from 'react'
import { askQuestion } from '@/utils/api'
import { useDoc } from './DocContext'

interface Reference {
  text: string
  chunk_index?: number
  score?: number
  rank?: number
  filtered?: boolean
}

interface Answer {
  answer: string
  references: Reference[]
  elapsed?: number
}

interface QARecord {
  question: string
  answer: string
  references: Reference[]
  documentIds?: string[]
  elapsed?: number
}

export default function QnAForm() {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState<Answer | null>(null)
  const [displayedAnswer, setDisplayedAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<QARecord[]>([])
  const [style, setStyle] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const { selectedDocIds } = useDoc()

  function scoreColor(score?: number) {
    if (score === undefined) return ''
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.5) return 'text-yellow-600'
    return 'text-red-600'
  }

  function relevanceTag(ref: Reference) {
    if (ref.rank !== undefined) {
      return (
        <span className="ml-1 px-1 text-xs bg-purple-200 text-purple-800 rounded">
          Top {ref.rank}
        </span>
      )
    }
    if (ref.score !== undefined) {
      return (
        <span className="ml-1 px-1 text-xs bg-gray-200 text-gray-800 rounded">
          {ref.score.toFixed(2)}
        </span>
      )
    }
    return null
  }

  function animateAnswer(text: string) {
    setDisplayedAnswer('')
    let i = 0
    const timer = setInterval(() => {
      setDisplayedAnswer(prev => prev + text.charAt(i))
      i += 1
      if (i >= text.length) clearInterval(timer)
    }, 30)
  }

  useEffect(() => {
    const saved = localStorage.getItem('qaHistory')
    if (saved) {
      try {
        setHistory(JSON.parse(saved))
      } catch {}
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('qaHistory', JSON.stringify(history))
  }, [history])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!question) return
    setLoading(true)
    try {
      const data = await askQuestion(question, selectedDocIds, style)
      setResponse(data)
      setElapsed(data.elapsed || 0)
      animateAnswer(data.answer)
      setHistory([
        { question, answer: data.answer, references: data.references, documentIds: selectedDocIds, elapsed: data.elapsed },
        ...history,
      ])
    } catch (err) {
      const errRecord = {
        question,
        answer: 'Error fetching answer',
        references: [],
        documentIds: selectedDocIds,
      }
      setResponse({ answer: errRecord.answer, references: [] })
      setHistory([errRecord, ...history])
    } finally {
      setLoading(false)
    }
  }

  async function handleReask(q: string, docIds?: string[]) {
    setLoading(true)
    try {
      const data = await askQuestion(q, docIds || [], style)
      setResponse(data)
      setElapsed(data.elapsed || 0)
      animateAnswer(data.answer)
      setHistory([
        { question: q, answer: data.answer, references: data.references, documentIds: docIds, elapsed: data.elapsed },
        ...history,
      ])
    } catch {
      setResponse({ answer: 'Error fetching answer', references: [] })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          className="flex-1 border rounded p-2"
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
      {loading && (
        <div className="flex justify-center">
          <div className="w-6 h-6 border-4 border-gray-300 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      {response && (
        <div className="space-y-2 bg-gray-100 p-4 rounded">
          <div className="flex justify-between">
            <p className="font-semibold">Answer:</p>
            <button
              type="button"
              onClick={() => navigator.clipboard.writeText(response.answer)}
              className="text-sm text-blue-600 underline"
            >
              複製回答
            </button>
          </div>
          <p className="text-sm text-gray-600">Time: {elapsed.toFixed(2)}s</p>
          <p>{displayedAnswer}</p>
          {response.references.length > 0 && (
            <div>
              <p className="font-semibold mt-2">References:</p>
              <ul className="list-disc list-inside space-y-1">
                {response.references.map((ref, idx) => (
                  <li key={idx} className={scoreColor(ref.score)}>
                    [{ref.chunk_index}] {ref.text}
                    {relevanceTag(ref)}
                    {ref.filtered && (
                      <span className="ml-1 text-xs text-red-500">filtered</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {history.length > 0 && (
        <div className="space-y-2">
          <p className="font-semibold">回答記錄:</p>
          <ul className="space-y-1">
            {history.map((h, idx) => (
              <li key={idx} className="border p-2 rounded space-y-1">
                <div className="flex justify-between">
                  <p className="text-sm font-semibold">Q: {h.question}</p>
                  <div className="space-x-2">
                    <button
                      type="button"
                      onClick={() => handleReask(h.question, h.documentIds)}
                      className="text-sm text-green-700 underline"
                    >
                      重新查詢
                    </button>
                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(h.answer)}
                      className="text-sm text-blue-600 underline"
                    >
                      複製回答
                    </button>
                  </div>
                </div>
                {h.elapsed !== undefined && (
                  <p className="text-xs text-gray-600">Time: {h.elapsed.toFixed(2)}s</p>
                )}
                <p className="text-sm">A: {h.answer}</p>
                {h.references.length > 0 && (
                  <ul className="list-disc list-inside text-sm mt-1 space-y-0.5">
                    {h.references.map((ref, rIdx) => (
                      <li key={rIdx} className={scoreColor(ref.score)}>
                        [{ref.chunk_index}] {ref.text}
                        {relevanceTag(ref)}
                        {ref.filtered && (
                          <span className="ml-1 text-xs text-red-500">filtered</span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
