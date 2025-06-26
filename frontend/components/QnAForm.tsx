import { useState } from 'react'
import { askQuestion } from '@/utils/api'

interface Answer {
  answer: string
  references: string[]
}

export default function QnAForm() {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState<Answer | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!question) return
    setLoading(true)
    try {
      const data = await askQuestion(question)
      setResponse(data)
    } catch (err) {
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
        <button
          type="submit"
          className="px-4 py-2 bg-green-600 text-white rounded"
          disabled={loading}
        >
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </form>
      {response && (
        <div className="space-y-2 bg-gray-100 p-4 rounded">
          <p className="font-semibold">Answer:</p>
          <p>{response.answer}</p>
          {response.references.length > 0 && (
            <div>
              <p className="font-semibold mt-2">References:</p>
              <ul className="list-disc list-inside space-y-1">
                {response.references.map((ref, idx) => (
                  <li key={idx}>{ref}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
