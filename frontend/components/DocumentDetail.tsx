import { useEffect, useState } from 'react'
import { fetchDocumentSegments } from '@/utils/api'

interface Segment {
  text: string
  chunk_index?: number
}

export default function DocumentDetail({ id }: { id: string }) {
  const [segments, setSegments] = useState<Segment[]>([])

  useEffect(() => {
    fetchDocumentSegments(id).then(data => setSegments(data.segments || []))
  }, [id])

  return (
    <div className="space-y-1">
      <h2 className="text-xl font-semibold mb-2">Segments</h2>
      <ol className="list-decimal list-inside space-y-1">
        {segments.map((s, idx) => (
          <li key={idx}>{s.text}</li>
        ))}
      </ol>
    </div>
  )
}
