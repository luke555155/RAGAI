import { useState } from 'react'
import { uploadFile } from '@/utils/api'

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null)
  const [message, setMessage] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    try {
      await uploadFile(file)
      setMessage('Upload succeeded')
      setFile(null)
    } catch (err) {
      setMessage('Upload failed')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2" encType="multipart/form-data">
      <input
        type="file"
        accept="application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        onChange={e => setFile(e.target.files?.[0] || null)}
        className="block w-full text-sm"
      />
      <button type="submit" className="px-4 py-2 bg-blue-500 text-white rounded">
        Upload
      </button>
      {message && <p className="text-sm text-gray-600">{message}</p>}
    </form>
  )
}
