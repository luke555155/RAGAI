import UploadForm from '@/components/UploadForm'
import QnAForm from '@/components/QnAForm'
import DocumentsList from '@/components/DocumentsList'

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto p-4 space-y-8">
      <h1 className="text-2xl font-bold text-center">Document Q&A</h1>
      <div className="flex space-x-4 justify-center text-sm">
        <a href="/AnswerHistory" className="text-blue-600 underline">回答歷史</a>
        <a href="/manual-context" className="text-blue-600 underline">自選段落模式</a>
      </div>
      <UploadForm />
      <DocumentsList />
      <QnAForm />
    </div>
  )
}
