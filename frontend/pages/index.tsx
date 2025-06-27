import UploadForm from '@/components/UploadForm'
import QnAForm from '@/components/QnAForm'
import DocumentsList from '@/components/DocumentsList'

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto p-4 space-y-8">
      <h1 className="text-2xl font-bold text-center">Document Q&A</h1>
      <UploadForm />
      <DocumentsList />
      <QnAForm />
    </div>
  )
}
