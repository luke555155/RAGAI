import { useRouter } from 'next/router'
import DocumentDetail from '@/components/DocumentDetail'
import Link from 'next/link'

export default function DocPage() {
  const router = useRouter()
  const { id } = router.query

  if (!id || Array.isArray(id)) return null

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      <Link href="/" className="text-blue-600 underline">
        返回
      </Link>
      <h1 className="text-2xl font-bold">Document {id}</h1>
      <DocumentDetail id={id} />
    </div>
  )
}
