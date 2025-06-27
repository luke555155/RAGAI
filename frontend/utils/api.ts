export async function uploadFile(file: File, tags: string) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('tags', tags);

  const res = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    throw new Error('Upload failed');
  }
  return res.json();
}

export async function askQuestion(question: string, documentIds: string[] = [], style?: string) {
  const body: Record<string, unknown> = { question }
  if (documentIds.length > 0) body.document_ids = documentIds
  if (style) body.style = style
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}

export async function fetchDocs() {
  const res = await fetch('/api/docs');
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}

export async function fetchDocumentSegments(id: string) {
  const res = await fetch(`/api/docs/${id}`);
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}

export async function deleteDoc(id: string) {
  const res = await fetch('/api/docs', {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ document_id: id }),
  });
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}

export async function resummarizeDoc(id: string) {
  const res = await fetch(`/api/docs/${id}/resummarize`, { method: 'POST' });
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}
