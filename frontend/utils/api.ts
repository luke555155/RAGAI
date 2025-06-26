export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    throw new Error('Upload failed');
  }
  return res.json();
}

export async function askQuestion(question: string) {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
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
