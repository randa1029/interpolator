"use client";

import { useState } from "react";
import Link from "next/link";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [previewMeta, setPreviewMeta] = useState<any>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [uploaded, setUploaded] = useState(false);

  async function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] || null;

    setFile(selected);
    setPreviewMeta(null);
    setPreviewData(null);
    setUploaded(false);
    setMessage("");

    if (!selected) return;

    if (!selected.name.endsWith(".pkl")) {
      setMessage("❌ Only .pkl files are allowed.");
      setFile(null);
      return;
    }

    setPreviewMeta({
      name: selected.name,
      size: (selected.size / 1024).toFixed(2) + " KB",
      modified: new Date(selected.lastModified).toLocaleString(),
    });

    // Auto-upload when file is selected
    await handleUpload(selected);
  }

  async function handleUpload(selectedFile: File) {
    setMessage("Uploading...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        setMessage("❌ Upload failed: " + (data.detail || res.statusText));
        setUploaded(false);
        return;
      }

      setMessage("✅ Upload successful");
      setUploaded(true);
    } catch (error: any) {
      setMessage("❌ Error: " + (error.message || "Backend not responding. Is the server running on http://localhost:8000?"));
      setUploaded(false);
    }
  }

  async function handlePreview() {
    if (!file) return;

    const res = await fetch("http://localhost:8000/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name }),
    });

    const data = await res.json();
    setPreviewData(data);
  }

  return (
    <main className="p-10 max-w-3xl mx-auto flex flex-col gap-6">
      <h1 className="text-3xl font-bold">Upload Dataset</h1>

      {/* File Input */}
      <input
        type="file"
        accept=".pkl"
        onChange={handleFileSelect}
        className="border p-3"
      />

      {/* File info preview */}
      {previewMeta && (
        <div className="border rounded p-4 bg-gray-50">
          <h2 className="font-semibold mb-2">File Information</h2>
          <p><strong>Name:</strong> {previewMeta.name}</p>
          <p><strong>Size:</strong> {previewMeta.size}</p>
          <p><strong>Last Modified:</strong> {previewMeta.modified}</p>
        </div>
      )}

      {/* Status message */}
      {message && (
        <p className="p-3 bg-gray-100 border rounded font-semibold">{message}</p>
      )}

      {/* Actions AFTER successful upload */}
      {uploaded && file && (
        <div className="flex flex-col gap-3 mt-2">

          {/* 1. Preview pickle file button */}
          <button
            onClick={handlePreview}
            className="px-4 py-2 bg-purple-600 text-white rounded w-fit hover:bg-purple-700"
          >
            Preview Pickle File
          </button>

          {/* 2. Go to training page */}
          <Link
            href={`/train?filename=${encodeURIComponent(file.name)}`}
            className="px-4 py-2 bg-green-600 text-white rounded w-fit hover:bg-green-700"
          >
            Go to Training →
          </Link>

        </div>
      )}

      {/* Dataset preview output */}
      {previewData && (
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold">Dataset Preview</h2>

          <p><strong>Shape:</strong> {previewData.shape?.join(" × ")}</p>

          <h3 className="font-medium mt-3">First Rows:</h3>
          <pre className="bg-gray-100 p-3 rounded overflow-auto">
            {JSON.stringify(previewData.head, null, 2)}
          </pre>

          {previewData.y_preview && (
            <>
              <h3 className="font-medium mt-3">First y Values:</h3>
              <pre className="bg-gray-100 p-3 rounded overflow-auto">
                {JSON.stringify(previewData.y_preview, null, 2)}
              </pre>
            </>
          )}
        </div>
      )}
    </main>
  );
}
