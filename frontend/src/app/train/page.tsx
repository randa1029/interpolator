"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

function TrainPageContent() {
  const searchParams = useSearchParams();
  const [filename, setFilename] = useState("");
  const [numLayers, setNumLayers] = useState(3);
  const [neurons, setNeurons] = useState<(number | string)[]>([64, 32, 16]);
  const [lr, setLr] = useState<number | string>(0.001);
  const [maxIt, setMaxIt] = useState<number | string>(200);

  const [result, setResult] = useState<any>(null);
  const [message, setMessage] = useState("");

  // Get filename from URL params
  useEffect(() => {
    const filenameParam = searchParams.get("filename");
    if (filenameParam) {
      setFilename(filenameParam);
    }
  }, [searchParams]);

  // When number of layers changes → adjust neurons array
  function updateLayerCount(value: string) {
    const n = Number(value) || 1;
    const updated = [...neurons];

    if (n > updated.length) {
      while (updated.length < n) updated.push(64);
    } else {
      updated.length = n;
    }
    setNeurons(updated);
    setNumLayers(n);
  }

  // Update neurons for a given layer
  function updateNeuronCount(i: number, value: string) {
    const updated = [...neurons];
    updated[i] = value;
    setNeurons(updated);
  }

  async function startTraining() {
    if (!filename || !filename.trim()) {
      setMessage("❌ No dataset file specified. Please upload a file first.");
      return;
    }

    setMessage("Training model...");
    setResult(null);

    try {
      const payload = {
        filename: filename.trim(),
        hidden_layers: neurons.map(n => Number(n) || 0),
        lr: Number(lr) || 0.001,
        max_it: Number(maxIt) || 200,
        batch_size: 64,
        test_size: 0.2,
        val_size: 0.1,
      };

      const res = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      
      if (!res.ok) {
        setMessage("❌ Training failed: " + (data.detail || res.statusText));
        setResult(null);
        return;
      }

      setResult(data);
      setMessage("✅ Training completed successfully!");
    } catch (error: any) {
      setMessage("❌ Error: " + error.message);
      setResult(null);
    }
  }

  return (
    <main className="p-10 flex flex-col gap-6 max-w-2xl mx-auto">
      <div className="flex items-center gap-4">
        <Link href="/upload" className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700">
          ← Go Back
        </Link>
        <h1 className="text-3xl font-bold">Train Model</h1>
      </div>

      {/* Filename (read-only, from upload) */}
      {filename && (
        <div className="border rounded p-4 bg-gray-50">
          <p className="font-semibold">Training Dataset:</p>
          <p className="text-lg">{filename}</p>
        </div>
      )}

      {/* Number of Layers */}
      <div>
        <label className="font-semibold">Number of Layers</label>
        <input
          type="number"
          min={1}
          className="border p-2 w-full mt-1"
          value={numLayers}
          onChange={(e) => updateLayerCount(e.target.value)}
        />
      </div>

      {/* Neurons per Layer */}
      <div>
        <label className="font-semibold">Neurons per Layer</label>
        <div className="flex flex-col gap-2 mt-2">
          {neurons.map((n, i) => (
            <input
              key={i}
              type="number"
              min={1}
              className="border p-2"
              value={n}
              onChange={(e) => updateNeuronCount(i, e.target.value)}
              placeholder={`Layer ${i + 1}`}
            />
          ))}
        </div>
      </div>

      {/* Learning Rate */}
      <div>
        <label className="font-semibold">Learning Rate</label>
        <input
          type="number"
          step="0.0001"
          className="border p-2 w-full mt-1"
          value={lr}
          onChange={(e) => setLr(e.target.value)}
        />
      </div>

      {/* Max Iterations */}
      <div>
        <label className="font-semibold">Max Iterations</label>
        <input
          type="number"
          className="border p-2 w-full mt-1"
          value={maxIt}
          onChange={(e) => setMaxIt(e.target.value)}
        />
      </div>

      <button
        onClick={startTraining}
        className="px-6 py-2 bg-green-600 text-white rounded mt-2"
      >
        Start Training
      </button>

      {message && <p>{message}</p>}

      {result && (
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-4 text-green-600">✓ Training Completed Successfully!</h2>
          
          {/* Model Configuration */}
          {result.config && (
            <div className="mb-4">
              <h3 className="font-semibold text-lg mb-2">Model Configuration</h3>
              <div className="grid grid-cols-2 gap-3 bg-gray-50 p-3 rounded">
                <div>
                  <span className="text-gray-600">Input Size:</span>
                  <span className="ml-2 font-semibold">{result.config.input_size}</span>
                </div>
                <div>
                  <span className="text-gray-600">Hidden Layers:</span>
                  <span className="ml-2 font-semibold">{result.config.hidden_layers.join(", ")}</span>
                </div>
                <div>
                  <span className="text-gray-600">Learning Rate:</span>
                  <span className="ml-2 font-semibold">{result.config.learning_rate}</span>
                </div>
                <div>
                  <span className="text-gray-600">Max Iterations:</span>
                  <span className="ml-2 font-semibold">{result.config.max_iterations}</span>
                </div>
                <div>
                  <span className="text-gray-600">Batch Size:</span>
                  <span className="ml-2 font-semibold">{result.config.batch_size}</span>
                </div>
              </div>
            </div>
          )}
          
          {/* Data Split */}
          {result.data_split && (
            <div className="mb-4">
              <h3 className="font-semibold text-lg mb-2">Data Split</h3>
              <div className="flex gap-4 bg-blue-50 p-3 rounded">
                <div className="flex-1 text-center">
                  <p className="text-gray-600 text-sm">Training Samples</p>
                  <p className="text-2xl font-bold text-blue-600">{result.data_split.train_samples}</p>
                </div>
                <div className="flex-1 text-center">
                  <p className="text-gray-600 text-sm">Test Samples</p>
                  <p className="text-2xl font-bold text-green-600">{result.data_split.test_samples}</p>
                </div>
                <div className="flex-1 text-center">
                  <p className="text-gray-600 text-sm">Validation Samples</p>
                  <p className="text-2xl font-bold text-purple-600">{result.data_split.val_samples}</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Go to Prediction page after successful training */}
          <Link
            href="/predict"
            className="inline-block mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Go to Prediction →
          </Link>
        </div>
      )}
    </main>
  );
}

export default function TrainPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <TrainPageContent />
    </Suspense>
  );
}
