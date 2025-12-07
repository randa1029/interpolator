"use client";

import { useState } from "react";
import Link from "next/link";

export default function PredictPage() {
  const [rows, setRows] = useState<string[][]>([["", "", "", "", ""]]);
  const [result, setResult] = useState<any>(null);
  const [message, setMessage] = useState("");

  // Update a single value inside 2D array
  function updateInput(rowIndex: number, colIndex: number, value: string) {
    const updated = [...rows];
    updated[rowIndex] = [...updated[rowIndex]];
    updated[rowIndex][colIndex] = value;
    setRows(updated);
  }

  // Add new row
  function addRow() {
    setRows([...rows, ["", "", "", "", ""]]);
  }

  // Remove a row
  function removeRow(index: number) {
    const updated = rows.filter((_, i) => i !== index);
    setRows(updated.length === 0 ? [["", "", "", "", ""]] : updated);
  }

  async function handlePredict() {
    setMessage("Predicting...");
    setResult(null);

    // Convert strings to numbers
    const numericRows = rows.map(row => row.map(val => Number(val) || 0));
    const payload = { inputs: numericRows };

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    setResult(data);
    setMessage("");
  }

  return (
    <main className="p-10 flex flex-col gap-6 max-w-3xl mx-auto">
      <div className="flex items-center gap-4">
        <Link href="/train" className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700">
          ← Go Back
        </Link>
        <h1 className="text-3xl font-bold">Make a Prediction</h1>
      </div>

      {/* Dynamic rows */}
      {rows.map((row, rowIndex) => (
        <div key={rowIndex} className="flex gap-2 items-center mb-2">
          {row.map((value, colIndex) => (
            <input
              key={colIndex}
              type="number"
              className="border p-2 w-24"
              placeholder={`x${colIndex + 1}`}
              value={value}
              onChange={(e) =>
                updateInput(rowIndex, colIndex, e.target.value)
              }
            />
          ))}

          <button
            onClick={() => removeRow(rowIndex)}
            className="px-2 py-1 bg-red-500 text-white rounded"
          >
            X
          </button>
        </div>
      ))}

      {/* Add Row Button */}
      <button
        onClick={addRow}
        className="px-4 py-2 bg-blue-600 text-white rounded w-fit"
      >
        + Add Input Row
      </button>

      {/* Predict Button */}
      <button
        onClick={handlePredict}
        className="px-6 py-2 bg-purple-600 text-white rounded mt-4"
      >
        Predict
      </button>

      {message && <p>{message}</p>}

      {/* Prediction Result */}
      {result && (
        <div className="border rounded p-4 bg-white shadow">
          <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
          
          <div className="space-y-3">
            {result.predictions && result.predictions.map((pred: number, idx: number) => (
              <div key={idx} className="flex items-center gap-3 p-3 bg-blue-50 rounded">
                <span className="font-semibold text-gray-700">Sample {idx + 1}:</span>
                <span className="text-2xl font-bold text-blue-600">{pred.toFixed(4)}</span>
              </div>
            ))}
          </div>
          
          <p className="mt-4 text-gray-600">
            Total predictions: <span className="font-semibold">{result.num_samples}</span>
          </p>
        </div>
      )}

      {/* Go to Home page - outside the results box */}
      {result && (
        <div className="flex justify-end">
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-700 text-white rounded hover:bg-slate-800"
          >
            <span className="text-xl">🏠</span>
            Back to Home
          </Link>
        </div>
      )}
    </main>
  );
}
