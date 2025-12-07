export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-10 bg-gray-50">
      <h1 className="text-4xl font-bold mb-10">5D Regression Application</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-3xl">

        {/* Upload */}
        <a
          href="/upload"
          className="border rounded-lg p-6 bg-white shadow hover:shadow-lg transition cursor-pointer"
        >
          <h2 className="text-xl font-semibold mb-2">Upload Dataset →</h2>
          <p className="text-gray-600">
            Upload your .pkl file and preview its contents.
          </p>
        </a>

        {/* Train */}
        <a
          href="/train"
          className="border rounded-lg p-6 bg-white shadow hover:shadow-lg transition cursor-pointer"
        >
          <h2 className="text-xl font-semibold mb-2">Train Model →</h2>
          <p className="text-gray-600">
            Configure hyperparameters and train your model.
          </p>
        </a>

        {/* Predict */}
        <a
          href="/predict"
          className="border rounded-lg p-6 bg-white shadow hover:shadow-lg transition cursor-pointer"
        >
          <h2 className="text-xl font-semibold mb-2">Make Predictions →</h2>
          <p className="text-gray-600">
            Enter input vectors and generate predictions.
          </p>
        </a>

      </div>
    </main>
  );
}
