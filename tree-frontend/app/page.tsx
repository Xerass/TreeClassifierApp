// we need to set it to use clinet since this espects user interaction such as reachthooks and camera usage
'use client';

//usestate to save key values like prediction even after re-render
import { useState, ChangeEvent, useRef } from 'react';

export default function TreeScanner(){
  //why use Usestate? everytime we want to output a change (specifically when pred updates)
  //we can use the setter func from this to rerender the page
  //it operates with value and setter, pred will be our state value, where when the model finishes
  //it will store the name and conf
  //while setPrediciton will be the setter function, this is used to change the state.
  //must be const, array of pred and conf, default null
  const [prediction, setPrediction] = useState<{ prediction: string; confidence: string } | null>(null);
  //for inference loading, default to false
  const [loading, setLoading] = useState(false)

  //useref for mutable inputs without having to cause reRenders
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  //asynchronous capture so stuff can happen while taking a pic
  const handleCapture = async(e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    //if no file back out
    if (!file) return;

    //loading is now true, inference will begin
    setLoading(true);
    //data we will pass
    const formData = new FormData();
    //append the captured image to formData, name and value, must match with FastAPI soi it can understand it
    formData.append('file', file);

    //try clause so we capture any possible API failures
    try {
      //locally, for now we will use the default fastapi local address
      const response = await fetch('http://localhost:8000/predict', {
        //post since we are sending data
        method: 'POST',
        //data to send
        body: formData,
      });
      
      //if response comes back faulty, we throw an error
      if (!response.ok) throw new Error("Prediction Failed! sorry :(");

      //convert result of response into json args {pred name, conf}
      const data = await response.json()
      //reload the screen with result
      setPrediction(data);
    } catch (err) {
      alert("Error: FastAPI is probably not yet setup!");
    } finally {//after allat, setloading to false so our loading stops
      setLoading(false);
    }
  };

  //actual page formatting time (i hate html)
return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-slate-50">
      <div className="w-full max-w-md p-8 bg-white rounded-2xl shadow-xl text-center">
        <h1 className="text-3xl font-bold text-green-700 mb-2">TreeID</h1>
        <p className="text-slate-500 mb-8">Identify species via camera or upload</p>

        <div className="flex flex-col gap-4">
          {/* CAMERA BUTTON */}
          <button 
            onClick={() => cameraInputRef.current?.click()}
            disabled={loading}
            className="bg-green-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-green-700 transition-all shadow-md disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "📸 Use Camera"}
          </button>

          {/* FILE UPLOAD BUTTON */}
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="bg-white border-2 border-green-600 text-green-600 px-8 py-4 rounded-xl font-semibold hover:bg-green-50 transition-all shadow-sm disabled:opacity-50"
          >
            📁 Upload from Gallery
          </button>
        </div>

        {/* HIDDEN INPUTS */}
        <input 
          type="file" 
          accept="image/*" 
          capture="environment" 
          ref={cameraInputRef}
          className="hidden" 
          onChange={handleCapture}
        />
        <input 
          type="file" 
          accept="image/*" 
          ref={fileInputRef}
          className="hidden" 
          onChange={handleCapture}
        />

        {prediction && (
          <div className="mt-10 p-6 bg-green-50 border border-green-200 rounded-xl">
            <h2 className="text-sm uppercase tracking-widest text-green-600 font-bold">Result</h2>
            <p className="text-2xl font-black text-slate-800 my-1">{prediction.prediction}</p>
            <p className="text-green-700 font-medium">Confidence: {prediction.confidence}</p>
            <button 
              onClick={() => setPrediction(null)}
              className="mt-4 text-xs text-slate-400 underline"
            >
              Clear Result
            </button>
          </div>
        )}
      </div>
    </main>
  );
}