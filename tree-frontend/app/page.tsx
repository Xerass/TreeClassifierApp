// we need to set it to use clinet since this espects user interaction such as reachthooks and camera usage
'use client';

//usestate to save key values like prediction even after re-render
import { useState, ChangeEvent, useRef } from 'react';
import treeData from '../content/treeData.json';

// Type definition for the tree data to help TypeScript
interface TreeEntry {
  common: string;
  facts: string;
}
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
  //for our popup
  const [showModal, setShowModal] = useState(false);
  
  //useref for mutable inputs without having to cause reRenders
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  //asynchronous capture so stuff can happen while taking a pic
  const handleCapture = async(e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    //if no file back out
    if (!file) return;

    //loading is now true, inference will begin, reset stattes so no flickering
    setPrediction(null);
    setShowModal(false);
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
      setShowModal(true); //open popup when results arrive
    } catch (err) {
      alert("Error: FastAPI is probably not yet setup!");
    } finally {//after allat, setloading to false so our loading stops
      setLoading(false);
      //clear the input val so same file can be uploaded twice (if ever)
      e.target.value = "";
    }
  };

  // Helper to look up data from JSON safely
  const treeDetails: TreeEntry | null = prediction 
    ? (treeData as Record<string, TreeEntry>)[prediction.prediction] || { common: "Unknown Species", facts: "No additional data available." }
    : null;

  //actual page formatting time (i hate html)
return (
    <main className="relative flex min-h-screen flex-col items-center justify-center p-6 bg-slate-50">
      <div className="w-full max-w-md p-8 bg-white rounded-2xl shadow-xl text-center">
        <h1 className="text-3xl font-bold text-green-700 mb-2">TreeID</h1>
        <p className="text-slate-500 mb-8">Identify species via camera or upload</p>

        <div className="flex flex-col gap-4">
          <button 
            onClick={() => cameraInputRef.current?.click()}
            disabled={loading}
            className="bg-green-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-green-700 transition-all shadow-md disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "📸 Use Camera"}
          </button>

          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="bg-white border-2 border-green-600 text-green-600 px-8 py-4 rounded-xl font-semibold hover:bg-green-50 transition-all shadow-sm disabled:opacity-50"
          >
            📁 Upload from Gallery
          </button>
        </div>

        {/* Hidden inputs as before */}
        <input type="file" accept="image/*" capture="environment" ref={cameraInputRef} className="hidden" onChange={handleCapture} />
        <input type="file" accept="image/*" ref={fileInputRef} className="hidden" onChange={handleCapture} />
      </div>

      {/* POPUP MODAL */}
      {showModal && treeDetails && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6 z-50">
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl relative">
            {/* Visual Header */}
            <div className="flex flex-col items-center mb-6">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-green-600 text-2xl mb-4">
                🌳
              </div>
              <h2 className="text-xs uppercase tracking-[0.2em] text-green-600 font-bold">Identification Success</h2>
            </div>

            {/* Tree Info */}
            <div className="text-center mb-6">
              <h3 className="text-3xl font-black text-slate-800 leading-tight">
                {treeDetails.common}
              </h3>
              <p className="text-lg italic text-slate-500 font-medium mt-1">
                {prediction?.prediction}
              </p>
            </div>

            {/* Flavor Text Box */}
            <div className="bg-green-50 border border-green-100 p-5 rounded-2xl text-left mb-6">
              <p className="text-green-900 text-sm leading-relaxed">
                <span className="font-bold text-green-700 block mb-1">Did you know?</span> 
                {treeDetails.facts}
              </p>
            </div>

            {/* Accuracy Footer */}
            <div className="flex justify-between items-center px-2 py-3 border-t border-slate-100 text-xs font-semibold text-slate-400">
              <span>MODEL CONFIDENCE</span>
              <span className="text-green-600 font-mono bg-green-50 px-2 py-1 rounded">
                {prediction?.confidence}
              </span>
            </div>

            {/* Close Button */}
            <button 
              onClick={() => setShowModal(false)}
              className="mt-6 w-full bg-slate-900 text-white py-4 rounded-2xl font-bold hover:bg-black transition-colors"
            >
              Got it!
            </button>
          </div>
        </div>
      )}
    </main>
  );
}