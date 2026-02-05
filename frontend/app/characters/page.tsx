"use client";

import { useState, useRef } from "react";
import DrawingCanvas from "../components/DrawingCanvas";
import FileUpload from "../components/FileUpload";
import CharacterPredictionDisplay from "../components/CharacterPredictionDisplay";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CharacterPrediction {
    predicted_character: string;
    predicted_index: number;
    confidence: number;
    probabilities: number[];
    warning?: string | null;
}

export default function CharactersPage() {
    const [activeTab, setActiveTab] = useState<"draw" | "upload">("draw");
    const [prediction, setPrediction] = useState<CharacterPrediction | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const canvasRef = useRef<{ getImageData: () => string; clearCanvas: () => void } | null>(null);

    const handlePredict = async (imageData: string) => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}/predict/character/canvas`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageData }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Prediction failed");
            }

            const result = await response.json();
            setPrediction(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to connect to API");
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileUpload = async (file: File) => {
        setIsLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch(`${API_BASE_URL}/predict/character`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Prediction failed");
            }

            const result = await response.json();
            setPrediction(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to connect to API");
        } finally {
            setIsLoading(false);
        }
    };

    const handleCanvasSubmit = () => {
        if (canvasRef.current) {
            const imageData = canvasRef.current.getImageData();
            handlePredict(imageData);
        }
    };

    const handleClear = () => {
        if (canvasRef.current) {
            canvasRef.current.clearCanvas();
        }
        setPrediction(null);
        setError(null);
    };

    return (
        <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header className="py-6 px-4">
                <div className="max-w-6xl mx-auto text-center">
                    <h1 className="text-4xl md:text-5xl font-bold mb-2">
                        <span className="bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent">
                            Character Recognizer
                        </span>
                    </h1>
                    <p className="text-gray-400 text-lg">
                        AI-Powered Handwritten Letter Recognition (A-Z)
                    </p>
                    <a
                        href="/"
                        className="inline-block mt-3 text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
                    >
                        ← Back to Digit Recognition
                    </a>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 px-4 pb-8">
                <div className="max-w-6xl mx-auto">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                        {/* Input Section */}
                        <div className="glass-card p-6">
                            <div className="flex justify-center mb-6 border-b border-gray-700">
                                <button
                                    className={`tab-btn ${activeTab === "draw" ? "active" : ""}`}
                                    onClick={() => setActiveTab("draw")}
                                >
                                    ✏️ Draw
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
                                    onClick={() => setActiveTab("upload")}
                                >
                                    📁 Upload
                                </button>
                            </div>

                            {activeTab === "draw" ? (
                                <div className="space-y-4">
                                    <DrawingCanvas ref={canvasRef} />

                                    <div className="flex gap-4 justify-center">
                                        <button
                                            className="btn-primary bg-gradient-to-r from-emerald-500 to-teal-500"
                                            onClick={handleCanvasSubmit}
                                            disabled={isLoading}
                                        >
                                            {isLoading ? "Processing..." : "Predict"}
                                        </button>
                                        <button
                                            className="btn-secondary"
                                            onClick={handleClear}
                                        >
                                            Clear
                                        </button>
                                    </div>

                                    <p className="text-center text-gray-500 text-sm">
                                        Draw an uppercase letter (A-Z)
                                    </p>
                                </div>
                            ) : (
                                <FileUpload
                                    onFileSelect={handleFileUpload}
                                    isLoading={isLoading}
                                />
                            )}

                            {error && (
                                <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-center">
                                    {error}
                                </div>
                            )}
                        </div>

                        {/* Results Section */}
                        <div className="glass-card p-6">
                            <h2 className="text-xl font-semibold text-center mb-6 text-gray-300">
                                Prediction Results
                            </h2>

                            <CharacterPredictionDisplay
                                prediction={prediction}
                                isLoading={isLoading}
                            />
                        </div>
                    </div>

                    {/* Info Section */}
                    <div className="mt-8 glass-card p-6">
                        <h3 className="text-lg font-semibold text-emerald-300 mb-4">About Character Recognition</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-gray-400">
                            <div className="text-center">
                                <div className="text-3xl mb-2">📝</div>
                                <h4 className="font-medium text-white mb-1">EMNIST Dataset</h4>
                                <p className="text-sm">Trained on 88,800+ handwritten letter samples from EMNIST Letters</p>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl mb-2">🔤</div>
                                <h4 className="font-medium text-white mb-1">26 Classes</h4>
                                <p className="text-sm">Recognizes uppercase letters A through Z</p>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl mb-2">🎯</div>
                                <h4 className="font-medium text-white mb-1">High Accuracy</h4>
                                <p className="text-sm">CNN with data augmentation for robust predictions</p>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="py-4 text-center text-gray-500 text-sm">
                <p>Built with ❤️ using TensorFlow, FastAPI & Next.js</p>
            </footer>
        </div>
    );
}
