'use client'
import { useState, useCallback } from 'react'

type OCRResult = { original: string; editable: boolean; values: string[] }

export default function OCRPage() {
    const [file, setFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [result, setResult] = useState<OCRResult | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [editMode, setEditMode] = useState(false)
    const [editedValues, setEditedValues] = useState<string[]>([])

    const processFile = useCallback((f: File) => {
        if (!['image/jpeg', 'image/png', 'application/pdf'].includes(f.type)) {
            alert('Nieobsługiwany format pliku'); return
        }
        setFile(f)
        setPreview(URL.createObjectURL(f))
        setResult(null)
    }, [])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) processFile(e.target.files[0])
    }

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0])
    }

    const handleSubmit = async () => {
        if (!file) return
        setIsLoading(true)
        const formData = new FormData()
        formData.append('file', file)
        try {
            const token = localStorage.getItem('token')
            const response = await fetch('/api/ocr/upload', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            })
            const data = await response.json()
            setResult(data)
            setEditedValues([...data.values])
        } catch (err) {
            console.error('OCR error:', err)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="border-2 border-dashed border-indigo-300 dark:border-indigo-700 rounded-xl p-12 text-center cursor-pointer hover:border-indigo-500 transition-colors"
                onDrop={handleDrop} onDragOver={(e) => e.preventDefault()}>
                <input type="file" accept=".jpg,.jpeg,.png,.pdf" className="hidden"
                    id="dropzone-input" data-testid="dropzone-input" onChange={handleFileChange} />
                <label htmlFor="dropzone-input" className="cursor-pointer">
                    <p className="text-2xl mb-2">📂</p>
                    <p className="text-lg mb-2 dark:text-white">Przeciągnij plik tutaj</p>
                    <p className="text-gray-500 dark:text-gray-400 mb-4">lub kliknij aby wybrać</p>
                    <p className="text-sm text-gray-400">Obsługiwane formaty: JPG, PNG, PDF (max 10MB)</p>
                </label>
            </div>

            {preview && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                    <h2 className="text-xl font-bold mb-4 dark:text-white">Podgląd</h2>
                    <div className="flex justify-center">
                        {file?.type.startsWith('image/') ? (
                            <img src={preview} alt="Preview" className="max-h-64 rounded" data-testid="file-preview" />
                        ) : (
                            <div className="bg-gray-100 dark:bg-slate-700 p-8 rounded-lg text-center" data-testid="file-preview">
                                <p className="text-6xl">📄</p>
                                <p className="mt-2 dark:text-white">{file?.name}</p>
                            </div>
                        )}
                    </div>
                    <button onClick={handleSubmit} disabled={isLoading}
                        className="mt-4 bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50">
                        {isLoading ? 'Przetwarzanie...' : 'Prześlij do analizy'}
                    </button>
                </div>
            )}

            {result && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-bold dark:text-white">Wyniki OCR</h2>
                        <button onClick={() => setEditMode(!editMode)}
                            className="bg-gray-200 dark:bg-slate-700 text-sm py-1 px-3 rounded-lg dark:text-white">
                            {editMode ? 'Zapisz' : 'Edytuj wartości'}
                        </button>
                    </div>
                    <div className="space-y-2">
                        {editMode
                            ? editedValues.map((v, i) => (
                                <input key={i} value={v} onChange={(e) => {
                                    const n = [...editedValues]; n[i] = e.target.value; setEditedValues(n)
                                }} className="w-full p-2 border border-gray-300 rounded-lg dark:bg-slate-700 dark:text-white" />
                            ))
                            : result.values.map((v, i) => <p key={i} className="dark:text-white">{v}</p>)
                        }
                    </div>
                </div>
            )}
        </div>
    )
}
