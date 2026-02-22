'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from '@/styles/Feature.module.css';

type OCRResult = { original: string; editable: boolean; values: string[] };

export default function OCRPage() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<OCRResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [editMode, setEditMode] = useState(false);
    const [editedValues, setEditedValues] = useState<string[]>([]);

    const processFile = useCallback((f: File) => {
        if (!['image/jpeg', 'image/png', 'application/pdf'].includes(f.type)) {
            alert('Nieobsługiwany format pliku. Wybierz JPG, PNG lub PDF.');
            return;
        }
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
    }, []);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) processFile(e.target.files[0]);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
    };

    const handleSubmit = async () => {
        if (!file) return;
        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('/api/ocr/upload', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            });
            const data = await response.json();
            setResult(data);
            setEditedValues([...data.values]);
        } catch (err) {
            console.error('OCR error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>Digitalizacja Dokumentów</h1>
                <p className={styles.subtitle}>Prześlij wyniki badań, aby AI automatycznie wyodrębniło kluczowe dane.</p>
            </header>

            <section
                className={`${styles.card} ${styles.glass} border-2 border-dashed border-primary/30 flex flex-col items-center justify-center py-20 text-center cursor-pointer hover:border-primary transition-all`}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
            >
                <input type="file" accept=".jpg,.jpeg,.png,.pdf" className="hidden"
                    id="dropzone-input" data-testid="dropzone-input" onChange={handleFileChange} />
                <label htmlFor="dropzone-input" className="cursor-pointer">
                    <div className="text-5xl mb-6">📄</div>
                    <h2 className={styles.sectionTitle}>Przeciągnij i upuść dokument</h2>
                    <p className="text-color-gray-500 mb-6">Lub kliknij, aby wybrać plik z dysku</p>
                    <p className="text-xs text-color-gray-400">WSPIERANE: JPG, PNG, PDF (DO 10MB)</p>
                </label>
            </section>

            <AnimatePresence>
                {preview && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={styles.card}
                    >
                        <h2 className={styles.sectionTitle}>Podgląd przesłanego pliku</h2>
                        <div className="flex flex-col items-center gap-6">
                            <div className="w-full max-w-lg bg-color-gray-50 dark:bg-color-gray-900 p-4 rounded-xl border border-border">
                                {file?.type.startsWith('image/') ? (
                                    <img src={preview} alt="Preview" className="w-full h-auto rounded-lg shadow-sm" data-testid="file-preview" />
                                ) : (
                                    <div className="flex flex-col items-center p-10" data-testid="file-preview">
                                        <span className="text-6xl mb-4">📑</span>
                                        <p className="font-bold">{file?.name}</p>
                                        <p className="text-xs text-color-gray-500 uppercase mt-1">Dokument PDF</p>
                                    </div>
                                )}
                            </div>
                            <button
                                onClick={handleSubmit}
                                disabled={isLoading}
                                className={`${styles.button} ${styles.primaryButton} w-full max-w-xs`}
                            >
                                {isLoading ? 'Przetwarzanie przez AI...' : 'Rozpocznij analizę OCR'}
                            </button>
                        </div>
                    </motion.div>
                )}

                {result && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`${styles.card} ${styles.glass}`}
                    >
                        <div className="flex justify-between items-center mb-6">
                            <h2 className={styles.sectionTitle}>Wyniki ekstrakcji danych</h2>
                            <button
                                onClick={() => setEditMode(!editMode)}
                                className={`${styles.button} ${styles.secondaryButton}`}
                            >
                                {editMode ? 'Zakończ edycję' : 'Edytuj dane'}
                            </button>
                        </div>

                        <div className="grid grid-cols-1 gap-3">
                            {editMode
                                ? editedValues.map((v, i) => (
                                    <div key={i}>
                                        <input
                                            id={`edited-value-${i}`}
                                            value={v}
                                            onChange={(e) => {
                                                const n = [...editedValues];
                                                n[i] = e.target.value;
                                                setEditedValues(n);
                                            }}
                                            className={styles.input}
                                        />
                                    </div>
                                ))
                                : result.values.map((v, i) => (
                                    <div key={i} className="p-3 bg-color-gray-50 dark:bg-color-gray-800 rounded-lg border border-border">
                                        <p className="text-sm">{v}</p>
                                    </div>
                                ))
                            }
                        </div>
                        {!editMode && (
                            <div className="mt-8 flex justify-end">
                                <button className={`${styles.button} ${styles.primaryButton}`}>
                                    Zatwierdź i zapisz w profilu
                                </button>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
