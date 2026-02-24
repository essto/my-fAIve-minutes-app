/* apps/web/src/app/(dashboard)/ocr/page.tsx */
'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shared/ui/Card/Card';
import { Button } from '@/components/shared/ui/Button/Button';
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
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
            >
                <Card className="border-2 border-dashed border-primary/30 flex flex-col items-center justify-center py-20 text-center cursor-pointer hover:border-primary transition-all bg-background/50 glass">
                    <input type="file" accept=".jpg,.jpeg,.png,.pdf" className="hidden"
                        id="dropzone-input" data-testid="dropzone-input" onChange={handleFileChange} />
                    <label htmlFor="dropzone-input" className="cursor-pointer">
                        <div className="text-5xl mb-6">📄</div>
                        <h2 className="text-xl font-bold mb-2 text-foreground">Przeciągnij i upuść dokument</h2>
                        <p className="text-color-gray-500 mb-6">Lub kliknij, aby wybrać plik z dysku</p>
                        <p className="text-xs text-color-gray-400 font-bold">WSPIERANE: JPG, PNG, PDF (DO 10MB)</p>
                    </label>
                </Card>
            </section>

            <AnimatePresence>
                {preview && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <Card>
                            <CardHeader>
                                <CardTitle>Podgląd przesłanego pliku</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="flex flex-col items-center gap-6 pt-4">
                                    <div className="w-full max-w-lg bg-color-gray-50 dark:bg-color-gray-900 p-4 rounded-xl border border-border">
                                        {file?.type.startsWith('image/') ? (
                                            <div className="relative w-full h-auto min-h-[300px]">
                                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                                <img src={preview} alt="Preview" className="w-full h-auto rounded-lg shadow-sm" data-testid="file-preview" />
                                            </div>
                                        ) : (
                                            <div className="flex flex-col items-center p-10" data-testid="file-preview">
                                                <span className="text-6xl mb-4">📑</span>
                                                <p className="font-bold">{file?.name}</p>
                                                <p className="text-xs text-color-gray-500 uppercase mt-1">Dokument PDF</p>
                                            </div>
                                        )}
                                    </div>
                                    <Button
                                        onClick={handleSubmit}
                                        isLoading={isLoading}
                                        className="w-full max-w-xs"
                                    >
                                        Rozpocznij analizę OCR
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>
                )}

                {result && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <Card glass gradientAccent>
                            <CardHeader className="flex flex-row justify-between items-center">
                                <CardTitle>Wyniki ekstrakcji danych</CardTitle>
                                <Button
                                    variant="outline"
                                    onClick={() => setEditMode(!editMode)}
                                >
                                    {editMode ? 'Zakończ edycję' : 'Edytuj dane'}
                                </Button>
                            </CardHeader>
                            <CardContent>
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
                                        <Button>
                                            Zatwierdź i zapisz w profilu
                                        </Button>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
