package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

// Struktury do obsługi JSON API
type GenerateContentRequest struct {
	Contents []Content `json:"contents"`
}

type Content struct {
	Parts []Part `json:"parts"`
}

type Part struct {
	Text string `json:"text"`
}

type GenerateContentResponse struct {
	Candidates []Candidate `json:"candidates"`
}

type Candidate struct {
	Content struct {
		Parts []Part `json:"parts"`
	} `json:"content"`
}

func main() {
	// Wczytanie zmiennych środowiskowych z pliku .env
	err := godotenv.Load()
	if err != nil {
		fmt.Println("Błąd podczas wczytywania pliku .env:", err)
		return
	}

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		fmt.Println("Nie znaleziono klucza API Google w zmiennych środowiskowych")
		return
	}

	// Przygotowanie zapytania
	model := "gemini-2.0-flash"
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", model, apiKey)

	// Tworzenie zawartości zapytania
	requestBody := GenerateContentRequest{
		Contents: []Content{
			{
				Parts: []Part{
					{
						Text: "Are you there?",
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		fmt.Println("Błąd podczas serializacji JSON:", err)
		return
	}

	// Wykonanie zapytania HTTP
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Błąd podczas tworzenia zapytania:", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Błąd podczas wykonywania zapytania:", err)
		return
	}
	defer resp.Body.Close()

	// Przetwarzanie odpowiedzi
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Błąd podczas odczytu odpowiedzi:", err)
		return
	}

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Otrzymano błąd HTTP %d: %s\n", resp.StatusCode, string(respBody))
		return
	}

	var response GenerateContentResponse
	err = json.Unmarshal(respBody, &response)
	if err != nil {
		fmt.Println("Błąd podczas deserializacji JSON:", err)
		return
	}

	// Wyświetlenie odpowiedzi
	if len(response.Candidates) > 0 && len(response.Candidates[0].Content.Parts) > 0 {
		fmt.Println(response.Candidates[0].Content.Parts[0].Text)
	} else {
		fmt.Println("Brak odpowiedzi w oczekiwanym formacie")
	}
}