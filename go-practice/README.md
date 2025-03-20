# Go Practice - Gemini API Client

This is a simple Go application that demonstrates how to interact with Google's Gemini API.

## Prerequisites

- Go installed on your system
- Google API key for Gemini

## Setup

1. Clone this repository or create a new directory for your project.

2. Create a `.env` file in the project root to store your API key:

```bash
touch .env
echo "GOOGLE_API_KEY=<YOUR-API-KEY>" > .env
```

Replace `<YOUR-API-KEY>` with your actual Google Gemini API key.

3. Initialize Go module and install dependencies:

```bash
go mod init go-practice
go get github.com/joho/godotenv
```

## Build and Run

Build the application:

```bash
go build
```

Run the application:

```bash
./go-practice
```

Or run directly without building:

```bash
go run main.go
```

## What This Does

This application:
- Loads your Google API key from the `.env` file
- Sends a request to the Gemini 2.0 Flash model with the prompt "Are you there?"
- Displays the response from the model

## Code Structure

The main components of the application include:
- Environment variable loading with godotenv
- HTTP request creation and handling
- JSON serialization/deserialization for the Gemini API
- Error handling for various failure scenarios

## Troubleshooting

If you encounter errors:
- Ensure your API key is correct
- Check that your `.env` file is in the correct location
- Verify you have an active internet connection
- Make sure the Gemini API is available and your key has proper permissions