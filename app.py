from google import genai
from dotenv import dotenv_values

env = dotenv_values(".env")

client = genai.Client(api_key=env["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Are you there?",
)

print(response.text)
