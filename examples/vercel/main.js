import { generateText } from "ai"
import { createOpenAI } from "@ai-sdk/openai"
const LANGDB_PROJECT_ID = "1641f1db-bab8-4687-8a0e-efecd95a5361"
const apiKey = "langdb_N01YN1M5c21hL0pFcXVSblZwNFJRbGNMblpGY2FkN2FJV0x3akJFeFpORHNZdTVUelAyaUEyYkdHWmd1ZWwrZWkzOEtPOWVERzFMZnJYKzJNbmZNUUdLK3ZxbnBGZG8wa0JpSUJjTXdUai9wVmpuUGRpa0VlVzFPcHFQZTh4VGtqUlVaUWwzL0hDNXJTcGpVSTQzRzhFajQxVFJ1WlcvNERWeWhMcTJ0KzVZZ0wzSW9hMm9YbGpaQnZ1bWFOM2JQU21mcjpBQUFBQUFBQUFBQUFBQUFB"
const baseURL = `https://api.us-east-1.langdb.ai/${LANGDB_PROJECT_ID}/v1`

const openai = createOpenAI({
  apiKey: apiKey,
  baseURL: baseURL
})

const { text } = await generateText({
  model: openai("gpt-4o-mini"),
  prompt: "What is Capital of France?"
})
console.log(text)