from langchain.llms import Ollama
ollama = Ollama(base_url = "http://localhost:11434",
                model = "chatbot")

print(ollama("I am not feeling well having nausea and headache what should I do ? Reply in 30 words"))