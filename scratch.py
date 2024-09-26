from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Issues no one are assigned to",
    model="text-embedding-3-large"
)

embed = response.data[0].embedding
print(embed)