from qdrant_client import QdrantClient

COLLECTION = "integration_kb"

client = QdrantClient(url="http://localhost:6333")
info = client.get_collection(COLLECTION)

print("Collection:", COLLECTION)
print("Vector size:", info.config.params.vectors.size)
print("Points count:", info.points_count)
