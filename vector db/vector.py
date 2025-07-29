import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

document_texts = [
    """
    Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
    is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
    in her free time in hopes of working at a tech company after graduating from the University of Washington.
    """,
    """
    The university chess club provides an outlet for students to come together and enjoy playing
    the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
    the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
    participate in tournaments, analyze famous chess matches, and improve members' skills.
    """,
    """
    The University of Washington, founded in 1861 in Seattle, is a public research university
    with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
    As the flagship institution of the six public universities in Washington state,
    UW encompasses over 500 buildings and 20 million square feet of space,
    including one of the largest library systems in the world.
    """
]

document_ids = ["student_info", "chess_club_info", "university_info"]

document_embeddings = embedding_model.encode(document_texts, convert_to_tensor=False)

document_embeddings = document_embeddings.astype('float32')

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

query_text = "what fun activities are there"

query_embedding = embedding_model.encode([query_text], convert_to_tensor=False)
query_embedding = query_embedding.astype('float32')

k = 2

distances, indices = index.search(query_embedding, k)

print("--- Query Results ---")
print(f"Query: '{query_text}'")
print(f"Found {k} nearest neighbors:")

for i in range(k):
    doc_index = indices[0][i]
    distance = distances[0][i]
    print(f"  Rank {i+1}:")
    print(f"    Document ID: {document_ids[doc_index]}")
    print(f"    Distance (L2): {distance:.4f}")
    print(f"    Content (first 100 chars): {document_texts[doc_index][:100]}...")
    print("-" * 20)


# import faiss
# import numpy as np

# # Create a dataset of 1000 random 128-dimensional vectors
# dimension = 128
# num_vectors = 1000
# data = np.random.random((num_vectors, dimension)).astype('float32')

# # Build a FAISS index for L2 distance
# index = faiss.IndexFlatL2(dimension)
# index.add(data) # Add vectors to the index

# # Query the index with a random vector
# query_vector = np.random.random((1, dimension)).astype('float32')
# distances, indices = index.search(query_vector, k=5) # Find 5 nearest neighbors

# print("Nearest neighbors:", indices)
# print("Distances:", distances)