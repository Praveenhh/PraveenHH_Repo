# Example notes for Semantic Search & RAG

A retrieval-augmented generation system has several stages. First, documents are divided into chunks. The chunks and user query are converted into vector embeddings. A retrieval step ranks chunks by similarity to the query. A generator can then use the selected passages to draft an answer.

Evaluation should examine retrieval quality and answer quality separately. Retrieval measures can include recall at k and mean reciprocal rank. Answer review can check whether claims are supported by the retrieved passages. Similarity scores are ranking signals and are not calibrated confidence values.

A small local application can keep uploaded content in memory, but production systems need access controls, retention policies, durable storage, monitoring, and evaluation on representative data. Retrieved content should be treated as untrusted input because it may contain misleading text or instructions.
