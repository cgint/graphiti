from graphiti_core.embedder.openai import OpenAIEmbedder
from typing import List


class NonBatchingOllamaEmbedder(OpenAIEmbedder):
    """
    Custom Ollama embedder that processes embeddings one by one instead of in batches.
    This avoids the "cannot decode batches with this context" error in Ollama.
    """
    
    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Override the batch method to process items sequentially instead of as a batch.
        This prevents Ollama's batch processing issues.
        """
        results: List[List[float]] = []
        for input_data in input_data_list:
            # Process each item individually using the single create method
            result = await self.create(input_data)
            results.append(result)
        return results 