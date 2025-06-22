#!/usr/bin/env python3
"""
Ollama-based Cross-Encoder Reranker Client
Uses Ollama API for reranking instead of local HuggingFace models
"""

import asyncio
import logging
from typing import Any
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None

from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)

class OllamaRerankerClient(CrossEncoderClient):
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "bge-m3"):
        """
        Initialize Ollama reranker client.
        
        Args:
            base_url: Ollama API base URL
            model_name: BGE embedding model for similarity-based reranking
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages using BGE embeddings and cosine similarity.
        Uses BGE-M3 model for better multilingual performance.
        """
        if not passages:
            return []
            
        try:
            # Get embedding for the query
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return [(passage, 0.5) for passage in passages]
            
            # Get embeddings for all passages
            passage_embeddings = await asyncio.gather(
                *[self._get_embedding(passage) for passage in passages]
            )
            
            # Calculate cosine similarities
            scores = []
            for i, passage_embedding in enumerate(passage_embeddings):
                if passage_embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, passage_embedding)
                    scores.append((passages[i], float(similarity)))
                else:
                    scores.append((passages[i], 0.0))
            
            # Sort by score descending
            ranked_passages = sorted(scores, key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Ranked {len(passages)} passages using BGE embeddings")
            return ranked_passages
            
        except Exception as e:
            logger.error(f"Error ranking passages with BGE: {e}")
            # Fallback: return passages in original order with neutral scores
            return [(passage, 0.5) for passage in passages]
    
    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using BGE model via Ollama API."""
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        if aiohttp is None:
            logger.error("aiohttp not available, cannot make API calls")
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("embedding")
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Failed to get BGE embedding from Ollama: {e}")
            return None
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2) 