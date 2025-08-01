"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import TYPE_CHECKING

from .client import CrossEncoderClient

if TYPE_CHECKING:
    import voyageai
else:
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            'voyageai is required for VoyageRerankerClient. '
            'Install it with: pip install graphiti-core[voyageai]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'rerank-2.5'


class VoyageRerankerClient(CrossEncoderClient):
    """
    Voyage AI Reranker Client
    
    This reranker uses the Voyage AI API to rerank passages based on their
    relevance to a query. It supports Voyage's state-of-the-art reranking models
    including rerank-2.5, rerank-2.5-lite, rerank-2, rerank-2-lite, and rerank-1.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: 'voyageai.Client | None' = None,
    ):
        """
        Initialize the VoyageRerankerClient with the provided configuration and client.

        Args:
            api_key (str | None): The Voyage AI API key. If not provided, 
                                  the client will try to use the VOYAGE_API_KEY environment variable.
            model (str): The reranking model to use. 
                        Options: rerank-2.5, rerank-2.5-lite, rerank-2, rerank-2-lite, rerank-1.
                        Defaults to 'rerank-2.5'.
            client (voyageai.Client | None): An optional Voyage AI client instance to use. 
                                            If not provided, a new client is created.
        """
        self.api_key = api_key
        self.model = model
        
        if client is None:
            self.client = voyageai.Client(api_key=api_key)
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages based on their relevance to the query using Voyage AI reranking.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage and its relevance score,
                                     sorted in descending order of relevance.
        """
        if not passages:
            return []
        
        if len(passages) == 1:
            return [(passages[0], 1.0)]

        try:
            # Use Voyage AI rerank API
            reranking_result = self.client.rerank(
                query=query,
                documents=passages,
                model=self.model,
                top_k=None,  # Return all results
                truncation=True  # Allow truncation if needed
            )
            
            # Convert Voyage results to expected format
            ranked_passages = [
                (result.document, result.relevance_score)
                for result in reranking_result.results
            ]
            
            return ranked_passages
            
        except Exception as e:
            logger.error(f'Error in Voyage reranking: {e}')
            # Fallback: return passages with equal scores
            return [(passage, 1.0) for passage in passages]