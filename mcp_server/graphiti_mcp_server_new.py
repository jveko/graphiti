#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()


DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X" or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


def create_azure_credential_token_provider() -> Callable[[], str]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    google_api_key: str | None = None
    llm_provider: str | None = None  # 'gemini', 'azure', or 'openai'

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        # Check for Google API key first (Gemini takes precedence)
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        if google_api_key:
            # Gemini takes precedence if API key is set
            return cls(
                google_api_key=google_api_key,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
                llm_provider='gemini',
            )

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        if azure_openai_endpoint is None:
            # Setup for OpenAI API
            # Log if empty model was provided
            if model_env == '':
                logger.debug(
                    f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
                )
            elif not model_env.strip():
                logger.warning(
                    f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
                )

            return cls(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
                llm_provider='openai',
            )
        else:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')

                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get('OPENAI_API_KEY', None)
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
                llm_provider='azure',
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                config.model = args.model
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance
        """

        if self.llm_provider == 'gemini' and self.google_api_key:
            # Gemini AI Studio setup
            llm_client_config = LLMConfig(
                api_key=self.google_api_key,
                model=self.model,
                small_model=self.small_model,
                temperature=self.temperature,
            )
            return GeminiClient(config=llm_client_config)
        elif self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                    ),
                )
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                    ),
                )
            else:
                raise ValueError('OPENAI_API_KEY must be set when using Azure OpenAI API')

        if not self.api_key:
            raise ValueError('OPENAI_API_KEY must be set when using OpenAI API')

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )

        # Set temperature
        llm_client_config.temperature = self.temperature

        return OpenAIClient(config=llm_client_config)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    voyage_api_key: str | None = None
    google_api_key: str | None = None
    embedder_provider: str | None = None  # 'voyage', 'gemini', 'azure', or 'openai'

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        # Check for Voyage API key first
        voyage_api_key = os.environ.get('VOYAGE_API_KEY', None)
        if voyage_api_key:
            # Voyage takes precedence if API key is set
            return cls(
                model=model,
                voyage_api_key=voyage_api_key,
                embedder_provider='voyage',
            )

        # Check for Google API key second (Gemini embeddings)
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        if google_api_key:
            # Gemini takes precedence over Azure/OpenAI if API key is set
            return cls(
                model=model,
                google_api_key=google_api_key,
                embedder_provider='gemini',
            )

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_EMBEDDING_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get(
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
        )
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )
        if azure_openai_endpoint is not None:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            azure_openai_deployment_name = os.environ.get(
                'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
            )
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set')

                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set'
                )

            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get('AZURE_OPENAI_EMBEDDING_API_KEY', None) or os.environ.get(
                    'OPENAI_API_KEY', None
                )
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                embedder_provider='azure',
            )
        else:
            return cls(
                model=model,
                api_key=os.environ.get('OPENAI_API_KEY'),
                embedder_provider='openai',
            )

    def create_client(self) -> EmbedderClient | None:
        if self.embedder_provider == 'voyage' and self.voyage_api_key:
            # Voyage AI setup
            voyage_config = VoyageAIEmbedderConfig(
                api_key=self.voyage_api_key,
                embedding_model=self.model,
            )
            return VoyageAIEmbedder(config=voyage_config)
        elif self.embedder_provider == 'gemini' and self.google_api_key:
            # Google Gemini embeddings setup
            gemini_config = GeminiEmbedderConfig(
                api_key=self.google_api_key,
                embedding_model=self.model,
            )
            return GeminiEmbedder(config=gemini_config)
        elif self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    model=self.model,
                )
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    model=self.model,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None
        else:
            # OpenAI API setup
            if not self.api_key:
                return None

            embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)

            return OpenAIEmbedder(config=embedder_config)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'  # Default to SSE transport
    stateless: bool = False  # For streamable-http stateless mode
    json_response: bool = False  # For JSON responses instead of SSE

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(
            transport=args.transport,
            stateless=getattr(args, 'stateless', False),
            json_response=getattr(args, 'json_response', False)
        )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Initialize Graphiti client
graphiti_client: Graphiti | None = None


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = config.embedder.create_client()

        # Check for cross-encoder (reranking) in priority order: Voyage → Gemini → OpenAI → None
        cross_encoder = None
        
        # 1. Try Voyage reranker first
        voyage_api_key = os.environ.get('VOYAGE_API_KEY')
        if voyage_api_key:
            try:
                from graphiti_core.cross_encoder.voyage_reranker_client import VoyageRerankerClient
                cross_encoder = VoyageRerankerClient(api_key=voyage_api_key)
                logger.info('Cross-encoder (reranking) enabled with Voyage AI')
            except Exception as e:
                logger.warning(f'Failed to initialize Voyage cross-encoder: {e}')
        
        # 2. Try Gemini reranker if no Voyage
        if cross_encoder is None:
            google_api_key = os.environ.get('GOOGLE_API_KEY')
            if google_api_key:
                try:
                    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
                    from graphiti_core.llm_client.config import LLMConfig
                    gemini_config = LLMConfig(api_key=google_api_key)
                    cross_encoder = GeminiRerankerClient(config=gemini_config)
                    logger.info('Cross-encoder (reranking) enabled with Gemini')
                except Exception as e:
                    logger.warning(f'Failed to initialize Gemini cross-encoder: {e}')
        
        # 3. Try OpenAI reranker as fallback
        if cross_encoder is None:
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if openai_api_key:
                try:
                    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
                    cross_encoder = OpenAIRerankerClient()
                    logger.info('Cross-encoder (reranking) enabled with OpenAI')
                except Exception as e:
                    logger.warning(f'Failed to initialize OpenAI cross-encoder: {e}')
        
        # 4. No reranker available
        if cross_encoder is None:
            logger.info('Cross-encoder (reranking) disabled - no API keys available for Voyage/Gemini/OpenAI')

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            cross_encoder=cross_encoder,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            logger.info(f'Using OpenAI model: {config.llm.model}')
            logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )
        logger.info(f'Using concurrency limit: {SEMAPHORE_LIMIT}')

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.
    """
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, episode_queues, queue_workers

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source.lower() == 'message':
            source_type = EpisodeType.message
        elif source.lower() == 'json':
            source_type = EpisodeType.json

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        # Cast group_id to str to satisfy type checker
        # The Graphiti client expects a str for group_id, not Optional[str]
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''

        # We've already checked that graphiti_client is not None above
        # This assert statement helps type checkers understand that graphiti_client is defined
        assert graphiti_client is not None, 'graphiti_client should not be None here'

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function
        async def process_episode():
            try:
                logger.info(f"Processing queued episode '{name}' for group_id: {group_id_str}")
                # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=group_id_str,  # Using the string version of group_id
                    uuid=uuid,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                )
                logger.info(f"Episode '{name}' added successfully")

                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{name}' for group_id {group_id_str}: {error_msg}"
                )

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()

        # Add the episode processing function to the queue
        await episode_queues[group_id_str].put(process_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries using advanced hybrid search.
    These contain a summary of all of a node's relationships with other nodes.
    
    Uses sophisticated search methodologies including BM25, cosine similarity, and adaptive 
    reranking strategies:
    - Cross-encoder reranking (when available) for highest quality semantic relevance
    - Node distance reranking when centering around a specific node
    - RRF (Reciprocal Rank Fusion) as a fast fallback for hybrid search
    
    The search automatically detects available reranking capabilities and selects the 
    optimal configuration for best results.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around (enables proximity-based reranking)
        entity: Optional entity type filter. Valid types: "Requirement", "Preference", "Procedure"
        
    Returns:
        NodeSearchResponse with matching nodes and their summaries, or ErrorResponse on failure
        
    Raises:
        ErrorResponse: If entity type is invalid or if Graphiti client is not initialized
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # Validate entity type if provided
        if entity is not None and entity not in ENTITY_TYPES:
            available_types = ', '.join(f'"{t}"' for t in ENTITY_TYPES)
            # Suggest similar entity types for possible typos
            suggestion = ""
            if entity.lower() in ['pref', 'preference', 'preferences']:
                suggestion = ' Did you mean "Preference"?'
            elif entity.lower() in ['proc', 'procedure', 'procedures']:
                suggestion = ' Did you mean "Procedure"?'
            elif entity.lower() in ['req', 'requirement', 'requirements']:
                suggestion = ' Did you mean "Requirement"?'
                
            return ErrorResponse(
                error=f'Invalid entity type "{entity}". Available types: {available_types}.{suggestion}'
            )
        
        # Handle entity filtering with graceful degradation
        entity_filter_warning = None
        if entity is not None and not config.use_custom_entities:
            # Instead of failing, fall back to unfiltered search with warning
            entity_filter_warning = (
                f'Entity filtering for "{entity}" skipped: custom entities not enabled. '
                f'Showing all nodes instead. To enable: restart server with --use-custom-entities flag.'
            )
            entity = None  # Disable entity filtering
            logger.warning(f'Graceful degradation: {entity_filter_warning}')

        filters = SearchFilters()
        if entity is not None:
            filters.node_labels = [entity]
            logger.debug(f'Applied entity filter: {entity}')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Configure the search with intelligent strategy selection
        # Check if cross-encoder is available by verifying if graphiti client has a cross_encoder
        has_cross_encoder = (
            hasattr(client, 'cross_encoder') and 
            client.cross_encoder is not None
        )
        
        # Smart search configuration selection based on context and available capabilities
        if center_node_uuid is not None:
            # Use node distance reranking when centering around a specific node for proximity-based results
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
            search_strategy = "node-distance (proximity-based)"
        elif has_cross_encoder:
            # Use cross-encoder for best semantic quality when available (slower but more accurate)
            search_config = NODE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
            search_strategy = "cross-encoder (high-quality semantic)"
        else:
            # Fallback to RRF for fast hybrid search when cross-encoder unavailable
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            search_strategy = "RRF hybrid (fast)"
        
        search_config.limit = max_nodes
        
        logger.debug(f'Search config: {search_strategy}, entity_filter: {entity}, groups: {effective_group_ids}')

        # Perform the search using the search_ method with timing
        import time
        start_time = time.time()
        
        search_results = await client.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )
        
        search_duration = time.time() - start_time
        logger.info(f'Search completed: {len(search_results.nodes)} nodes found in {search_duration:.2f}s using {search_strategy}')
        
        # Enhanced logging for debugging entity filtering
        if entity is not None and len(search_results.nodes) == 0:
            logger.warning(f'Entity filter "{entity}" returned no results. Suggestions:\n'
                          f'1. Use list_node_labels tool to check existing entity types\n'
                          f'2. Verify --use-custom-entities was enabled when adding data\n'
                          f'3. Try searching without entity filter first')
        elif entity is not None:
            entity_types_found = set()
            for node in search_results.nodes:
                entity_types_found.update([label for label in node.labels if label != 'Entity'])
            logger.debug(f'Entity types in results: {list(entity_types_found)}')
            
        if not search_results.nodes:
            suggestion_msg = "No relevant nodes found"
            if entity is not None:
                suggestion_msg += f" with entity type '{entity}'. Try using list_node_labels tool to debug."
            return NodeSearchResponse(message=suggestion_msg, nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        # Prepare response message with any warnings
        response_message = 'Nodes retrieved successfully'
        if entity_filter_warning:
            response_message = f'⚠️ {entity_filter_warning} Found {len(formatted_nodes)} nodes.'
        elif len(formatted_nodes) > 0:
            response_message = f'Found {len(formatted_nodes)} nodes'
            
        return NodeSearchResponse(message=response_message, nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error='Group ID must be a string')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def list_node_labels(
    group_id: str | None = None,
    entity_type: str | None = None,
    limit: int = 50
) -> dict[str, Any] | ErrorResponse:
    """List existing nodes with their labels for debugging entity filtering.
    
    This tool helps diagnose why entity filtering might not be working by showing
    what labels are actually stored in the database, with optional filtering and statistics.
    
    Args:
        group_id: Optional group ID to filter results. If not provided, uses default group_id.
        entity_type: Optional entity type to filter by (e.g., "Requirement", "Preference", "Procedure")
        limit: Maximum number of nodes to return (default: 50, max: 200)
        
    Returns:
        Dictionary with node information, entity type counts, and metadata
    """
    global graphiti_client
    
    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')
    
    try:
        # Validate inputs
        if limit > 200:
            limit = 200
        if limit < 1:
            limit = 50
            
        # Validate entity_type if provided
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            available_types = ', '.join(f'"{t}"' for t in ENTITY_TYPES)
            return ErrorResponse(
                error=f'Invalid entity_type "{entity_type}". Available types: {available_types}'
            )
        
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id
        
        if not isinstance(effective_group_id, str):
            return ErrorResponse(error='Group ID must be a string')
        
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None
        
        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)
        
        # Get filtered nodes - handle entity_type filtering via WHERE clause to avoid f-string
        if entity_type is not None:
            records, _, _ = await client.driver.execute_query(
                """
                MATCH (n:Entity {group_id: $group_id})
                WHERE $entity_type IN labels(n)
                RETURN n.uuid as uuid, n.name as name, labels(n) as labels, 
                       n.created_at as created_at, n.summary as summary
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                group_id=effective_group_id,
                entity_type=entity_type,
                limit=limit,
                routing_='r'
            )
        else:
            records, _, _ = await client.driver.execute_query(
                """
                MATCH (n:Entity {group_id: $group_id})
                RETURN n.uuid as uuid, n.name as name, labels(n) as labels, 
                       n.created_at as created_at, n.summary as summary
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                group_id=effective_group_id,
                limit=limit,
                routing_='r'
            )
        
        # Get entity type counts
        count_records, _, _ = await client.driver.execute_query(
            """
            MATCH (n:Entity {group_id: $group_id})
            WITH n, [label IN labels(n) WHERE label <> 'Entity'] as entity_labels
            UNWIND CASE WHEN size(entity_labels) = 0 THEN ['Untyped'] ELSE entity_labels END as entity_type
            RETURN entity_type, count(*) as count
            ORDER BY count DESC
            """,
            group_id=effective_group_id,
            routing_='r'
        )
        
        entity_counts = {}
        total_nodes = 0
        for record in count_records:
            entity_type_name = record['entity_type']
            count = record['count']
            entity_counts[entity_type_name] = count
            total_nodes += count
        
        nodes_info = []
        for record in records:
            # Extract entity types (exclude 'Entity' base label)
            entity_labels = [label for label in record['labels'] if label != 'Entity']
            nodes_info.append({
                'uuid': record['uuid'],
                'name': record['name'],
                'labels': record['labels'],
                'entity_types': entity_labels,
                'created_at': record['created_at'].isoformat() if record['created_at'] else None,
                'summary': record['summary'][:100] + '...' if record['summary'] and len(record['summary']) > 100 else record['summary']
            })
        
        filter_msg = f" matching entity type '{entity_type}'" if entity_type else ""
        
        return {
            'message': f'Found {len(nodes_info)} nodes{filter_msg} in group {effective_group_id} (showing up to {limit})',
            'nodes': nodes_info,
            'entity_type_counts': entity_counts,
            'total_nodes_in_group': total_nodes,
            'custom_entities_enabled': config.use_custom_entities,
            'filter_applied': entity_type
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error listing node labels: {error_msg}')
        return ErrorResponse(error=f'Error listing node labels: {error_msg}')


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return StatusResponse(status='error', message='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status='ok', message='Graphiti MCP server is running and connected to Neo4j'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        )


def configure_mcp_server(mcp_config: MCPConfig) -> None:
    """Configure the FastMCP server with the given configuration."""
    global mcp
    
    # Configure for streamable-http if needed
    if mcp_config.transport == 'streamable-http':
        # Note: These attributes may not be available in all FastMCP versions
        # mcp.stateless_http = mcp_config.stateless
        # mcp.json_response = mcp_config.json_response
        pass


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, mcp

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. This is an arbitrary string used to organize related data. '
        'If not provided, a random UUID will be generated.',
    )
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'streamable-http'],
        default='sse',
        help='Transport to use for communication with the client. (default: sse)',
    )
    parser.add_argument(
        '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--small-model',
        help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST', 'localhost'),
        help='Host to bind the MCP server to (default: localhost or MCP_SERVER_HOST env var)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('MCP_SERVER_PORT', '3000')),
        help='Port to bind the MCP server to (default: 3000 or MCP_SERVER_PORT env var)',
    )
    parser.add_argument(
        '--stateless',
        action='store_true',
        help='Enable stateless mode for streamable-http transport (no session persistence)',
    )
    parser.add_argument(
        '--json-response',
        action='store_true',
        help='Return JSON responses instead of SSE streams for streamable-http transport',
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    # Create MCP configuration
    mcp_config = MCPConfig.from_cli(args)
    
    # Configure MCP server with settings
    configure_mcp_server(mcp_config)
    
    # Initialize Graphiti
    await initialize_graphiti()

    # Configure host and port
    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        mcp.settings.host = args.host
    
    if args.port:
        logger.info(f'Setting MCP server port to: {args.port}')
        mcp.settings.port = args.port

    # Return MCP configuration
    return mcp_config


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with stdio transport for MCP in the same event loop
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_sse_async()
    elif mcp_config.transport == 'streamable-http':
        logger.info(
            f'Running MCP server with Streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_streamable_http_async()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
