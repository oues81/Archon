from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from k.core.utils.utils import get_env_var

# Définition du modèle d'embedding et de ses dimensions
# Les modèles OpenAI ont des dimensions différentes:
# - text-embedding-3-small: 1536 dimensions
# - text-embedding-ada-002: 1536 dimensions
# - text-embedding-3-large: 3072 dimensions
# Les modèles Ollama ont généralement 768 dimensions (nomic-embed-text)
embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-ada-002'  # Utiliser ada-002 par défaut
embedding_dimensions = int(get_env_var('EMBEDDING_DIMENSIONS') or 1536)  # Lire depuis les variables d'environnement

# Dimensions de vecteur par défaut (selon le modèle)
EMBEDDING_DIMENSIONS = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
    'nomic-embed-text': 768,
}

# Obtenir les dimensions du vecteur selon le modèle
def get_embedding_dimension(model: str) -> int:
    """Retourne les dimensions du vecteur d'embedding pour un modèle donné."""
    # Priorité à la valeur explicite de la variable d'environnement
    if embedding_dimensions > 0:
        return embedding_dimensions
    return EMBEDDING_DIMENSIONS.get(model, 1536)  # Valeur par défaut: 1536

async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Utiliser la dimension correcte pour le modèle actuel
        dim = get_embedding_dimension(embedding_model)
        return [0] * dim  # Return zero vector with correct dimension on error

async def retrieve_relevant_documentation_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str) -> str:
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)
        
        # Log pour debug - permet de voir la dimension de l'embedding généré
        print(f"Dimension de l'embedding généré: {len(query_embedding)}")
        
        # Vérifier la dimension des embeddings dans la base de données
        try:
            # Tentative pour obtenir la dimension des vecteurs stockés
            sample = supabase.table('site_pages').select('embedding[0:1]').limit(1).execute()
            if sample.data and 'embedding' in sample.data[0]:
                print(f"Dimension des embeddings dans Supabase: {len(sample.data[0]['embedding'])}")
        except Exception as e:
            print(f"Impossible de vérifier la dimension des embeddings stockés: {e}")
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 4,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}" 

async def list_documentation_pages_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

async def get_page_content_tool(supabase: Client, url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together but limit the characters in case the page is massive (there are a coule big ones)
        # This will be improved later so if the page is too big RAG will be performed on the page itself
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

def get_file_content_tool(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server

    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        print(f"Error retrieving file contents: {e}")
        return f"Error retrieving file contents: {str(e)}"           
