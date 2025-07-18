import os
from typing import Any, Dict, List, Optional
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from .utils import get_env_var, write_to_log

class SupabaseGraphQLClient:
    def __init__(self):
        """Initialize the Supabase GraphQL client."""
        self.url = f"{get_env_var('SUPABASE_URL')}/graphql/v1"
        self.api_key = get_env_var('SUPABASE_ANON_KEY') or get_env_var('SUPABASE_SERVICE_KEY')
        
        if not self.api_key:
            error_msg = "Neither SUPABASE_ANON_KEY nor SUPABASE_SERVICE_KEY found in environment variables"
            write_to_log(error_msg)
            raise ValueError(error_msg)
            
        headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            transport = AIOHTTPTransport(url=self.url, headers=headers)
            self.client = Client(transport=transport, fetch_schema_from_transport=True)
            write_to_log("Successfully initialized Supabase GraphQL client")
        except Exception as e:
            error_msg = f"Failed to initialize GraphQL client: {str(e)}"
            write_to_log(error_msg)
            raise

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query.
        
        Args:
            query: The GraphQL query string
            variables: Optional dictionary of variables for the query
            
        Returns:
            Dict containing the query results
        """
        try:
            gql_query = gql(query)
            result = await self.client.execute_async(gql_query, variable_values=variables or {})
            write_to_log(f"Executed GraphQL query: {query[:100]}...")
            return result
        except Exception as e:
            error_msg = f"GraphQL query failed: {str(e)}\nQuery: {query}"
            if variables:
                # Ne pas logger les embeddings complets pour éviter de surcharger les logs
                safe_vars = {k: "[REDACTED]" if k == "query_embedding" else v 
                           for k, v in variables.items()}
                error_msg += f"\nVariables: {safe_vars}"
            write_to_log(error_msg)
            raise

    async def search_pages(
        self, 
        query_embedding: List[float], 
        match_threshold: float = 0.7, 
        match_count: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search pages using vector similarity.
        
        Args:
            query_embedding: The embedding vector to search with
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Maximum number of results to return
            filter: Optional dictionary of filters to apply
            
        Returns:
            List of matching pages with their details
        """
        query = """
        query SearchPages(
            $query_embedding: [Float!]!, 
            $match_threshold: Float!, 
            $match_count: Int!,
            $filter: jsonb
        ) {
            search_pages(
                args: {
                    query_embedding: $query_embedding,
                    match_threshold: $match_threshold,
                    match_count: $match_count,
                    filter: $filter
                }
            ) {
                id
                url
                chunk_number
                title
                summary
                content
                metadata
                similarity
            }
        }
        """
        
        variables = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter": filter or {}
        }
        
        result = await self.execute_query(query, variables)
        return result.get("search_pages", [])

    async def insert_page(
        self,
        url: str,
        title: str,
        content: str,
        chunk_number: int,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Insert a new page into the site_pages table.
        
        Args:
            url: The URL of the page
            title: The page title
            content: The page content
            chunk_number: The chunk number if the page is split
            summary: A summary of the page content
            embedding: The vector embedding of the content
            metadata: Optional metadata as a dictionary
            
        Returns:
            The inserted page data
        """
        mutation = """
        mutation InsertPage($object: site_pages_insert_input!) {
            insert_site_pages_one(object: $object) {
                id
                url
                title
                chunk_number
            }
        }
        """
        
        variables = {
            "object": {
                "url": url,
                "title": title,
                "content": content,
                "chunk_number": chunk_number,
                "summary": summary,
                "embedding": embedding,
                "metadata": metadata or {}
            }
        }
        
        result = await self.execute_query(mutation, variables)
        return result.get("insert_site_pages_one", {})

# Singleton instance
graphql_client = SupabaseGraphQLClient()
