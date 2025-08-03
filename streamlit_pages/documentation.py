import streamlit as st
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.crawl_pydantic_ai_docs import start_crawl_with_requests, clear_existing_records
from archon.crawl_mcp_docs import start_crawl_with_requests as start_mcp_crawl, clear_existing_records as clear_mcp_records
from utils.utils import get_env_var, create_new_tab_button

def documentation_tab(supabase_client):
    """Display the documentation interface"""
    st.header("Documentation")
    
    # Create tabs for different documentation sources
    doc_tabs = st.tabs(["Pydantic AI Docs", "MCP Documentation", "Future Sources"])
    
    with doc_tabs[0]:
        st.subheader("Pydantic AI Documentation")
        st.markdown("""
        This section allows you to crawl and index the Pydantic AI documentation.
        The crawler will:
        
        1. Fetch URLs from the Pydantic AI sitemap
        2. Crawl each page and extract content
        3. Split content into chunks
        4. Generate embeddings for each chunk
        5. Store the chunks in the Supabase database
        
        This process may take several minutes depending on the number of pages.
        """)
        
        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
            create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_docs")
        else:
            # Initialize session state for tracking crawl progress
            if "crawl_tracker" not in st.session_state:
                st.session_state.crawl_tracker = None
            
            if "crawl_status" not in st.session_state:
                st.session_state.crawl_status = None
                
            if "last_update_time" not in st.session_state:
                st.session_state.last_update_time = time.time()
            
            # Create columns for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to start crawling
                if st.button("Crawl Pydantic AI Docs", key="crawl_pydantic") and not (st.session_state.crawl_tracker and st.session_state.crawl_tracker.is_running):
                    try:
                        # Define a callback function to update the session state
                        def update_progress(status):
                            st.session_state.crawl_status = status
                        
                        # Start the crawling process in a separate thread
                        st.session_state.crawl_tracker = start_crawl_with_requests(update_progress)
                        st.session_state.crawl_status = st.session_state.crawl_tracker.get_status()
                        
                        # Force a rerun to start showing progress
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error starting crawl: {str(e)}")
            
            with col2:
                # Button to clear existing Pydantic AI docs
                if st.button("Clear Pydantic AI Docs", key="clear_pydantic"):
                    with st.spinner("Clearing existing Pydantic AI docs..."):
                        try:
                            # Run the function to clear records
                            clear_existing_records()
                            st.success("✅ Successfully cleared existing Pydantic AI docs from the database.")
                            
                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error clearing Pydantic AI docs: {str(e)}")
            
            # Display crawling progress if a crawl is in progress or has completed
            if st.session_state.crawl_tracker:
                # Create a container for the progress information
                progress_container = st.container()
                
                with progress_container:
                    # Get the latest status
                    current_time = time.time()
                    # Update status every second
                    if current_time - st.session_state.last_update_time >= 1:
                        st.session_state.crawl_status = st.session_state.crawl_tracker.get_status()
                        st.session_state.last_update_time = current_time
                    
                    status = st.session_state.crawl_status
                    
                    # Display a progress bar
                    if status and status["urls_found"] > 0:
                        progress = status["urls_processed"] / status["urls_found"]
                        st.progress(progress)
                    
                    # Display status metrics
                    col1, col2, col3, col4 = st.columns(4)
                    if status:
                        col1.metric("URLs Found", status["urls_found"])
                        col2.metric("URLs Processed", status["urls_processed"])
                        col3.metric("Successful", status["urls_succeeded"])
                        col4.metric("Failed", status["urls_failed"])
                    else:
                        col1.metric("URLs Found", 0)
                        col2.metric("URLs Processed", 0)
                        col3.metric("Successful", 0)
                        col4.metric("Failed", 0)
                    
                    # Display logs in an expander
                    with st.expander("Crawling Logs", expanded=True):
                        if status and "logs" in status:
                            logs_text = "\n".join(status["logs"][-20:])  # Show last 20 logs
                            st.code(logs_text)
                        else:
                            st.code("No logs available yet...")
                    
                    # Show completion message
                    if status and not status["is_running"] and status["end_time"]:
                        if status["urls_failed"] == 0:
                            st.success("✅ Crawling process completed successfully!")
                        else:
                            st.warning(f"⚠️ Crawling process completed with {status['urls_failed']} failed URLs.")
                
                # Auto-refresh while crawling is in progress
                if not status or status["is_running"]:
                    st.rerun()
        
        # Display database statistics
        st.subheader("Database Statistics")
        try:            
            # Query the count of Pydantic AI docs
            result = supabase_client.table("site_pages").select("count", count="exact").eq("metadata->>source", "pydantic_ai_docs").execute()
            count = result.count if hasattr(result, "count") else 0
            
            # Display the count
            st.metric("Pydantic AI Docs Chunks", count)
            
            # Add a button to view the data
            if count > 0 and st.button("View Indexed Data", key="view_pydantic_data"):
                # Query a sample of the data
                sample_data = supabase_client.table("site_pages").select("url,title,summary,chunk_number").eq("metadata->>source", "pydantic_ai_docs").limit(10).execute()
                
                # Display the sample data
                st.dataframe(sample_data.data)
                st.info("Showing up to 10 sample records. The database contains more records.")
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
    
    with doc_tabs[1]:
        st.subheader("MCP Documentation")
        st.markdown("""
        Cette section vous permet de crawler et d'indexer la documentation du Model Context Protocol (MCP).
        
        Le crawler va :
        1. Récupérer les URLs importantes de la documentation MCP sur GitHub
        2. Crawler chaque page et extraire le contenu
        3. Diviser le contenu en chunks
        4. Générer des embeddings pour chaque chunk
        5. Stocker les chunks dans la base de données Supabase
        
        Ce processus peut prendre quelques minutes selon le nombre de pages.
        """)
        
        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase n'est pas configuré. Veuillez d'abord configurer vos variables d'environnement.")
            create_new_tab_button("Aller à la section Environnement", "Environment", key="goto_env_from_mcp_docs")
        else:
            # Initialize session state for tracking crawl progress
            if "mcp_crawl_tracker" not in st.session_state:
                st.session_state.mcp_crawl_tracker = None
            
            if "mcp_crawl_status" not in st.session_state:
                st.session_state.mcp_crawl_status = None
                
            if "mcp_last_update_time" not in st.session_state:
                st.session_state.mcp_last_update_time = time.time()
            
            # Create columns for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to start crawling
                if st.button("Crawler la documentation MCP", key="crawl_mcp") and not (st.session_state.mcp_crawl_tracker and st.session_state.mcp_crawl_tracker.is_running):
                    try:
                        # Define a callback function to update the session state
                        def update_progress(status):
                            st.session_state.mcp_crawl_status = status
                        
                        # Start the crawling process in a separate thread
                        st.session_state.mcp_crawl_tracker = start_mcp_crawl(update_progress)
                        st.session_state.mcp_crawl_status = st.session_state.mcp_crawl_tracker.get_status()
                        
                        # Force a rerun to start showing progress
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors du démarrage du crawl: {str(e)}")
            
            with col2:
                # Button to clear existing MCP docs
                if st.button("Effacer la documentation MCP", key="clear_mcp"):
                    with st.spinner("Effacement de la documentation MCP existante..."):
                        try:
                            # Run the function to clear records
                            clear_mcp_records()
                            st.success("✅ Documentation MCP effacée avec succès de la base de données.")
                            
                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'effacement de la documentation MCP: {str(e)}")
            
            # Display crawling progress if a crawl is in progress or has completed
            if st.session_state.mcp_crawl_tracker:
                # Create a container for the progress information
                progress_container = st.container()
                
                with progress_container:
                    # Get the latest status
                    current_time = time.time()
                    # Update status every second
                    if current_time - st.session_state.mcp_last_update_time >= 1:
                        st.session_state.mcp_crawl_status = st.session_state.mcp_crawl_tracker.get_status()
                        st.session_state.mcp_last_update_time = current_time
                    
                    status = st.session_state.mcp_crawl_status
                    
                    # Display a progress bar
                    if status and status["urls_found"] > 0:
                        progress = status["urls_processed"] / status["urls_found"]
                        st.progress(progress)
                    
                    # Display status metrics
                    col1, col2, col3, col4 = st.columns(4)
                    if status:
                        col1.metric("URLs trouvées", status["urls_found"])
                        col2.metric("URLs traitées", status["urls_processed"])
                        col3.metric("Réussies", status["urls_succeeded"])
                        col4.metric("Échouées", status["urls_failed"])
                    else:
                        col1.metric("URLs trouvées", 0)
                        col2.metric("URLs traitées", 0)
                        col3.metric("Réussies", 0)
                        col4.metric("Échouées", 0)
                    
                    # Display logs in an expander
                    with st.expander("Logs de crawling", expanded=True):
                        if status and "logs" in status:
                            logs_text = "\n".join(status["logs"][-20:])  # Show last 20 logs
                            st.code(logs_text)
                        else:
                            st.code("Aucun log disponible pour le moment...")
                    
                    # Show completion message
                    if status and not status["is_running"] and status["end_time"]:
                        if status["urls_failed"] == 0:
                            st.success("✅ Processus de crawling terminé avec succès!")
                        else:
                            st.warning(f"⚠️ Processus de crawling terminé avec {status['urls_failed']} URLs échouées.")
                
                # Auto-refresh while crawling is in progress
                if not status or status["is_running"]:
                    st.rerun()
            
            # Display database statistics
            st.subheader("Statistiques de la base de données")
            try:            
                # Query the count of MCP docs
                result = supabase_client.table("site_pages").select("count", count="exact").eq("metadata->>source", "mcp_docs").execute()
                count = result.count if hasattr(result, "count") else 0
                
                # Display the count
                st.metric("Chunks de documentation MCP", count)
                
                # Add a button to view the data
                if count > 0 and st.button("Voir les données indexées", key="view_mcp_data"):
                    # Query a sample of the data
                    sample_data = supabase_client.table("site_pages").select("url,title,summary,chunk_number").eq("metadata->>source", "mcp_docs").limit(10).execute()
                    
                    # Display the sample data
                    st.dataframe(sample_data.data)
                    st.info("Affichage d'un échantillon de 10 enregistrements. La base de données contient plus d'enregistrements.")
            except Exception as e:
                st.error(f"Erreur lors de la requête à la base de données: {str(e)}")
    
    with doc_tabs[2]:
        st.info("Additional documentation sources will be available in future updates.")