from __future__ import annotations

# Fix imports first
import sys
import os
from pathlib import Path

# Ajouter les chemins nécessaires au sys.path pour résoudre les problèmes d'importation
# IMPORTANT: Garder '/app/src' en premier pour que le package top-level `archon` pointe vers '/app/src/archon'
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
# Ne pas insérer '/app/src/archon' en tête, cela masque le package top-level et casse 'archon.utils.utils'
# Si nécessaire pour des imports relatifs avancés, on l'ajoutera en fin de sys.path
if '/app/src/archon' not in sys.path:
    sys.path.append('/app/src/archon')

# Ajout d'une backup du dossier utils au niveau de /app/src
utils_path = Path('/app/src/archon/utils')
target_path = Path('/app/src/utils')

if utils_path.exists() and not target_path.exists():
    try:
        # Créer le répertoire cible s'il n'existe pas
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copier les fichiers nécessaires
        for py_file in utils_path.glob('*.py'):
            with open(py_file, 'r') as src_file:
                content = src_file.read()
            
            with open(target_path / py_file.name, 'w') as dest_file:
                dest_file.write(content)
        
        print(f"✅ Utils copiés vers {target_path}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la copie des utils: {str(e)}")

from dotenv import load_dotenv
import streamlit as st
import logfire
import asyncio
import json

# Set page config - must be the first commande Streamlit
st.set_page_config(
    page_title="Archon - Agent Builder",
    page_icon="🤖",
    layout="wide"
)

# Utilities and styles
# Prefer package-qualified import; fallback to backup copy in /app/src/utils
try:
    from archon.utils.utils import get_clients
except ModuleNotFoundError:
    from utils.utils import get_clients
from streamlit_pages.styles import load_css
from archon.utils.utils import load_env_vars  # for fallback Supabase discovery

# Streamlit pages
from streamlit_pages.intro import intro_tab
from streamlit_pages.chat import chat_tab
from streamlit_pages.environment import environment_tab
from streamlit_pages.database import database_tab
from streamlit_pages.documentation import documentation_tab
from streamlit_pages.agent_service import agent_service_tab
from streamlit_pages.mcp import mcp_tab
from streamlit_pages.future_enhancements import future_enhancements_tab
from streamlit_pages.neo4j import neo4j_tab

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client, supabase, neo4j_client = get_clients()

# Fallback: if Supabase not initialized by get_clients(), try cross-profile discovery here
if supabase is None:
    try:
        # Prefer modern supabase-py v2
        try:
            from supabase import create_client as _supabase_create_client  # type: ignore
        except Exception:
            _supabase_create_client = None  # type: ignore
        if _supabase_create_client is not None:
            env = load_env_vars() or {}
            profiles = (env.get("profiles") or {})
            supabase_url = None
            supabase_key = None
            # Search any profile for both credentials
            for _pname, _pcfg in profiles.items():
                if not supabase_url:
                    supabase_url = _pcfg.get("SUPABASE_URL") or supabase_url
                if not supabase_key:
                    supabase_key = _pcfg.get("SUPABASE_SERVICE_KEY") or supabase_key
                if supabase_url and supabase_key:
                    break
            if supabase_url and supabase_key:
                try:
                    supabase = _supabase_create_client(supabase_url, supabase_key)
                    print("✅ Supabase fallback client initialized via streamlit_ui.py")
                except Exception as e:
                    print(f"⚠️ Supabase fallback init failed: {e}")
    except Exception as _e:
        print(f"⚠️ Supabase fallback probe error: {_e}")

# Load custom CSS styles
load_css()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

async def main():
    # Check for tab query parameter
    query_params = st.query_params
    if "tab" in query_params:
        tab_name = query_params["tab"]
        if tab_name in ["Intro", "Chat", "Environment", "Database", "Documentation", "Agent Service", "MCP", "Neo4j", "Future Enhancements"]:
            st.session_state.selected_tab = tab_name

    # Add sidebar navigation
    with st.sidebar:
        # Logo removed to avoid MediaFileStorageError when asset is not present in the image
        st.caption("Archon UI")
        
        # Navigation options with vertical buttons
        st.write("### Navigation")
        
        # Initialize session state for selected tab if not present
        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Intro"
        
        # Vertical navigation buttons
        intro_button = st.button("Intro", use_container_width=True, key="intro_button")
        chat_button = st.button("Chat", use_container_width=True, key="chat_button")
        env_button = st.button("Environment", use_container_width=True, key="env_button")
        db_button = st.button("Database", use_container_width=True, key="db_button")
        docs_button = st.button("Documentation", use_container_width=True, key="docs_button")
        service_button = st.button("Agent Service", use_container_width=True, key="service_button")
        mcp_button = st.button("MCP", use_container_width=True, key="mcp_button")
        neo4j_button = st.button("Neo4j", use_container_width=True, key="neo4j_button")
        future_enhancements_button = st.button("Future Enhancements", use_container_width=True, key="future_enhancements_button")
        
        # Update selected tab based on button clicks
        if intro_button:
            st.session_state.selected_tab = "Intro"
        elif chat_button:
            st.session_state.selected_tab = "Chat"
        elif mcp_button:
            st.session_state.selected_tab = "MCP"
        elif env_button:
            st.session_state.selected_tab = "Environment"
        elif service_button:
            st.session_state.selected_tab = "Agent Service"
        elif db_button:
            st.session_state.selected_tab = "Database"
        elif docs_button:
            st.session_state.selected_tab = "Documentation"
        elif neo4j_button:
            st.session_state.selected_tab = "Neo4j"
        elif future_enhancements_button:
            st.session_state.selected_tab = "Future Enhancements"
    
    # Display the selected tab
    if st.session_state.selected_tab == "Intro":
        st.title("Archon - Introduction")
        intro_tab()
    elif st.session_state.selected_tab == "Chat":
        st.title("Archon - Agent Builder")
        await chat_tab()
    elif st.session_state.selected_tab == "MCP":
        st.title("Archon - MCP Configuration")
        mcp_tab()
    elif st.session_state.selected_tab == "Environment":
        st.title("Archon - Environment Configuration")
        environment_tab()
    elif st.session_state.selected_tab == "Agent Service":
        st.title("Archon - Agent Service")
        agent_service_tab()
    elif st.session_state.selected_tab == "Database":
        st.title("Archon - Database Configuration")
        database_tab(supabase)
    elif st.session_state.selected_tab == "Documentation":
        st.title("Archon - Documentation")
        documentation_tab(supabase)
    elif st.session_state.selected_tab == "Neo4j":
        st.title("Archon - Neo4j Graph Database")
        neo4j_tab(neo4j_client)
    elif st.session_state.selected_tab == "Future Enhancements":
        st.title("Archon - Future Enhancements")
        future_enhancements_tab()

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(main())
