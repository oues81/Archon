"""
Interface Streamlit pour la gestion de Neo4j dans Archon.

Ce module fournit une interface utilisateur pour configurer et interagir avec 
la base de données graphe Neo4j utilisée via Local AI Packaged.
"""
import streamlit as st
import os
import pandas as pd
from typing import Optional

# Import des classes et fonctions nécessaires de manière conditionnelle pour éviter les erreurs
try:
    # Essayer d'abord en utilisant un chemin d'accès absolu
    from utils.neo4j_client import Neo4jClient
    from utils.utils import get_env_var, save_env_var as set_env_var, write_to_log
    print("Import réussi via 'utils.*'")
except ImportError:
    try:
        # Si ça échoue, essayer un autre chemin
        sys_path_modified = False
        import sys
        if '/app/src' not in sys.path:
            sys.path.append('/app/src')
            sys_path_modified = True
        
        from utils.neo4j_client import Neo4jClient
        from utils.utils import get_env_var, save_env_var as set_env_var, write_to_log
        print("Import réussi après modification du sys.path")
    except ImportError as e:
        # Fallback avec message d'erreur
        print(f"ERREUR D'IMPORT: {str(e)}")
        
        # Créer des versions stub des classes/fonctions manquantes pour éviter les erreurs fatales
        class Neo4jClient:
            def __init__(self, *args, **kwargs):
                print("Neo4jClient (stub) initialisé")
                self.connected = False
            
            def is_connected(self):
                return False
        
        def get_env_var(name, default=None):
            return os.environ.get(name, default)
        
        def set_env_var(name, value):
            os.environ[name] = value
        
        def write_to_log(message, level="INFO"):
            print(f"[{level}] {message}")
        
        print("Utilisation des versions stub des fonctions manquantes")
        st.error("Impossible de charger les modules nécessaires pour Neo4j. Certaines fonctionnalités peuvent être limitées.")


def neo4j_tab(neo4j_client: Optional[Neo4jClient] = None):
    """
    Affiche l'onglet de configuration et gestion Neo4j dans l'interface Streamlit.
    
    Args:
        neo4j_client: Instance de Neo4jClient déjà initialisée (optionnelle)
    """
    st.header("Neo4j Configuration")
    
    # Configuration
    with st.expander("Configuration", expanded=not neo4j_client):
        # Récupérer les variables d'environnement actuelles
        neo4j_uri = st.text_input(
            "Neo4j URI",
            value=get_env_var("NEO4J_URI") or "bolt://localhost:7687",
            help="URI de connexion à Neo4j (ex: bolt://localhost:7687)"
        )
        
        neo4j_user = st.text_input(
            "Neo4j Username",
            value=get_env_var("NEO4J_USER") or "neo4j",
            help="Nom d'utilisateur Neo4j (défaut: neo4j)"
        )
        
        neo4j_password = st.text_input(
            "Neo4j Password",
            value=get_env_var("NEO4J_PASSWORD") or "",
            type="password",
            help="Mot de passe Neo4j"
        )
        
        neo4j_database = st.text_input(
            "Neo4j Database",
            value=get_env_var("NEO4J_DATABASE") or "neo4j",
            help="Nom de la base Neo4j (défaut: neo4j)"
        )
        
        # Bouton pour sauvegarder la configuration
        if st.button("Save Neo4j Configuration"):
            # Mise à jour des variables d'environnement
            set_env_var("NEO4J_URI", neo4j_uri)
            set_env_var("NEO4J_USER", neo4j_user)
            set_env_var("NEO4J_PASSWORD", neo4j_password)
            set_env_var("NEO4J_DATABASE", neo4j_database)
            
            # Afficher un message de confirmation
            st.success("Neo4j configuration saved to environment variables")
            write_to_log("Neo4j configuration updated")
    
    # Test de connexion
    with st.expander("Connection Test", expanded=not neo4j_client):
        if st.button("Test Neo4j Connection"):
            try:
                # Récupérer les variables d'environnement actuelles
                uri = get_env_var("NEO4J_URI")
                user = get_env_var("NEO4J_USER")
                password = get_env_var("NEO4J_PASSWORD")
                database = get_env_var("NEO4J_DATABASE") or "neo4j"
                
                if not all([uri, user, password]):
                    st.error("Missing Neo4j connection parameters")
                    return
                
                # Création d'un client Neo4j temporaire pour le test
                test_client = Neo4jClient(uri, user, password)
                
                # Exécuter une requête simple pour vérifier la connexion
                result = test_client.run_query("RETURN 'Connection successful' AS message")
                
                if result and result[0].get('message') == 'Connection successful':
                    st.success("Connected to Neo4j successfully!")
                    # Récupérer des statistiques de base (robuste si la méthode manque)
                    if hasattr(test_client, 'get_database_stats'):
                        stats = test_client.get_database_stats()
                    else:
                        # Fallback direct via Cypher
                        rows = test_client.run_query(
                            """
                            MATCH (n)
                            RETURN 
                              count(n) as node_count,
                              size([()--() | 1]) as relationship_count,
                              count(distinct labels(n)) as label_count
                            """
                        )
                        stats = rows[0] if rows else {"node_count": 0, "relationship_count": 0, "label_count": 0}
                    st.info(f"Database Stats: {stats['node_count']} nodes, {stats['relationship_count']} relationships")
                else:
                    st.error("Connection test returned unexpected results")
                
                # Fermer la connexion de test
                test_client.close()
                
            except Exception as e:
                st.error(f"Failed to connect to Neo4j: {str(e)}")
                write_to_log(f"Neo4j connection test failed: {str(e)}")
    
    # Si un client est fourni ou peut être créé, afficher les fonctionnalités de gestion
    client = neo4j_client
    if not client:
        # Essayer de créer un client avec les variables d'environnement actuelles
        uri = get_env_var("NEO4J_URI")
        user = get_env_var("NEO4J_USER")
        password = get_env_var("NEO4J_PASSWORD")
        database = get_env_var("NEO4J_DATABASE") or "neo4j"
        
        if all([uri, user, password]):
            try:
                client = Neo4jClient(uri, user, password)
            except Exception as e:
                st.warning("Neo4j client could not be initialized automatically.")
    
    if client:
        # Afficher les fonctionnalités de gestion Neo4j
        st.subheader("Database Management")
        
        with st.expander("Database Information"):
            try:
                # Récupérer des statistiques
                if hasattr(client, 'get_database_stats'):
                    stats = client.get_database_stats()
                else:
                    rows = client.run_query(
                        """
                        MATCH (n)
                        RETURN 
                          count(n) as node_count,
                          size([()--() | 1]) as relationship_count,
                          count(distinct labels(n)) as label_count
                        """
                    )
                    stats = rows[0] if rows else {"node_count": 0, "relationship_count": 0, "label_count": 0}
                st.metric("Nodes", stats['node_count'])
                st.metric("Relationships", stats['relationship_count'])
                st.metric("Node Labels", stats['label_count'])
                
                # Afficher les labels disponibles
                if hasattr(client, 'get_node_labels'):
                    labels = client.get_node_labels()
                else:
                    labels = client.run_query(
                        """
                        CALL db.labels() YIELD label
                        RETURN label, count{MATCH (n) WHERE label in labels(n) RETURN n} as count
                        ORDER BY count DESC
                        """
                    )
                if labels:
                    st.subheader("Node Labels")
                    label_df = pd.DataFrame(labels)
                    st.dataframe(label_df)
                
            except Exception as e:
                st.error(f"Error retrieving database information: {str(e)}")
        
        with st.expander("Run Cypher Query"):
            query = st.text_area(
                "Cypher Query",
                height=150,
                help="Enter a Cypher query to execute",
                placeholder="MATCH (n) RETURN n LIMIT 10"
            )
            
            if st.button("Run Query"):
                if query:
                    try:
                        results = client.run_query(query)
                        st.success(f"Query executed successfully. {len(results)} results returned.")
                        
                        if results:
                            # Convertir les résultats en dataframe pour l'affichage
                            # Note: Cela peut nécessiter un traitement supplémentaire pour les types complexes
                            try:
                                # Tenter d'aplatir les résultats pour l'affichage
                                flat_results = []
                                for record in results:
                                    flat_record = {}
                                    for key, value in record.items():
                                        if isinstance(value, dict):
                                            for k, v in value.items():
                                                flat_record[f"{key}.{k}"] = v
                                        else:
                                            flat_record[key] = value
                                    flat_results.append(flat_record)
                                
                                df = pd.DataFrame(flat_results)
                                st.dataframe(df)
                            except Exception as e:
                                st.warning(f"Could not display results as table: {str(e)}")
                                st.json(results)
                        else:
                            st.info("Query executed, but no results were returned.")
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
                else:
                    st.warning("Please enter a query to execute.")
        
        with st.expander("Node Management"):
            st.subheader("Create Node")
            
            # Sélection du label
            node_label = st.text_input("Node Label", placeholder="Person")
            
            # Propriétés dynamiques
            st.subheader("Properties")
            property_count = st.number_input("Number of properties", min_value=1, max_value=10, value=2)
            
            properties = {}
            for i in range(property_count):
                col1, col2 = st.columns(2)
                with col1:
                    key = st.text_input(f"Property {i+1} name", key=f"prop_name_{i}")
                with col2:
                    value = st.text_input(f"Property {i+1} value", key=f"prop_value_{i}")
                if key:
                    properties[key] = value
            
            if st.button("Create Node"):
                if node_label and properties:
                    try:
                        result = client.create_entity(node_label, properties)
                        if result:
                            st.success(f"Node created successfully!")
                        else:
                            st.warning("Node creation did not return a result.")
                    except Exception as e:
                        st.error(f"Error creating node: {str(e)}")
                else:
                    st.warning("Please provide a label and at least one property.")
            
            st.subheader("Find Nodes")
            search_label = st.text_input("Search by Label", placeholder="Person")
            search_property_key = st.text_input("Property Key (optional)", placeholder="name")
            search_property_value = st.text_input("Property Value (optional)", placeholder="John")
            
            if st.button("Search Nodes"):
                try:
                    search_props = {}
                    if search_property_key and search_property_value:
                        search_props[search_property_key] = search_property_value
                    
                    results = client.find_entity(search_label, search_props, limit=20)
                    if results:
                        st.success(f"Found {len(results)} nodes")
                        
                        # Convertir en dataframe pour l'affichage
                        nodes_data = []
                        for node in results:
                            node_data = {"id": node.id}
                            node_data.update(node._properties)
                            nodes_data.append(node_data)
                        
                        st.dataframe(pd.DataFrame(nodes_data))
                    else:
                        st.info("No nodes found matching the criteria.")
                except Exception as e:
                    st.error(f"Error searching nodes: {str(e)}")
    else:
        st.info("Configure and test your Neo4j connection to access database management features.")
