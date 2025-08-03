"""
Test de l'intégration Neo4j dans Archon.

Ce script teste la connexion à Neo4j via Local AI Packaged
et valide les fonctionnalités de base du client Neo4j.
"""
import os
import pytest
from dotenv import load_dotenv
from archon.utils.neo4j_client import Neo4jClient
from archon.utils.utils import get_env_var

# Chargement des variables d'environnement
load_dotenv()

def test_neo4j_client_initialization():
    """Test l'initialisation du client Neo4j avec les variables d'environnement."""
    # Récupérer les variables d'environnement
    uri = get_env_var("NEO4J_URI")
    username = get_env_var("NEO4J_USER")
    password = get_env_var("NEO4J_PASSWORD")
    database = get_env_var("NEO4J_DATABASE") or "neo4j"
    
    # Vérifier si les variables d'environnement sont définies
    if not all([uri, username, password]):
        pytest.skip("Variables d'environnement Neo4j non configurées")
    
    try:
        # Créer le client Neo4j
        client = Neo4jClient(uri, username, password, database)
        
        # Vérifier que le client est correctement initialisé
        assert client is not None
        assert client.uri == uri
        assert client.username == username
        assert client.database == database
        
        # Vérifier la connexion en exécutant une requête simple
        result = client.run_query("RETURN 'test' AS test")
        assert result[0]['test'] == 'test'
        
        # Fermer la connexion
        client.close()
    except Exception as e:
        pytest.fail(f"Erreur lors de l'initialisation du client Neo4j: {str(e)}")

def test_neo4j_crud_operations():
    """Test les opérations CRUD de base avec le client Neo4j."""
    # Récupérer les variables d'environnement
    uri = get_env_var("NEO4J_URI")
    username = get_env_var("NEO4J_USER")
    password = get_env_var("NEO4J_PASSWORD")
    database = get_env_var("NEO4J_DATABASE") or "neo4j"
    
    # Vérifier si les variables d'environnement sont définies
    if not all([uri, username, password]):
        pytest.skip("Variables d'environnement Neo4j non configurées")
    
    try:
        # Créer le client Neo4j
        client = Neo4jClient(uri, username, password, database)
        
        # 1. Créer une entité de test
        test_properties = {
            "name": "TestEntity",
            "value": "This is a test",
            "created_by": "test_neo4j_integration.py"
        }
        
        # Supprimer les entités de test existantes pour éviter les conflits
        client.run_query(
            "MATCH (n:TestNode {created_by: 'test_neo4j_integration.py'}) DETACH DELETE n"
        )
        
        # Créer une nouvelle entité
        result = client.create_entity("TestNode", test_properties)
        assert result is not None
        
        # 2. Rechercher l'entité créée
        entities = client.find_entity("TestNode", {"name": "TestEntity"})
        assert len(entities) > 0
        assert entities[0]["name"] == "TestEntity"
        assert entities[0]["value"] == "This is a test"
        
        # Récupérer l'ID pour les tests suivants
        entity_id = entities[0].id
        
        # 3. Créer une autre entité pour tester les relations
        related_properties = {
            "name": "RelatedEntity",
            "value": "This is a related entity",
            "created_by": "test_neo4j_integration.py"
        }
        related_result = client.create_entity("TestNode", related_properties)
        assert related_result is not None
        
        # Récupérer l'ID de l'entité liée
        related_entities = client.find_entity("TestNode", {"name": "RelatedEntity"})
        assert len(related_entities) > 0
        related_id = related_entities[0].id
        
        # 4. Créer une relation entre les deux entités
        relation_properties = {
            "type": "TEST_RELATION",
            "created_by": "test_neo4j_integration.py"
        }
        relation = client.create_relationship(entity_id, related_id, "RELATES_TO", relation_properties)
        assert relation is not None
        
        # 5. Vérifier les relations
        related = client.find_related_entities(entity_id)
        assert len(related) > 0
        assert related[0]["entity"]["name"] == "RelatedEntity"
        assert related[0]["relation_type"] == "RELATES_TO"
        
        # 6. Récupérer des statistiques de base de données
        stats = client.get_database_stats()
        assert "node_count" in stats
        assert "relationship_count" in stats
        
        # 7. Nettoyer: supprimer les entités de test
        client.run_query(
            "MATCH (n:TestNode {created_by: 'test_neo4j_integration.py'}) DETACH DELETE n"
        )
        
        # Vérifier que les entités ont bien été supprimées
        remaining = client.find_entity("TestNode", {"created_by": "test_neo4j_integration.py"})
        assert len(remaining) == 0
        
        # Fermer la connexion
        client.close()
    except Exception as e:
        pytest.fail(f"Erreur lors des opérations CRUD avec Neo4j: {str(e)}")

if __name__ == "__main__":
    print("Test de l'intégration Neo4j dans Archon...")
    
    # Test de l'initialisation du client
    try:
        test_neo4j_client_initialization()
        print("✅ Test d'initialisation réussi")
    except Exception as e:
        print(f"❌ Test d'initialisation échoué: {str(e)}")
    
    # Test des opérations CRUD
    try:
        test_neo4j_crud_operations()
        print("✅ Test des opérations CRUD réussi")
    except Exception as e:
        print(f"❌ Test des opérations CRUD échoué: {str(e)}")
