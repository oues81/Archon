"""
Client Neo4j pour Archon.

Ce module fournit un client Neo4j pour l'intégration avec Archon,
permettant d'interagir avec la base de données graphe fournie par Local AI Packaged.
"""
from typing import Dict, List, Any, Optional, Union
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Client pour interagir avec Neo4j dans le cadre du projet Archon"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialiser la connexion Neo4j
        
        Args:
            uri: URI de connexion au serveur Neo4j (ex: "bolt://localhost:7687")
            username: Nom d'utilisateur Neo4j
            password: Mot de passe Neo4j
            database: Nom de la base de données (défaut: "neo4j")
        """
        self.uri = uri
        self.username = username
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Vérifier la connexion
        try:
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            self.driver.close()
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise Exception(f"Failed to connect to Neo4j: {str(e)}")
    
    def __del__(self):
        """Fermer la connexion lors de la destruction de l'objet"""
        self.close()
    
    def close(self):
        """Fermer la connexion Neo4j"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            self.driver = None
    
    def run_query(self, query: str, params: Dict = None) -> List[Dict[str, Any]]:
        """
        Exécuter une requête Cypher
        
        Args:
            query: Requête Cypher à exécuter
            params: Paramètres pour la requête
            
        Returns:
            Liste de résultats sous forme de dictionnaires
        """
        if params is None:
            params = {}
        
        try:    
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                records = [dict(record) for record in result]
                return records
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise Exception(f"Error executing query: {str(e)}")
    
    def create_entity(self, label: str, properties: Dict) -> Dict:
        """
        Créer une entité dans Neo4j
        
        Args:
            label: Type d'entité (label Neo4j)
            properties: Propriétés de l'entité
            
        Returns:
            Dictionnaire représentant l'entité créée
        """
        query = f"""
        CREATE (n:{label} $props)
        RETURN n
        """
        result = self.run_query(query, {"props": properties})
        return result[0]["n"] if result else None
    
    def create_relationship(self, 
                           from_entity_id: int, 
                           to_entity_id: int, 
                           relation_type: str, 
                           properties: Dict = None) -> Dict:
        """
        Créer une relation entre deux entités
        
        Args:
            from_entity_id: ID de l'entité source
            to_entity_id: ID de l'entité cible
            relation_type: Type de relation
            properties: Propriétés de la relation
            
        Returns:
            Dictionnaire représentant la relation créée
        """
        if properties is None:
            properties = {}
            
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:{relation_type} $props]->(b)
        RETURN r
        """
        result = self.run_query(query, {
            "from_id": from_entity_id,
            "to_id": to_entity_id,
            "props": properties
        })
        return result[0]["r"] if result else None
    
    def find_entity(self, label: str = None, properties: Dict = None, limit: int = 10) -> List[Dict]:
        """
        Rechercher des entités correspondant aux critères
        
        Args:
            label: Type d'entité (optionnel)
            properties: Propriétés de filtrage (optionnel)
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des entités correspondantes
        """
        if properties is None:
            properties = {}
            
        # Construire la clause WHERE dynamiquement
        where_clauses = []
        params = {}
        
        for key, value in properties.items():
            where_clauses.append(f"n.{key} = ${key}")
            params[key] = value
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        label_clause = f":{label}" if label else ""
        
        query = f"""
        MATCH (n{label_clause})
        WHERE {where_clause}
        RETURN n
        LIMIT {limit}
        """
        
        result = self.run_query(query, params)
        return [record["n"] for record in result]
    
    def find_related_entities(self, entity_id: int, relation_type: str = None, 
                            direction: str = "OUTGOING", limit: int = 10) -> List[Dict]:
        """
        Trouver les entités liées à une entité donnée
        
        Args:
            entity_id: ID de l'entité
            relation_type: Type de relation (optionnel)
            direction: Direction de la relation ("OUTGOING" ou "INCOMING")
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des entités liées
        """
        rel_type = f":{relation_type}" if relation_type else ""
        
        if direction == "OUTGOING":
            query = f"""
            MATCH (a)-[r{rel_type}]->(b)
            WHERE id(a) = $entity_id
            RETURN b, r, type(r) as relation_type
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (a)<-[r{rel_type}]-(b)
            WHERE id(a) = $entity_id
            RETURN b, r, type(r) as relation_type
            LIMIT {limit}
            """
        
        result = self.run_query(query, {"entity_id": entity_id})
        return [{
            "entity": record["b"],
            "relation": record["r"],
            "relation_type": record["relation_type"]
        } for record in result]
    
    def delete_entity(self, entity_id: int) -> bool:
        """
        Supprimer une entité et toutes ses relations
        
        Args:
            entity_id: ID de l'entité à supprimer
            
        Returns:
            True si la suppression a réussi
        """
        query = """
        MATCH (n)
        WHERE id(n) = $entity_id
        DETACH DELETE n
        """
        self.run_query(query, {"entity_id": entity_id})
        return True
    
    def get_database_stats(self) -> Dict:
        """
        Obtenir des statistiques sur la base de données
        
        Returns:
            Dictionnaire contenant les statistiques (nombre de nœuds, relations, etc.)
        """
        query = """
        MATCH (n)
        RETURN 
            count(n) as node_count,
            size([()--() | 1]) as relationship_count,
            count(distinct labels(n)) as label_count
        """
        result = self.run_query(query)
        return result[0] if result else {"node_count": 0, "relationship_count": 0, "label_count": 0}
    
    def get_node_labels(self) -> List[Dict]:
        """
        Obtenir la liste des labels de nœuds avec leur compte
        
        Returns:
            Liste des labels et leur nombre d'occurrences
        """
        query = """
        CALL db.labels() YIELD label
        RETURN label, count{MATCH (n) WHERE label in labels(n) RETURN n} as count
        ORDER BY count DESC
        """
        return self.run_query(query)
