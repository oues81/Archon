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
    
    def create_entity(self, label: str, entity_name: str, properties: Dict = None) -> Dict:
        """
        Créer une entité dans Neo4j
        
        Args:
            label: Type d'entité (label Neo4j)
            entity_name: Nom de l'entité (identifiant)
            properties: Propriétés additionnelles de l'entité
            
        Returns:
            Dictionnaire représentant l'entité créée
        """
        if properties is None:
            properties = {}
        
        # S'assurer que le nom est inclus dans les propriétés
        props = {"name": entity_name, **properties}
        
        query = f"""
        CREATE (n:{label} $props)
        RETURN n
        """
        result = self.run_query(query, {"props": props})
        return result[0]["n"] if result else None
    
    def create_relation(self, from_entity_name: str, to_entity_name: str, relation_type: str, properties: Dict = None) -> Dict:
        """
        Créer une relation entre deux entités en utilisant leur nom
        
        Args:
            from_entity_name: Nom de l'entité source
            to_entity_name: Nom de l'entité cible
            relation_type: Type de relation
            properties: Propriétés de la relation
            
        Returns:
            Dictionnaire représentant la relation créée
        """
        if properties is None:
            properties = {}
            
        query = f"""
        MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
        CREATE (a)-[r:{relation_type} $props]->(b)
        RETURN r
        """
        result = self.run_query(query, {
            "from_name": from_entity_name,
            "to_name": to_entity_name,
            "props": properties
        })
        return result[0]["r"] if result else None
            
    def create_relationship(self, 
                           from_entity_id: int, 
                           to_entity_id: int, 
                           relation_type: str, 
                           properties: Dict = None) -> Dict:
        """
        Créer une relation entre deux entités en utilisant leur ID
        
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
        
    def find_entities_by_properties(self, properties_dict: Dict[str, Any], 
                                   label: str = None, operator: str = "AND",
                                   use_regex: bool = False, limit: int = 50) -> List[Dict]:
        """
        Recherche avancée d'entités par propriétés avec opérateurs flexibles
        
        Args:
            properties_dict: Dictionnaire des propriétés à rechercher
            label: Type d'entité (optionnel)
            operator: Opérateur de combinaison ("AND" ou "OR")
            use_regex: Utiliser des expressions régulières pour la correspondance
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des entités correspondantes
        """
        if operator not in ["AND", "OR"]:
            operator = "AND"
            
        # Construire la clause WHERE dynamiquement
        where_clauses = []
        params = {}
        
        for i, (key, value) in enumerate(properties_dict.items()):
            param_name = f"prop_{i}"
            if use_regex:
                where_clauses.append(f"n.{key} =~ ${param_name}")
            else:
                where_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
        
        where_clause = f" {operator} ".join(where_clauses) if where_clauses else "1=1"
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
        Supprimer une entité et toutes ses relations par ID
        
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
        
    def delete_entity_by_name(self, entity_name: str, label: str = None) -> bool:
        """
        Supprimer une entité et toutes ses relations par son nom
        
        Args:
            entity_name: Nom de l'entité à supprimer
            label: Type d'entité (optionnel)
            
        Returns:
            True si la suppression a réussi, False si l'entité n'existe pas
        """
        label_clause = f":{label}" if label else ""
        
        query = f"""
        MATCH (n{label_clause} {{name: $name}})
        WITH n, count(n) as count
        DETACH DELETE n
        RETURN count > 0 as deleted
        """
        
        result = self.run_query(query, {"name": entity_name})
        return result[0]["deleted"] if result else False
        
    def update_entity(self, entity_name: str, properties: Dict[str, Any]) -> Dict:
        """
        Met à jour les propriétés d'une entité existante
        
        Args:
            entity_name: Nom de l'entité à mettre à jour
            properties: Nouvelles propriétés à appliquer (fusionnées avec les existantes)
            
        Returns:
            Dictionnaire représentant l'entité mise à jour
        """
        query = """
        MATCH (n {name: $name})
        SET n += $props
        RETURN n
        """
        
        result = self.run_query(query, {"name": entity_name, "props": properties})
        return result[0]["n"] if result else None
    
    def find_shortest_path(self, from_entity_name: str, to_entity_name: str, max_depth: int = 4) -> List[Dict]:
        """
        Trouve le chemin le plus court entre deux entités
        
        Args:
            from_entity_name: Nom de l'entité de départ
            to_entity_name: Nom de l'entité d'arrivée
            max_depth: Profondeur maximale de recherche
            
        Returns:
            Liste des nœuds et relations formant le chemin
        """
        query = f"""
        MATCH path = shortestPath((a {{name: $from_name}})-[*1..{max_depth}]-(b {{name: $to_name}}))
        RETURN path, length(path) as path_length
        """
        
        result = self.run_query(query, {
            "from_name": from_entity_name,
            "to_name": to_entity_name
        })
        
        return result[0]["path"] if result else []
    
    def find_paths_between(self, from_entity_name: str, to_entity_name: str, 
                         max_depth: int = 3, limit: int = 5) -> List[Dict]:
        """
        Trouve tous les chemins entre deux entités
        
        Args:
            from_entity_name: Nom de l'entité de départ
            to_entity_name: Nom de l'entité d'arrivée
            max_depth: Profondeur maximale de recherche
            limit: Nombre maximum de chemins à retourner
            
        Returns:
            Liste des chemins trouvés
        """
        query = f"""
        MATCH path = (a {{name: $from_name}})-[*1..{max_depth}]-(b {{name: $to_name}})
        RETURN path, length(path) as path_length
        ORDER BY path_length ASC
        LIMIT {limit}
        """
        
        result = self.run_query(query, {
            "from_name": from_entity_name,
            "to_name": to_entity_name
        })
        
        return [record["path"] for record in result]
        
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
        
    def find_connections_between_topics(self, topic1: str, topic2: str, max_depth: int = 3) -> List[Dict]:
        """
        Trouve les connexions entre deux sujets technologiques
        
        Args:
            topic1: Premier sujet ou technologie
            topic2: Second sujet ou technologie
            max_depth: Profondeur maximale de recherche
            
        Returns:
            Liste des chemins connectant les deux sujets
        """
        # Cette méthode utilise des expressions régulières pour trouver les entités liées aux sujets
        # même si les noms ne correspondent pas exactement
        query = f"""
        MATCH (a), (b)
        WHERE a.name =~ $topic1_pattern AND b.name =~ $topic2_pattern
        WITH a, b
        MATCH path = shortestPath((a)-[*1..{max_depth}]-(b))
        RETURN path, length(path) as path_length
        ORDER BY path_length ASC
        LIMIT 5
        """
        
        result = self.run_query(query, {
            "topic1_pattern": f"(?i).*{topic1}.*",  # Case insensitive, contains topic1
            "topic2_pattern": f"(?i).*{topic2}.*"   # Case insensitive, contains topic2
        })
        
        return [record["path"] for record in result]
        
    def find_influential_nodes(self, label: str = None, limit: int = 10) -> List[Dict]:
        """
        Trouve les nœuds les plus influents dans le graphe (ayant le plus de relations)
        
        Args:
            label: Filtrer par type d'entité (optionnel)
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des entités les plus influentes avec leur score
        """
        label_clause = f":{label}" if label else ""
        
        query = f"""
        MATCH (n{label_clause})
        MATCH (n)-[r]-()
        WITH n, COUNT(r) as degree
        RETURN n.name as name, n as entity, degree
        ORDER BY degree DESC
        LIMIT {limit}
        """
        
        return self.run_query(query)
        
    def recommend_related_technologies(self, tech_name: str, limit: int = 5) -> List[Dict]:
        """
        Recommande des technologies liées à une technologie donnée
        Spécialement conçu pour le projet STI Map Generator
        
        Args:
            tech_name: Nom de la technologie
            limit: Nombre de recommandations
            
        Returns:
            Liste des technologies recommandées avec score
        """
        query = f"""
        MATCH (t {{name: $tech_name}})-[r*1..2]-(related)
        WHERE related.name <> $tech_name
        WITH related, count(r) as relevance_score
        RETURN related.name as name, related as entity, relevance_score
        ORDER BY relevance_score DESC
        LIMIT {limit}
        """
        
        return self.run_query(query, {"tech_name": tech_name})
    
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
