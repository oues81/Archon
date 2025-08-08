# This file is a reconstruction based on project documentation.
# It is intended to restore functionality after accidental deletion.

from neo4j import GraphDatabase
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.database = database
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test a simple query to confirm connectivity
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def is_connected(self) -> bool:
        return self.driver is not None

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.driver is None:
            logger.error("Neo4j driver not initialized.")
            return []
        if parameters is None:
            parameters = {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            # Convert to plain dicts
            return [dict(r) for r in result]

    # The following methods were previously stubs; provide minimal, working implementations.

    def create_entity(self, label, properties):
        logger.warning("create_entity is a stub and not fully implemented.")
        return {}

    def create_relation(self, from_node, to_node, relation_type, properties):
        logger.warning("create_relation is a stub and not fully implemented.")
        return {}

    def update_entity(self, node_id, properties):
        logger.warning("update_entity is a stub and not fully implemented.")
        return {}

    def delete_entity_by_name(self, name):
        logger.warning("delete_entity_by_name is a stub and not fully implemented.")
        return False

    def find_entities_by_properties(self, label, properties):
        logger.warning("find_entities_by_properties is a stub and not fully implemented.")
        return []

    def find_shortest_path(self, start_node, end_node):
        logger.warning("find_shortest_path is a stub and not fully implemented.")
        return []

    def find_paths_between(self, start_node, end_node):
        logger.warning("find_paths_between is a stub and not fully implemented.")
        return []

    def find_connections_between_topics(self, topic1, topic2):
        logger.warning("find_connections_between_topics is a stub and not fully implemented.")
        return []

    def find_influential_nodes(self, label):
        logger.warning("find_influential_nodes is a stub and not fully implemented.")
        return []

    def recommend_related_technologies(self, tech_name):
        logger.warning("recommend_related_technologies is a stub and not fully implemented.")
        return []

    # Added: Stats and labels APIs used by Streamlit page
    def get_database_stats(self) -> Dict[str, Any]:
        if self.driver is None:
            return {"node_count": 0, "relationship_count": 0, "label_count": 0}
        query = (
            "MATCH (n)\n"
            "RETURN count(n) as node_count,\n"
            "       size([()--() | 1]) as relationship_count,\n"
            "       count(distinct labels(n)) as label_count"
        )
        rows = self.run_query(query)
        return rows[0] if rows else {"node_count": 0, "relationship_count": 0, "label_count": 0}

    def get_node_labels(self) -> List[Dict[str, Any]]:
        if self.driver is None:
            return []
        # Works on Neo4j 5.x
        query = (
            "CALL db.labels() YIELD label\n"
            "RETURN label, count{MATCH (n) WHERE label in labels(n) RETURN n} as count\n"
            "ORDER BY count DESC"
        )
        return self.run_query(query)
