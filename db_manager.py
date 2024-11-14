import json
import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DBManager:
    def __init__(self):
        # Initialize the Neo4j driver with AuraDB connection details
        NEO4J = json.load(open('configs/neo4j.json'))
        self.driver = GraphDatabase.driver(NEO4J['NEO4J_URI'], auth=(NEO4J["NEO4J_USERNAME"], NEO4J["NEO4J_PASSWORD"]))

    def close(self):
        """Close the Neo4j database connection."""
        self.driver.close()
        
    def get_scene_by_id(self, scene_id):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Scene {id: $scene_id}) RETURN s.description AS description, s.npc AS npc",
                scene_id=scene_id
            )
            record = result.single()
            return (record["description"], record["npc"]) if record else None

    
    def create_scene_and_npc_data(self):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (scene1:Scene {id: 1, description: "A mystical forest.", npc: "Eldara"})
                MERGE (scene2:Scene {id: 2, description: "An ancient ruins.", npc: "Guardian"})
                MERGE (scene1)-[:LEADS_TO]->(scene2)
                """
            )

    def get_next_scene(self, current_scene_id):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Scene {id: $current_scene_id})-[:LEADS_TO]->(next:Scene) RETURN next.id AS id",
                current_scene_id=current_scene_id
            )
            record = result.single()
            return record["id"] if record else None
        

if __name__ == "__main__":
    db_manager = DBManager()
    db_manager.create_scene_and_npc_data()
    print(db_manager.get_scene_by_id(1))
    print(db_manager.get_next_scene(1))
    db_manager.close()