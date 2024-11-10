# db_manager.py
from neo4j import GraphDatabase

class DBManager:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="prakriti@game"):
        # Connect to the Neo4j database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Close the Neo4j connection
        self.driver.close()

    def load_initial_data(self, data):
        """
        Load initial scenes and NPCs into the Neo4j graph database.
        
        Args:
            data (dict): A dictionary containing scenes and relationships.
        """
        with self.driver.session() as session:
            # Load scenes and NPCs
            for scene in data["scenes"]:
                session.run(
                    """
                    MERGE (s:Scene {id: $id, context: $context})
                    MERGE (npc:NPC {name: $npc_name})
                    MERGE (s)-[:INCLUDES]->(npc)
                    """,
                    id=scene["id"],
                    context=scene["context"],
                    npc_name=scene["npc"]
                )
            
            # Load relationships between scenes
            for relationship in data["relationships"]:
                session.run(
                    """
                    MATCH (s1:Scene {id: $from_id})
                    MATCH (s2:Scene {id: $to_id})
                    MERGE (s1)-[:LEADS_TO]->(s2)
                    """,
                    from_id=relationship["from"],
                    to_id=relationship["to"]
                )

    def get_scene_by_id(self, scene_id):
        """
        Retrieve a scene's context based on its ID.
        
        Args:
            scene_id (int): The ID of the scene to retrieve.
        
        Returns:
            str: The context of the scene.
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Scene {id: $scene_id}) RETURN s.context AS context",
                scene_id=scene_id
            )
            record = result.single()
            return record["context"] if record else None

    def get_next_scene_options(self, current_scene_id):
        """
        Retrieve possible next scenes based on the current scene.
        
        Args:
            current_scene_id (int): The ID of the current scene.
        
        Returns:
            list of dict: List of possible next scenes with ID and context.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scene {id: $current_scene_id})-[:LEADS_TO]->(next:Scene)
                RETURN next.id AS id, next.context AS context
                """,
                current_scene_id=current_scene_id
            )
            return [{"id": record["id"], "context": record["context"]} for record in result]
