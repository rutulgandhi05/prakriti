import json
from neo4j import GraphDatabase

class DBManager:
    def __init__(self):
        # Initialize the Neo4j driver with AuraDB connection details
        NEO4J = json.load(open('configs/neo4j.json'))
        self.driver = GraphDatabase.driver(NEO4J['NEO4J_URI'], auth=(NEO4J["NEO4J_USERNAME"], NEO4J["NEO4J_PASSWORD"]))

    def close(self):
        """Close the Neo4j database connection."""
        self.driver.close()

    def get_scene_by_id(self, scene_id):
        """
        Retrieve the description and any NPCs in a given scene by scene ID.
        
        Args:
            scene_id (int): The ID of the scene to retrieve.
        
        Returns:
            tuple: A tuple containing the scene description and a list of NPC names.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scene {id: $scene_id})
                OPTIONAL MATCH (s)-[:INCLUDES]->(npc:NPC)
                RETURN s.description AS description, collect(npc.name) AS npcs
                """,
                scene_id=scene_id
            )
            print(result)
            record = result.single()
            if record:
                return record["description"], record["npcs"]
            return None, []

    def get_next_scene_options(self, current_scene_id):
        """
        Retrieve possible next scenes based on the current scene.
        
        Args:
            current_scene_id (int): The ID of the current scene.
        
        Returns:
            list: A list of dictionaries, each containing the next scene's ID and description.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Scene {id: $current_scene_id})-[:LEADS_TO]->(next:Scene)
                RETURN next.id AS id, next.description AS description
                """,
                current_scene_id=current_scene_id
            )
            return [{"id": record["id"], "description": record["description"]} for record in result]

    def create_scene_and_npc_data(self):
        """
        Create initial scene and NPC nodes, along with their relationships, in the database
        if they do not already exist.
        """
        with self.driver.session() as session:
            # Create or ensure Scene nodes exist
            session.run(
                """
                MERGE (s1:Scene {id: 1})
                ON CREATE SET s1.description = "You stand at the edge of a dense forest, with mist drifting between the trees."
                
                MERGE (s2:Scene {id: 2})
                ON CREATE SET s2.description = "A hidden grove filled with vibrant flowers and a calm, crystal-clear pond."
                
                MERGE (s3:Scene {id: 3})
                ON CREATE SET s3.description = "An old, abandoned camp lies here, with a burnt-out fire pit and scattered belongings."
                
                MERGE (s4:Scene {id: 4})
                ON CREATE SET s4.description = "You arrive at ancient ruins, with towering stone pillars and intricate carvings on the walls."
                """
            )
            
            # Create or ensure NPC nodes exist
            session.run(
                """
                MERGE (npc1:NPC {name: "Forest Guardian"})
                ON CREATE SET npc1.description = "A mysterious figure who protects the forest."
                
                MERGE (npc2:NPC {name: "Old Traveler"})
                ON CREATE SET npc2.description = "A wise traveler with stories of distant lands."
                
                MERGE (npc3:NPC {name: "Stone Guardian"})
                ON CREATE SET npc3.description = "An ancient guardian that awakens only for the worthy."
                """
            )
            
            # Create relationships between scenes and NPCs if they don't exist
            session.run(
                """
                MATCH (s1:Scene {id: 1}), (npc1:NPC {name: "Forest Guardian"})
                MERGE (s1)-[:INCLUDES]->(npc1)
                
                MATCH (s2:Scene {id: 2}), (npc2:NPC {name: "Old Traveler"})
                MERGE (s2)-[:INCLUDES]->(npc2)
                
                MATCH (s4:Scene {id: 4}), (npc3:NPC {name: "Stone Guardian"})
                MERGE (s4)-[:INCLUDES]->(npc3)
                """
            )
            
            # Create scene transitions only if they don't already exist
            session.run(
                """
                MATCH (s1:Scene {id: 1}), (s2:Scene {id: 2}), (s3:Scene {id: 3}), (s4:Scene {id: 4})
                MERGE (s1)-[:LEADS_TO]->(s2)
                MERGE (s1)-[:LEADS_TO]->(s3)
                MERGE (s2)-[:LEADS_TO]->(s4)
                MERGE (s3)-[:LEADS_TO]->(s4)
                """
            )
            print("Scene, NPC, and relationship data created successfully (if not already existing).")

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

