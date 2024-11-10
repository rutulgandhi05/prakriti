from db_manager import DBManager

db_manager = DBManager(password="your_password")
initial_scene = db_manager.get_scene_by_id(1)
next_scenes = db_manager.get_next_scene_options(1)
db_manager.close()

print(initial_scene)