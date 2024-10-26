# app.py

import streamlit as st
from game_manager.story_manager import StoryManager

# Initialize StoryManager
story_manager = StoryManager(story_file="config.yaml")

# Streamlit UI Setup
st.set_page_config(page_title="Prakriti", layout="wide")
st.title("Prakriti - An Epic Adventure")
st.subheader("Embark on a journey through Eldoria, a land of magic and mystery!")

# Sidebar Setup for Player Status and Inventory
with st.sidebar:
    st.header("Player Status")
    st.write(f"Name: {story_manager.player.name}")
    st.write(f"Profession: {story_manager.player.profession}")
    st.write(f"Current State: {story_manager.player.current_state}")
    st.write("Inventory:")
    for item in story_manager.player.get_inventory():
        st.write(f"- {item}")
    st.write("Attributes:")
    for attr, value in story_manager.player.get_attributes().items():
        st.write(f"{attr}: {value}")
    st.markdown("---")

# Initialize session state for the current quest if not already set
if "current_quest" not in st.session_state:
    st.session_state['current_quest'] = 0  # Start at Quest 1
    st.session_state['quest_complete'] = False

# Function to display the game intro and move to Quest 1
def start_game():
    st.session_state['current_quest'] = 1  # Set to Quest 1 after intro
    intro_text = story_manager.start_intro()  # Show intro dialogue
    st.write(intro_text)

def handle_quest(quest_id):
    """
    Process player actions and display NPC interactions based on structured responses from StoryManager.
    """
    quest = story_manager.get_quest(quest_id)
    
    # Ask for player input until the quest is complete
    player_input = st.text_input(f"Quest {quest_id}: What will you do?", key=f"input_{quest_id}")
    
    if st.button(f"Submit Action", key=f"submit_{quest_id}"):
        # Process the player's input and retrieve the quest response
        quest_data = story_manager.play_quest(quest_id, player_input)
        
        for data in quest_data:
            # Display generated image with a loading message if image generation is available
            if data["image"]:
                with st.spinner("Generating scene..."):
                    st.image(data["image"], use_column_width="auto")
            
            # Display the NPC name and dialogue
            st.write(f"**{data['npc']} says:** {data['dialogue']}")

        # Check if the quest is complete and handle progression
        if quest.status == "completed":
            st.write(f"Quest {quest_id} is complete!")
            st.session_state['quest_complete'] = True
            if quest_id < 3:  # Move to the next quest if it's not the final quest
                st.session_state['current_quest'] += 1
                st.session_state['quest_complete'] = False  # Reset for next quest
            else:
                st.write("Congratulations! You have completed all the quests.")
                st.session_state['current_quest'] = 0  # Reset to the intro after completion

# Start the game with a "Play" button if at the intro stage
if st.session_state['current_quest'] == 0:
    if st.button("Start Your Adventure"):
        start_game()

# Handle quests sequentially if a quest is active
elif st.session_state['current_quest'] > 0:
    st.header(f"Quest {st.session_state['current_quest']}")
    handle_quest(st.session_state['current_quest'])
