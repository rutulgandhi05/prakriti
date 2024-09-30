import streamlit as st
from game_manager.story_manager import StoryManager

# Initialize StoryManager and Stable Diffusion Image Generator
story_manager = StoryManager(story_file="config.yaml")

# Streamlit UI Setup
st.title("Prakriti")
st.subheader("Embark on an epic quest in the kingdom of Eridell!")

# Initialize session state for current quest if not already set
if "current_quest" not in st.session_state:
    st.session_state['current_quest'] = 0  # Start at Quest 1

# Function to display the game intro and move to Quest 1
def start_game():
    st.session_state['current_quest'] = 1  # Set to Quest 1 after intro
    intro_text = story_manager.start_intro()  # Show intro dialogue
    st.write(intro_text)


def handle_quest(quest_id):
    quest = story_manager.get_quest(quest_id)
    
    # Repeatedly ask for player input until quest is complete
    while not st.session_state['quest_complete']:
        player_input = st.text_input(f"Quest {quest_id}: What will you do?", key=f"input_{quest_id}")
        
        if st.button(f"Submit", key=f"submit_{quest_id}"):
            # Process the player move and get quest response
            quest_data = story_manager.play_quest(quest_id, player_input)
            
            for data in quest_data:
                st.image(data.image)  # Display the generated image
                st.write(data.dialogue)  # Display the LLM response
            
            # Check if the quest is complete
            if quest.status == "completed":
                st.write(f"Quest {quest_id} is complete!")
                st.session_state['quest_complete'] = True
                if quest_id < 3:  # Move to the next quest if it's not the final quest
                    st.session_state['current_quest'] += 1
                    st.session_state['quest_complete'] = False  # Reset for next quest
                else:
                    st.write("Congratulations! You have completed all the quests.")
                    st.session_state['current_quest'] = 0  # Reset to the intro after completion



# Start the game with a Play button
if st.session_state['current_quest'] == 0:
    if st.button("Play"):
        start_game()

# Run the quests sequentially
elif st.session_state['current_quest'] > 0:
    handle_quest(st.session_state['current_quest'])
