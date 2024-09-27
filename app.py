import streamlit as st
from story_manager import StoryManager

# Initialize StoryManager
story_manager = StoryManager(story_file="config.yaml")

# Streamlit app title
st.title("Eridell: A Quest Through Time")

# Initialize game state in session
if 'game_started' not in st.session_state:
    st.session_state['game_started'] = False
    st.session_state['current_quest'] = 1  # Start with Quest 1

# Button to start the game and display intro
if not st.session_state['game_started']:
    if st.button("Play"):
        st.write(story_manager.start_intro())
        st.session_state['game_started'] = True

# Once the game is started, automatically start quests
if st.session_state['game_started']:
    # Check which quest is currently active
    current_quest = st.session_state['current_quest']
    
    # Automatically start the next quest in sequence
    if current_quest == 1:
        st.write("Starting Quest 1: The Blackmoor Curse")
        story_manager.play_quest(1)
        image_path = "./outputs/quest_1/generated_image.png"
        st.image(image_path, caption="Quest 1: The Blackmoor Curse")
        st.session_state['current_quest'] = 2  # Move to the next quest
    
    elif current_quest == 2:
        st.write("Starting Quest 2: The Mayor’s Secret")
        story_manager.play_quest(2)
        image_path = "./outputs/quest_2/generated_image.png"
        st.image(image_path, caption="Quest 2: The Mayor’s Secret")
        st.session_state['current_quest'] = 3  # Move to the final quest
    
    elif current_quest == 3:
        st.write("Starting Quest 3: The Demon’s Reckoning")
        story_manager.play_quest(3)
        image_path = "./outputs/quest_3/generated_image.png"
        st.image(image_path, caption="Quest 3: The Demon’s Reckoning")
        st.session_state['current_quest'] = None  # Mark the game as finished
    
    # Display the input box for player actions
    player_input = st.text_input("Enter your action:", "")
    
    if st.button("Submit Action"):
        # Use the player's input and pass it to the LLM for response
        if player_input:
            current_quest_name = f"Quest {current_quest}"
            llm_response = story_manager.llm.generate_response(current_quest_name, player_input)
            
            # Generate and display updated visual based on LLM response
            story_manager.generate_quest_visual(story_manager.quests[current_quest-1], llm_response)
            image_path = f"./outputs/quest_{current_quest}/generated_image.png"            
            
            st.image(image_path, caption=f"Updated Visual for {current_quest_name}")
            st.write(f"LLM Response: {llm_response}")
    
    # End of the game after Quest 3
    if current_quest is None:
        st.write("All quests completed! The adventure has ended.")
