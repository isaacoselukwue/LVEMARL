import numpy as np
from pommerman import constants

ACTION_MAP = {
    0: "Stop", 1: "Move Up", 2: "Move Down", 3: "Move Left", 4: "Move Right", 5: "Place Bomb"
}

def get_board_description(board, agent_pos):
    """Creates a textual description of the board from the agent's perspective."""
    descriptions = []
    for r_offset in [-1, 0, 1]:
        for c_offset in [-1, 0, 1]:
            if r_offset == 0 and c_offset == 0: continue
            
            r, c = agent_pos[0] + r_offset, agent_pos[1] + c_offset
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                item = constants.Item(board[r, c])
                if item == constants.Item.Rigid:
                    descriptions.append(f"a rigid wall at ({r}, {c})")
                elif item == constants.Item.Wood:
                    descriptions.append(f"a wooden wall at ({r}, {c})")
                elif constants.Item.AgentDummy.value <= item.value <= constants.Item.Agent3.value:
                     descriptions.append(f"an enemy at ({r}, {c})")
                elif constants.Item.ExtraBomb.value <= item.value <= constants.Item.Kick.value:
                    descriptions.append(f"a power-up at ({r}, {c})")
    return ", ".join(descriptions) if descriptions else "clear surroundings"

def format_prompt(obs, action_probs):
    """
    Translates a raw Pommerman observation dictionary into a detailed prompt for the LLM.
    
    Args:
        obs (dict): The raw observation dictionary from the environment.
        action_probs (np.array): Array of probabilities for each action from the RL policy.

    Returns:
        str: A fully formatted prompt.
    """
    try:
        agent_pos = tuple(obs['position'])
        board_desc = get_board_description(obs['board'], agent_pos)

        top_k_indices = np.argsort(action_probs)[::-1][:3]
        suggestions = []
        for i in top_k_indices:
            if i in ACTION_MAP and action_probs[i] > 0.01:
                suggestions.append(f"- {ACTION_MAP[i]} (Confidence: {action_probs[i]:.0%})")
        
        chosen_action = ACTION_MAP.get(np.argmax(action_probs), "Unknown")

        prompt = f"""You are a strategic advisor for a player in a 4-player combat game. Your goal is to be the last one standing.

**Current Situation:**
- **Your Position:** {agent_pos}.
- **Your Status:** Ammo: {obs['ammo']}, Blast Strength: {obs['blast_strength']}, Can Kick: {obs['can_kick']}.
- **Environment:** You see {board_desc}.

**Your AI policy suggests these actions:**
{chr(10).join(suggestions)}

**Task:** Evaluate the top suggested action, '{chosen_action}'. Is this a safe and effective move to help win the game? Be critical. A bad move is worse than no move. Respond ONLY with a JSON object containing your reasoning and a numerical score from -1.0 (critically bad) to 1.0 (excellent).

Example Response:
{{
  "reasoning": "Placing a bomb is a good move because it can clear a path to the power-up, but it is risky because there is no immediate escape route.",
  "score": 0.2
}}

Your Response:
"""
        return prompt
    except (KeyError, IndexError) as e:
        print(f"Warning: Failed to format prompt due to missing key. Error: {e}")
        return "" # Return empty prompt on failure
