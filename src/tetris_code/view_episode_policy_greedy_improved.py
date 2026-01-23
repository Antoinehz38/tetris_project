import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
import src.tetris_code.policies_improved as policies_improved

if __name__ == "__main__":
    #create the environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
    #initialize the environment including the RNG
    env.reset()
    #keep track of episode end, total reward and time
    terminated = False
    total_reward = 0
    t = 0
    #loop until the game terminates i.e. the bricks reach to top of the screen
    while not terminated:
        t += 1
        #render the current state of the game
        env.render()
        #select action according to the improved greedy policy
        action = policies_improved.policy_lookahead(env)
        #perform action, observe the reward and the next state
        observation, reward, terminated, truncated, info = env.step(action)
        #add the reward to the sum of rewards so far
        key = cv2.waitKey(10) 
        
        total_reward = reward + total_reward
        #display the current score in the terminal
        print("Score:",total_reward,end='\r', flush=True)
    #when game terminates output the final score i.e. the sum of rewards across the episode
    print(f"\nGame Over! Final score: {total_reward}")
