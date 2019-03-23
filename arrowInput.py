import gym
import time
from IPython.display import clear_output

import curses, time

#--------------------------------------
def input_char(message):
    try:
        win = curses.initscr()
        win.addstr(0, 0, message)
        while True:
            ch = win.getch()
            if ch in range(32, 127): break
            time.sleep(0.05)
    except: raise
    finally:
        curses.endwin()
    return chr(ch)
#--------------------------------------

env = gym.make('CartPole-v0')
env.reset()
for _ in range(10000):
    env.render()
    a = input_char('Press 1 for push to left or 2 for push to right:')
    a=int(a)-1
    if (a!=0 and a!=1):
        a=0
    print("action: " + str(a) )
    env.step(a) # take a random action
