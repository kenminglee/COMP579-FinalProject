import gymnasium as gym
from minigrid.utils.window import Window


def redraw(img):
    img = env.render()
    window.show_img(img)


def reset():
    # env.seed(123)

    obs, _ = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, terminated, truncated, info = env.step(action)
    print("step=%s, reward=%.2f" % (env.step_count, reward))
    print(obs)
    if terminated:
        print("done!")
        print(info)
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left":
        step(env.actions.left)
        return
    if event.key == "right":
        step(env.actions.right)
        return
    if event.key == "up":
        step(env.actions.forward)
        return

    # Spacebar
    # if event.key == " ":
    #     step(env.actions.toggle)
    #     return
    # if event.key == "pageup":
    #     step(env.actions.pickup)
    #     return
    if event.key == " ":  # press space to pick up item
        step(env.actions.pickup)
        return
    # if event.key == "pagedown":
    #     step(env.actions.drop)
    #     return
    #
    # if event.key == "enter":
    #     step(env.actions.done)
    #     return


import store_env  # for registering the env
from store_env.environment.wrapper import make_discrete_env

# env = gym.make("StoreEnv-v1", render_mode="rgb_array")
env = make_discrete_env("StoreEnv-v1", 1, 0, False, None, False, layout="prefer-yellow")()

window = Window("StoreEnv-v1")
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
