## Deep Grid Project

  We explore to use deep reinforcement learning methods for emergency control in power grid system.  

### Modification of Gym source code  

Note that we need to change the source code of OpenAI Gym Baseline models because To modify the source code,   

```  
cd your_path_to_anaconda_directory/envs/py3ml/lib/python3.6/site-packages/baselines/deepq
```
eg, mine will be  
```  
cd ~/anaconda/envs/py3ml/lib/python3.6/site-packages/baselines/deepq
```

Subsequently you need to modify the ```learn()``` function in ```simple.py``` script. 
First, add ```trained_model = none``` to the funtion argument list.  Then, replace lines of 229 to 255 by following codes.  

```
    episode_rewards = [0.0]
    saved_mean_reward = None
    obs, starttime,durationtime = env.reset()

    #i = 0
    #noise = 0.01 * np.random.randn(4,8,301)
    #np.save("./noise", noise)
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            action = act(np.array(obs)[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs #* (1 + noise[:,:,i].flatten())
            #i += 1

            episode_rewards[-1] += rew
            if done:
                #i = 0
                obs, starttime, durationtime = env.reset()
                episode_rewards.append(0.0)
```

Alternatively, you can simply use the ```simple.py``` in the root directory of this repo to replace the same file in the directory <our_path_to_anaconda_directory>/envs/py3ml/lib/python3.6/site-packages/baselines/deepq
But you will be at risk of potential compatibility issue with newer version of OpenAI Gym, as we may not test it against future versions of OpenAI Gym. If you have confront such an issue, please report to us.

Remember to remove the original complied cache file as follows: 

```
cd your\_path\_to\_simple.py\_file/__pycache__/
rm simple.cpython-36.pyc
```
