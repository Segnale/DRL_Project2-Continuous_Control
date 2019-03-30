import pickle
import numpy as np

# widget bar to display progress
#!pip install progressbar
import progressbar as pb

def train(agent,
          iterations=1000,
          log_each=10,
          solved=90,
          out_file=None,
          writer=None):
    rewards = []
    last_saved = 0

    widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=iterations).start()

    for it in range(iterations):
        frac = 1.0 - it / (iterations-1)
        agent.step()

        if len(agent.episodes_reward) >= 100:
            r = agent.episodes_reward[:-101:-1]
            rewards.append((agent.steps, min(r), max(r), np.mean(r), np.std(r)))
            if (it+1)%50 ==0 :
                print("Iteration: {0:d}, score: {1:f}".format(it+1,np.mean(r)))

        if (it+1) % log_each == 0:
            summary = ''
            #pylint: disable=line-too-long
            if rewards:
                mean = rewards[-1][3]
                minimum = rewards[-1][1]
                maximum = rewards[-1][2]
                summary = f', Rewards: {mean:.2f}/{rewards[-1][4]:.2f}/{minimum:.2f}/{maximum:.2f} mean/std/min/max'

                if writer is not None:
                    writer.add_scalar('data/score_mean', mean, it+1)
                    writer.add_scalar('data/score_min', minimum, it+1)
                    writer.add_scalar('data/score_max', maximum, it+1)

                if out_file and mean >= solved and mean > last_saved:
                    last_saved = mean
                    agent.save(out_file)
                    summary += " (saved)"

        
            
        timer.update(it+1)

            #print(f"Iteration: {it+1:d}, Episodes: {len(agent.episodes_reward)}, Steps: {agent.steps:d}, lrate: {agent.running_lrate:.2E}, Clip: {agent.ratio_clip:.3f}{summary}")

    timer.finish()
    
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'score': list(zip(*rewards))[3]})
    # plot the score moving avarages to reduce the noise\n",
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(111)
    plt.title("Learning")
    plt.plot(np.arange(len(rewards)), df)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    pickle.dump(rewards, open('rewards.p', 'wb'))