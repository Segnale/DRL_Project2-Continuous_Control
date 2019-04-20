from utilities import Plotting
import pickle
import numpy as np

# widget bar to display progress
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
          pb.Bar(), ' ', pb.ETA(), ' | ', pb.DynamicMessage('Score')]
    timer = pb.ProgressBar(widgets=widget, maxval=iterations).start()

    # Score Trend Initialization
    plot = Plotting(
        title ='Learning Process',
        y_label = 'Score',
        x_label = 'Episode #',
        x_range = 250,
    )
    Pscore = []
    plot.show()

    for it in range(iterations):
        frac = 1.0 - it / (iterations-1)
        agent.step()

        if len(agent.episodes_reward) >= 100:
            r = agent.episodes_reward[:-101:-1]
            rewards.append((agent.steps, min(r), max(r), np.mean(r), np.std(r)))
            Pscore.append(np.mean(r))
            plot.Update(list(range(it+1)),Pscore)
        else:
            Pscore.append(np.sum(agent.episodes_reward)/100)
            plot.Update(list(range(it+1)),Pscore)
            
        timer.update(it+1, Score = Pscore[-1])

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