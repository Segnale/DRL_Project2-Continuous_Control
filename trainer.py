import torch
from train_ppo import train
from unity_env import UnityEnv
from agent_ppo_bkup import Agent
from model import Policy

if __name__ == '__main__':
    #pylint: disable=invalid-name
    iterations = 500
    gamma = 0.99
    timesteps = 100
    ratio_clip = 0.2
    batch_size = int(10*20)
    epochs = 10
    gradient_clip = 10.0
    lrate = 1e-4
    log_each = 10
    beta = 0.01
    gae_tau = 0.95
    decay_steps = None
    solved = 30.0
    out_file = 'saved/ppo.ckpt'
    core = [100,100]

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    env = UnityEnv(env_file='Reachers_Windows_x86_64/Reacher.exe', no_graphics=False)
    policy = Policy(env.state_size, env.action_size, core).to(device)

    print("\nRunning on: ", device, "\n")

    a = Agent(
        env,
        policy,
        nsteps=timesteps,
        gamma=gamma,
        epochs=epochs,
        nbatchs =batch_size,
        ratio_clip=ratio_clip,
        lrate=lrate,
        gradient_clip=gradient_clip,
        beta=beta,
        gae_tau=gae_tau
    )

    train(a, iterations=iterations, log_each=log_each, 
          solved=solved, out_file=out_file)