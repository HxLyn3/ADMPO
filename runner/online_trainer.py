import numpy as np
from tqdm import tqdm

from agent import AGENT
from buffer import BUFFER
from components.static_fns import STATICFUNC

from .base_trainer import BASETrainer

class ONTrainer(BASETrainer):
    """ online MBRL trainer """

    def __init__(self, args):
        super(ONTrainer, self).__init__(args)

        # init armpo agent
        task = args.env_name.split('-')[0]
        static_fn = STATICFUNC[task.lower()]
        self.agent = AGENT["armpo"](
            obs_shape=args.obs_shape,
            hidden_dims=args.ac_hidden_dims,
            action_dim=args.action_dim,
            action_space=args.action_space,
            static_fn=static_fn,
            max_arm_step=args.max_arm_step,
            actor_freq=args.actor_freq,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            model_lr=args.model_lr,
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            alpha_lr=args.alpha_lr,
            target_entropy=args.target_entropy,
            penalty_coef=args.penalty_coef,
            device=args.device
        )
        self.agent.train()

        # init replay buffer to store environmental data
        self.memory = BUFFER["seq-sample"](
            buffer_size=args.buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )

        # creat memory to store model data
        model_rollout_size = args.rollout_batch_size*args.rollout_schedule[2]
        model_buffer_size = int(model_rollout_size*args.model_retain_steps/args.model_update_interval)
        self.model_memory = BUFFER["vanilla"](
            buffer_size=model_buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )

        # func 4 calculate new rollout length (x->y over steps a->b)
        a, b, x, y = args.rollout_schedule
        self.make_rollout_len = lambda it: int(min(max(x+(it-a)/(b-a)*(y-x), x), y))
        # func 4 calculate new model buffer size
        self.make_model_buffer_size = lambda it: \
            int(args.rollout_batch_size*self.make_rollout_len(it) * \
            args.model_retain_steps/args.model_rollout_interval)

        # other parameters
        self.max_arm_step = args.max_arm_step
        self.model_update_interval = args.model_update_interval
        self.model_rollout_interval = args.model_rollout_interval
        self.rollout_batch_size = args.rollout_batch_size
        self.real_ratio = args.real_ratio
        self.n_steps = args.n_steps
        self.start_learning = args.start_learning
        self.update_interval = args.update_interval
        self.updates_per_step = args.updates_per_step
        self.eval_interval = args.eval_interval
        self.save_interval = args.save_interval

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        # init
        records = {"step": [], "loss": {"model": [], "actor": [], "critic1": [], "critic2": []}, 
            "alpha": [], "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": []}
        obs = self._warm_up()

        model_loss, actor_loss, critic1_loss, critic2_loss, eval_reward = [None]*5
        pbar = tqdm(range(self.n_steps), desc="Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            # update dynamics model
            if it % self.model_update_interval == 0:
                model_loss = self.agent.learn_dynamics_from(self.memory, self.batch_size, max_holdout=500)

            if it % self.model_rollout_interval == 0:
                # update imaginary memory
                new_model_buffer_size = self.make_model_buffer_size(it)
                if self.model_memory.capacity != new_model_buffer_size:
                    new_buffer = BUFFER["vanilla"](
                        buffer_size=new_model_buffer_size,
                        obs_shape=self.model_memory.obs_shape,
                        action_dim=self.model_memory.action_dim
                    )
                    old_transitions = self.model_memory.sample_all()
                    new_buffer.store_batch(**old_transitions)
                    self.model_memory = new_buffer

                # rollout
                init_seq_transitions = self.memory.sample_nstep(self.rollout_batch_size, self.max_arm_step-1)
                rollout_len = self.make_rollout_len(it)
                fake_transitions = self.agent.rollout(init_seq_transitions, rollout_len)
                self.model_memory.store_batch(**fake_transitions)

            # step in env
            action, _ = self.agent.act(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.memory.store(obs, action, reward, next_obs, terminated, truncated)
            # to next state
            obs = next_obs
            if terminated or truncated: obs, _ = self.env.reset()

            # update policy
            if it % self.update_interval == 0:
                for _ in range(int(self.update_interval*self.updates_per_step)):
                    real_sample_size = int(self.batch_size*self.real_ratio)
                    fake_sample_size = self.batch_size - real_sample_size
                    real_batch = self.memory.sample(batch_size=real_sample_size)
                    fake_batch = self.model_memory.sample(batch_size=fake_sample_size)
                    transitions = {key: np.concatenate(
                        (real_batch[key], fake_batch[key]), axis=0) for key in real_batch.keys()}
                    transitions.pop("timeout")
                    learning_info = self.agent.learn(**transitions)
                    actor_loss = learning_info["loss"]["actor"]
                    critic1_loss = learning_info["loss"]["critic1"]
                    critic2_loss = learning_info["loss"]["critic2"]
                    alpha = learning_info["alpha"]

            # evaluate policy
            if it % self.eval_interval == 0:
                episode_rewards = self._eval_policy()
                records["step"].append(it)
                records["loss"]["model"].append(model_loss)
                records["loss"]["actor"].append(actor_loss)
                records["loss"]["critic1"].append(critic1_loss)
                records["loss"]["critic2"].append(critic2_loss)
                records["alpha"].append(alpha)
                records["reward_mean"].append(float(np.mean(episode_rewards)))
                records["reward_std"].append(float(np.std(episode_rewards)))
                records["reward_min"].append(float(np.min(episode_rewards)))
                records["reward_max"].append(float(np.max(episode_rewards)))
                eval_reward = records["reward_mean"][-1]
                
                self.logger.add_scalar("loss/model", model_loss, it)
                self.logger.add_scalar("loss/actor", actor_loss, it)
                self.logger.add_scalar("loss/critic1", critic1_loss, it)
                self.logger.add_scalar("loss/critic2", critic2_loss, it)
                self.logger.add_scalar("alpha", alpha, it)
                self.logger.add_scalar("eval/reward", eval_reward, it)

            pbar.set_postfix(
                alpha=alpha,
                model_loss=model_loss,
                actor_loss=actor_loss, 
                critic1_loss=critic1_loss, 
                critic2_loss=critic2_loss, 
                eval_reward=eval_reward
            )

            # save
            if it % self.save_interval == 0: self._save(records)

        self._save(records)
        self.logger.close()
