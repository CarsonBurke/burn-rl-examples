use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::optim::AdamWConfig;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn_rl::agent::{PPOModel, PPOOutput, PPOTrainingConfig, PPO};
use burn_rl::base::{Action, Agent, ElemType, Environment, Memory, Model, State};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    // Replaces the first dense layer
    conv: Conv1d<B>,

    // Actor / Critic heads remain linear
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> Net<B> {
    /// `input_size` is the feature length of the flat state vector.
    /// `dense_size` becomes the Conv1d out_channels (feature channels).
    /// `kernel_size` controls receptive field (odd numbers are convenient).
    pub fn new(input_size: usize, dense_size: usize, output_size: usize, kernel_size: usize) -> Self {

        // Kaiming/He is a good default for SiLU
        let initializer = Initializer::KaimingUniform { gain: 1.0, fan_out_only: true, };

        Self {
            conv: Conv1dConfig::new(/* in_channels */ 1, /* out_channels */ dense_size, kernel_size)
                // default: stride=1, padding=0; we don't depend on output length thanks to GAP
                .init(&Default::default()),

            linear_actor: LinearConfig::new(dense_size, output_size)
                .with_initializer(initializer.clone())
                .init(&Default::default()),

            linear_critic: LinearConfig::new(dense_size, 1)
                .with_initializer(initializer)
                .init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        // (B, F) -> (B, 1, F)
        let x = input.unsqueeze_dim(1);
        // (B, C, L_out)
        let x = silu(self.conv.forward(x));
        // Global average over the temporal axis -> (B, C)
        let x = x.mean_dim(2).squeeze(2);

        let policies = softmax(self.linear_actor.forward(x.clone()), 1);
        let values = self.linear_critic.forward(x);
        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.unsqueeze_dim(1);
        let x = silu(self.conv.forward(x));
        let x = x.mean_dim(2).squeeze(2);
        softmax(self.linear_actor.forward(x), 1)
    }
}

impl<B: Backend> PPOModel<B> for Net<B> {}

// ======= training harness remains the same, with a kernel_size argument =======
#[allow(unused)]
const MEMORY_SIZE: usize = 512;
const DENSE_SIZE: usize = 128;
const KERNEL_SIZE: usize = 4; // odd works well with SAME-ish behavior if you later add padding

type MyAgent<E, B> = PPO<E, B, Net<B>>;

#[allow(unused)]
pub fn run<E: Environment, B: AutodiffBackend>(
    num_episodes: usize,
    visualized: bool,
) -> impl Agent<E> {
    let mut env = E::new(visualized);

    let mut model = Net::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
        KERNEL_SIZE,
    );

    let agent = MyAgent::default();
    let config = PPOTrainingConfig {
        batch_size: 64,
        entropy_weight: 0.01,
        learning_rate: 0.001,
        epochs: 4,
        clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        ..Default::default()
    };

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(config.clip_grad.clone())
        .init();
    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();

    for episode in 0..num_episodes {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;

        env.reset();
        while !episode_done {
            let state = env.state();
            if let Some(action) = MyAgent::<E, _>::react_with_model(&state, &model) {
                let snapshot = env.step(action);
                episode_reward += <<E as Environment>::RewardType as Into<ElemType>>::into(snapshot.reward().clone());

                memory.push(
                    state,
                    *snapshot.state(),
                    action,
                    snapshot.reward().clone(),
                    snapshot.done(),
                );

                if memory.len() >= MEMORY_SIZE {
                    println!("Memory limit reached - training model");
                    model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
                    memory.clear();
                }

                episode_duration += 1;
                episode_done = snapshot.done() || episode_duration >= E::MAX_STEPS;
            } else {
                println!("no action selected");
            }
        }

        println!(
            "{{\"episode\": {episode}, \"reward\": {episode_reward:.4}, \"duration\": {episode_duration}}}",
        );
    }

    agent.valid(model)
}
