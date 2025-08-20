#![recursion_limit = "256"]

use crate::utils::demo_model;
use burn::backend::{Autodiff, NdArray, Vulkan, Wgpu};
use burn_rl::base::ElemType;
use burn_rl::environment::CartPole;

mod dqn;
mod ppo;
mod ppo_convolutional;
mod sac;
mod utils;

type Backend = Autodiff<Vulkan<ElemType>>;
type Env = CartPole;

fn main() {
    let agent = ppo_convolutional::run::<Env, Backend>(512, false);
    // let agent = ppo::run::<Env, Backend>(512, false);

    demo_model::<Env>(agent);
}
