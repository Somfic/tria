use common::prelude::*;

mod components;
pub use components::*;

mod systems;
pub use systems::*;

mod messages;
pub use messages::*;

mod resources;
pub use resources::*;

use crate::{action::Action, state::StateKey};

pub struct GoapPlugin<K, A> {
    _phantom: PhantomData<(K, A)>,
}

impl<K, A> Default for GoapPlugin<K, A> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<K, A> Plugin for GoapPlugin<K, A>
where
    K: StateKey,
    A: Action<K>,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<PlanConfig>()
            .add_message::<ActionStartRequested<K, A>>()
            .add_message::<ActionCompleted<K, A>>()
            .add_systems(
                Update,
                (
                    give_plan::<K, A>,
                    execute_plan::<K, A>,
                    advance_plan::<K, A>,
                )
                    .chain(),
            );
    }
}
