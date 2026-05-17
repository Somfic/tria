use crate::{
    action::{Action, PlanContext},
    plugin::{
        ActionCompleted, ActionSet, ActionStartRequested, CurrentGoal, CurrentPlan, PerceivedState,
        PlanConfig,
    },
    state::StateKey,
};
use common::prelude::*;

pub fn give_plan<K: StateKey, A: Action<K>>(
    mut commands: Commands,
    agents: Query<
        (
            Entity,
            &PerceivedState<K>,
            &ActionSet<K, A>,
            &CurrentGoal<K>,
        ),
        Without<CurrentPlan<K, A>>,
    >,
    config: Res<PlanConfig>,
) {
    let ctx = PlanContext {};

    for (entity, state, actions, goal) in &agents {
        match crate::plan(state.0, &goal.0, &actions.actions, &ctx, &config) {
            Ok(plan) => {
                commands.entity(entity).insert(CurrentPlan {
                    plan,
                    step_index: 0,
                    step_started: false,
                });
            }
            Err(_err) => {
                // TODO: fire error, couldn't find a plan
            }
        }
    }
}

pub fn execute_plan<K: StateKey, A: Action<K>>(
    mut commands: Commands,
    mut messages: MessageWriter<ActionStartRequested<K, A>>,
    mut agents: Query<(Entity, &PerceivedState<K>, &mut CurrentPlan<K, A>)>,
) {
    for (entity, state, mut current) in &mut agents {
        if current.step_started {
            continue;
        }

        let Some(step) = current.plan.steps.get(current.step_index) else {
            // finished
            commands.entity(entity).remove::<CurrentPlan<K, A>>();
            continue;
        };

        let pre = step.preconditions();
        if !state.0.satisfies(&pre) {
            // conditions no longer satisfied, throw away the plan
            commands.entity(entity).remove::<CurrentPlan<K, A>>();
            continue;
        }

        messages.write(ActionStartRequested::new(entity, step.clone()));
        current.step_started = true;
    }
}

pub fn advance_plan<K: StateKey, A: Action<K>>(
    mut commands: Commands,
    mut reader: MessageReader<ActionCompleted<K, A>>,
    mut agents: Query<&mut CurrentPlan<K, A>>,
) {
    for msg in reader.read() {
        if let Ok(mut current) = agents.get_mut(msg.agent) {
            let expected = current.plan.steps.get(current.step_index);
            if expected != Some(&msg.action) || !current.step_started {
                // ignore stale or unexpected messages
                continue;
            }

            if !msg.success {
                // the action failed, throw away the plan
                commands.entity(msg.agent).remove::<CurrentPlan<K, A>>();
                continue;
            }

            current.step_index += 1;
            current.step_started = false; // mark next step as not yet started

            if current.step_index >= current.plan.steps.len() {
                // finished the last step, remove the plan
                commands.entity(msg.agent).remove::<CurrentPlan<K, A>>();
            }
        }
    }
}
