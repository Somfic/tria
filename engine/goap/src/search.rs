use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use crate::{
    action::{Action, PlanContext},
    goal::Goal,
    plan::Plan,
    plugin::PlanConfig,
    state::{StateKey, WorldState},
};
use common::prelude::*;

type NodeId = usize;

#[derive(Clone, Debug)]
struct SearchNode<K: StateKey> {
    state: WorldState<K>,
    g_cost: f32,
    action_index: Option<usize>,
    parent: Option<NodeId>,
}

#[derive(Copy, Clone)]
struct OpenEntry {
    f_cost: f32,
    node_id: NodeId,
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Eq for OpenEntry {}
impl PartialEq for OpenEntry {
    fn eq(&self, o: &Self) -> bool {
        self.f_cost == o.f_cost
    }
}

type ClosedSet<K> = HashMap<WorldState<K>, f32>;

pub fn plan<K: StateKey, A: Action<K>>(
    start: WorldState<K>,
    goal: &Goal<K>,
    actions: &[A],
    ctx: &PlanContext,
    config: &PlanConfig,
) -> Result<Plan<K, A>, PlanError> {
    let mut arena: Vec<SearchNode<K>> = Vec::new();
    let mut open = BinaryHeap::new();
    let mut closed: HashMap<WorldState<K>, f32> = HashMap::new();

    // Push the start node.
    let h0 = start.distance_to(goal.pattern(), goal.mask()) as f32;
    arena.push(SearchNode {
        state: start,
        g_cost: 0.0,
        action_index: None,
        parent: None,
    });
    open.push(OpenEntry {
        f_cost: h0,
        node_id: 0,
    });

    let mut nodes_expanded = 0;

    while let Some(OpenEntry { node_id, .. }) = open.pop() {
        let node = arena[node_id].clone();

        // Goal test.
        if node.state.matches(goal.pattern(), goal.mask()) {
            return Ok(reconstruct_plan(&arena, node_id, actions));
        }

        // Budget check.
        nodes_expanded += 1;
        if nodes_expanded > config.max_nodes {
            return Err(PlanError::BudgetExceeded);
        }

        // Skip if we've seen this state with a better or equal cost.
        if let Some(&prev_g) = closed.get(&node.state) {
            if prev_g <= node.g_cost {
                continue;
            }
        }
        closed.insert(node.state, node.g_cost);

        // Expand: try every action.
        for (action_idx, action) in actions.iter().enumerate() {
            // Precondition check.
            if !node.state.satisfies(&action.preconditions()) {
                continue;
            }

            let new_state = node.state.apply(&action.effects());

            // Trivial filter: action didn't change state (no progress possible).
            if new_state == node.state {
                continue;
            }

            let edge_cost = action.cost(&node.state, ctx);
            let new_g = node.g_cost + edge_cost;
            let new_h = new_state.distance_to(goal.pattern(), goal.mask()) as f32;
            let new_f = new_g + new_h;

            // Skip if closed with better cost.
            if let Some(&prev_g) = closed.get(&new_state) {
                if prev_g <= new_g {
                    continue;
                }
            }

            arena.push(SearchNode {
                state: new_state,
                g_cost: new_g,
                action_index: Some(action_idx),
                parent: Some(node_id),
            });
            let new_id = arena.len() - 1;
            open.push(OpenEntry {
                f_cost: new_f,
                node_id: new_id,
            });
        }
    }

    Err(PlanError::NoPlan)
}

fn reconstruct_plan<K: StateKey, A: Action<K>>(
    arena: &[SearchNode<K>],
    goal_node_id: NodeId,
    actions: &[A],
) -> Plan<K, A> {
    let mut steps: Vec<A> = Vec::new();
    let mut current = goal_node_id;

    while let Some(action_idx) = arena[current].action_index {
        steps.push(actions[action_idx].clone());
        current = arena[current]
            .parent
            .expect("non-start node must have parent");
    }

    steps.reverse(); // we collected from goal to start, flip it

    Plan::new(steps)
}

#[derive(Debug, Clone)]
pub enum PlanError {
    NoPlan,
    BudgetExceeded,
    InvalidGoal,
}

#[cfg(test)]
mod tests {
    use crate::{
        plugin::PlanConfig,
        state::{Effect, Precondition},
    };

    use super::*;

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    #[repr(u8)]
    enum TestStateKey {
        HasGun = 0,
        HasAmmo = 1,
        GunLoaded = 2,
        InCover = 3,
        NearGun = 4,
        NearAmmo = 5,
        NearCover = 6,
        TargetDead = 7,
    }

    impl StateKey for TestStateKey {
        fn bit_position(self) -> u8 {
            self as u8
        }
        fn all() -> &'static [Self] {
            &[
                Self::HasGun,
                Self::HasAmmo,
                Self::GunLoaded,
                Self::InCover,
                Self::NearGun,
                Self::NearAmmo,
                Self::NearCover,
                Self::TargetDead,
            ]
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    enum TestAction {
        MoveToGun,
        PickupGun,
        MoveToAmmo,
        PickupAmmo,
        LoadGun,
        MoveToCover,
        EnterCover,
        Shoot,
        RushAttack,
    }

    impl Action<TestStateKey> for TestAction {
        fn preconditions(&self) -> Precondition<TestStateKey> {
            use TestStateKey::*;
            match self {
                TestAction::MoveToGun => Precondition::new(),
                TestAction::PickupGun => Precondition::new()
                    .requires(NearGun, true)
                    .requires(HasGun, false),
                TestAction::MoveToAmmo => Precondition::new(),
                TestAction::PickupAmmo => Precondition::new()
                    .requires(NearAmmo, true)
                    .requires(HasAmmo, false),
                TestAction::LoadGun => Precondition::new()
                    .requires(HasGun, true)
                    .requires(HasAmmo, true)
                    .requires(GunLoaded, false),
                TestAction::MoveToCover => Precondition::new(),
                TestAction::EnterCover => Precondition::new()
                    .requires(NearCover, true)
                    .requires(InCover, false),
                TestAction::Shoot => Precondition::new()
                    .requires(HasGun, true)
                    .requires(GunLoaded, true)
                    .requires(InCover, true),
                TestAction::RushAttack => Precondition::new(),
            }
        }

        fn effects(&self) -> Effect<TestStateKey> {
            use TestStateKey::*;
            match self {
                TestAction::MoveToGun => Effect::new().sets(NearGun, true),
                TestAction::PickupGun => Effect::new().sets(HasGun, true),
                TestAction::MoveToAmmo => Effect::new().sets(NearAmmo, true),
                TestAction::PickupAmmo => Effect::new().sets(HasAmmo, true),
                TestAction::LoadGun => Effect::new().sets(GunLoaded, true).sets(HasAmmo, false),
                TestAction::MoveToCover => Effect::new().sets(NearCover, true),
                TestAction::EnterCover => Effect::new().sets(InCover, true),
                TestAction::Shoot => Effect::new().sets(TargetDead, true),
                TestAction::RushAttack => Effect::new().sets(TargetDead, true),
            }
        }

        fn cost(&self, _state: &WorldState<TestStateKey>, _ctx: &PlanContext) -> f32 {
            match self {
                TestAction::RushAttack => 20.0,
                _ => 1.0,
            }
        }
    }

    fn shooter_action_set() -> Vec<TestAction> {
        vec![
            TestAction::MoveToGun,
            TestAction::PickupGun,
            TestAction::MoveToAmmo,
            TestAction::PickupAmmo,
            TestAction::LoadGun,
            TestAction::MoveToCover,
            TestAction::EnterCover,
            TestAction::Shoot,
        ]
    }

    #[test]
    fn planner_finds_full_chain() {
        let actions = shooter_action_set();
        let start = WorldState::<TestStateKey>::new();
        let goal = Goal::<TestStateKey>::new("kill_target").wants(TestStateKey::TargetDead, true);
        let ctx = PlanContext {};
        let config = PlanConfig::default();

        let plan = plan(start, &goal, &actions, &ctx, &config).expect("planner should find a plan");

        // The plan must end with Shoot.
        assert_eq!(plan.steps.last(), Some(&TestAction::Shoot));

        // All required precursor actions must appear before Shoot.
        let shoot_pos = plan
            .steps
            .iter()
            .position(|n| *n == TestAction::Shoot)
            .unwrap();
        let names_before: &[TestAction] = &plan.steps[..shoot_pos];

        for required in [
            TestAction::PickupGun,
            TestAction::PickupAmmo,
            TestAction::LoadGun,
            TestAction::EnterCover,
        ] {
            assert!(
                names_before.contains(&required),
                "expected {required:?} to appear before Shoot in plan: {:?}",
                names_before
            );
        }

        // Ordering constraints. The planner can interleave but these must hold:
        let pos = |name: TestAction| plan.steps.iter().position(|n| *n == name).unwrap();
        assert!(
            pos(TestAction::PickupGun) < pos(TestAction::LoadGun),
            "PickupGun must come before LoadGun: {:?}",
            plan.steps
        );
        assert!(
            pos(TestAction::PickupAmmo) < pos(TestAction::LoadGun),
            "PickupAmmo must come before LoadGun: {:?}",
            plan.steps
        );
        assert!(
            pos(TestAction::EnterCover) < pos(TestAction::Shoot),
            "EnterCover must come before Shoot: {:?}",
            plan.steps
        );
        assert!(
            pos(TestAction::LoadGun) < pos(TestAction::Shoot),
            "LoadGun must come before Shoot: {:?}",
            plan.steps
        );

        // Move actions must come before their corresponding pickup/enter.
        assert!(pos(TestAction::MoveToGun) < pos(TestAction::PickupGun));
        assert!(pos(TestAction::MoveToAmmo) < pos(TestAction::PickupAmmo));
        assert!(pos(TestAction::MoveToCover) < pos(TestAction::EnterCover));
    }

    #[test]
    fn planner_returns_no_plan_for_unreachable_goal() {
        // Empty action set, non-trivial goal.
        let actions: Vec<TestAction> = vec![];
        let start = WorldState::<TestStateKey>::new();
        let goal = Goal::<TestStateKey>::new("kill_target").wants(TestStateKey::TargetDead, true);
        let ctx = PlanContext {};
        let config = PlanConfig::default();

        let result = plan(start, &goal, &actions, &ctx, &config);
        assert!(matches!(result, Err(PlanError::NoPlan)));
    }

    #[test]
    fn planner_returns_empty_plan_when_goal_already_satisfied() {
        let actions = shooter_action_set();
        let mut start = WorldState::<TestStateKey>::new();
        start.set(TestStateKey::TargetDead, true);

        let goal = Goal::<TestStateKey>::new("kill_target").wants(TestStateKey::TargetDead, true);
        let ctx = PlanContext {};
        let config = PlanConfig::default();

        let plan = plan(start, &goal, &actions, &ctx, &config)
            .expect("planner should succeed with empty plan");
        assert_eq!(
            plan.steps.len(),
            0,
            "expected empty plan when goal already satisfied, got {:?}",
            plan.steps
        );
    }

    #[test]
    fn planner_returns_budget_exceeded_when_search_too_constrained() {
        let actions = shooter_action_set();
        let start = WorldState::<TestStateKey>::new();
        let goal = Goal::<TestStateKey>::new("kill_target").wants(TestStateKey::TargetDead, true);
        let ctx = PlanContext {};

        let config = PlanConfig {
            max_nodes: 3,
            ..PlanConfig::default()
        };

        let result = plan(start, &goal, &actions, &ctx, &config);
        assert!(
            matches!(result, Err(PlanError::BudgetExceeded)),
            "expected BudgetExceeded, got {:?}",
            result
        );
    }

    #[test]
    fn planner_picks_cheaper_path_when_alternatives_exist() {
        // Careful path total cost: 8 (eight steps of cost 1).
        // RushAttack: single step, cost 20.
        // A* should pick the careful path.

        let mut actions = shooter_action_set();
        actions.push(TestAction::RushAttack);

        let start = WorldState::<TestStateKey>::new();
        let goal = Goal::<TestStateKey>::new("kill_target").wants(TestStateKey::TargetDead, true);
        let ctx = PlanContext {};
        let config = PlanConfig::default();

        let plan = plan(start, &goal, &actions, &ctx, &config).expect("planner should find a plan");

        assert!(
            !plan.steps.contains(&TestAction::RushAttack),
            "planner should have avoided RushAttack, got plan: {:?}",
            plan.steps
        );
        assert_eq!(
            plan.steps.last(),
            Some(&TestAction::Shoot),
            "expected the careful path ending in Shoot, got plan: {:?}",
            plan.steps
        );
    }
}
