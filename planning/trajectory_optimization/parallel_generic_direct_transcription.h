#pragma once

#include <memory>
#include <functional>

#include "drake/common/symbolic/decompose.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/planning/trajectory_optimization/multiple_shooting.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/system_symbolic_inspector.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using ConstructSimulatorFunction =
    std::function<std::unique_ptr<systems::Simulator<double>>()>;

using SetContextFunction = std::function<void(
    systems::Context<double>*, const Eigen::Ref<const VectorX<double>>&)>;

using GetStateFunction =
    std::function<VectorX<double>(const systems::Context<double>&)>;

/// ParallelGenericDirectTranscription is perhaps the simplest implementation of
/// a multiple shooting method, where we have decision variables representing
/// the control and input at every sample time in the trajectory, and a fixed
/// integration interval provides the dynamic constraints between those decision
/// variables.
/// @ingroup planning_trajectory
class ParallelGenericDirectTranscription
    : public planning::trajectory_optimization::MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ParallelGenericDirectTranscription)

  /// Constructs the %MathematicalProgram% and adds the transcription
  /// constraints.
  ///
  /// @param construct_simulator_fn A function that constructs a simulator for
  ///   the system.
  /// @param set_state_fn A function that sets relevant values in a System's
  ///   context, from a vector.
  /// @param set_input_fn A function that sets relevant input values among a
  ///   System's input ports, from a vector.
  /// @param get_state_fn A function that gets relevant "state" values among a
  ///   System's contect, and returns a vector.
  /// @param num_states the integer size of the state vector being optimized.
  /// @param num_inputs the integer size of the input vector being optimized.
  /// @param num_time_samples The number of sample times the the trajectory.
  /// @param fixed_time_step Defines the fixed duration over which to integrate
  ///   the System's state when evaluating the constraint.
  ParallelGenericDirectTranscription(
      const ConstructSimulatorFunction& construct_simulator_fn,
      const SetContextFunction& set_state_fn,
      const SetContextFunction& set_input_fn,
      const GetStateFunction& get_state_fn, int num_states, int num_inputs,
      int num_time_samples, double fixed_time_step);

  /// Get the input trajectory at the solution as a ZoH trajectory.
  trajectories::PiecewisePolynomial<double> ReconstructInputTrajectory(
      const solvers::MathematicalProgramResult& result) const override;

  /// Get the state trajectory at the solution as a pChip PiecewisePolynomial.
  trajectories::PiecewisePolynomial<double> ReconstructStateTrajectory(
      const solvers::MathematicalProgramResult& result) const override;

 protected:
  void DoAddRunningCost(const symbolic::Expression& g) override;
};
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake