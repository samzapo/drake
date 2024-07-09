#include "drake/planning/trajectory_optimization/parallel_generic_direct_transcription.h"

#include <future>
#include <iostream>
#include <semaphore>

// FIXME(samzapo): maybe get this value elsewhere, must be a constexpr (i.e.
// `std::thread::hardware_concurrency()` is not sufficient).
#ifndef MAX_HARDWARE_CONCURRENCY
#define MAX_HARDWARE_CONCURRENCY 64
#endif

namespace drake {
namespace planning {
namespace trajectory_optimization {

namespace {

template <typename Derived>
std::ostream& operator<<(std::ostream& out,
                         const Eigen::MatrixBase<Derived>& m) {
  out << "(" << m.rows() << "x" << m.cols() << "), (norm=" << m.norm() << "), ";
  const bool is_vector = m.cols() == 1 || m.rows() == 1;
  if (!is_vector) out << std::endl;
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      out << m(i, j) << " ";
    }
    if (!is_vector) out << std::endl;
  }

  return out;
}

// Implements a constraint on the defect between the state variables
// advanced for one update interval and the decision variable representing the
// next state.
class DirectTranscriptionConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DirectTranscriptionConstraint)

  // @param system A pointer the System.
  // @param construct_simulator_fn A function that constructs a simulator for
  //   the system.
  // @param set_state_fn A function that sets relevant "state" values in a
  //   System's context, from a vector.
  // @param set_input_fn A function that sets relevant "input" values among a
  //   System's input ports, from a vector.
  // @param get_state_fn A function that gets relevant "state" values among a
  //   System's contect, and returns a vector.
  // @param num_states the integer size of the state vector being optimized.
  // @param num_inputs the integer size of the input vector being optimized.
  // @param evaluation_time The time along the trajectory at which this
  //   constraint is evaluated.
  // @param fixed_time_step Defines the fixed duration over which to integrate
  //   the System's state when evaluating the constraint.
  DirectTranscriptionConstraint(
      const ConstructSimulatorFunction& construct_simulator_fn,
      const SetContextFunction& set_state_fn,
      const SetContextFunction& set_input_fn,
      const GetStateFunction& get_state_fn, int num_states /* m */,
      int num_inputs /* n */, double evaluation_time, double fixed_time_step)
      : Constraint(num_states, num_inputs + 2 * num_states,
                   VectorX<double>::Zero(num_states),
                   VectorX<double>::Zero(num_states)),
        set_state_fn_(set_state_fn),
        set_input_fn_(set_input_fn),
        get_state_fn_(get_state_fn),
        num_states_(num_states),
        num_inputs_(num_inputs),
        construct_simulator_fn_(construct_simulator_fn),
        fixed_time_step_(fixed_time_step),
        evaluation_time_(evaluation_time) {}

  ~DirectTranscriptionConstraint() override = default;

 protected:
  void EvaluateConstraint(const Eigen::Ref<const VectorX<double>>& x,
                          VectorX<double>* y) const {
    // Extract our input variables:
    const double h = fixed_time_step_;
    const auto input = x.head(num_inputs_);
    const auto state = x.segment(num_inputs_, num_states_);
    const auto next_state = x.tail(num_states_);

    auto simulator = construct_simulator_fn_();
    auto& context = simulator->get_mutable_context();
    DRAKE_ASSERT(x.size() == num_inputs_ + (2 * num_states_));

    // TODO: Set to correct time.
    context.SetTime(evaluation_time_);
    set_state_fn_(&context, state);
    set_input_fn_(&context, input);
    simulator->Initialize();

    try {
      simulator->AdvanceTo(evaluation_time_ + h);
    } catch (const std::runtime_error&) {
      log()->critical(
          "A simulation run terminated early: @t={}s, which is {}s after start "
          "time, t={}s",
          context.get_time(), context.get_time() - evaluation_time_,
          evaluation_time_);
    }
    // Calculate constraint value.
    *y = next_state - get_state_fn_(context);
  }

  // Performs a "forward differnence" finite differencing of the derivative of
  // the constraint, y, w.r.t. the decision variables, x. Note: All constraint
  // evalutations are performed concurrently.
  void EvaluateConstraintWithFiniteDiffDerivatives(
      const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y_in,
      bool calc_derivatives) const {
    // FIXME(samzapo): This counter is static, it controls the number of
    // threads that can be started in all calls made to this function.
    static std::counting_semaphore<MAX_HARDWARE_CONCURRENCY> usable_threads{
        std::thread::hardware_concurrency()};

    const auto x_t = math::ExtractValue(x);

    // Evaluate constraint at x.
    usable_threads.acquire();
    std::future<VectorX<double>> y_t_result =
        std::async(std::launch::async, [&]() {
          VectorX<double> y_t;
          EvaluateConstraint(x_t, &y_t);
          usable_threads.release();
          return y_t;
        });

    // Calc the derivative of the constraint value w.r.t. x.
    if (calc_derivatives) {
      const int m = y_in->rows();
      const int n = x_t.rows();

      std::vector<std::future<VectorX<double>>> y_prime_result(n);

      const double dx = 1e-6;
      for (int j = 0; j < n; ++j) {
        usable_threads.acquire();
        // Note: `std::async` returns a `std::promise` that will populate the
        // `std::future` when the thread rejoins.
        y_prime_result[j] = std::async(std::launch::async, [&, j]() {
          VectorX<double> x_prime = x_t;
          x_prime[j] += dx;  // perturb by some small delta.

          VectorX<double> y_prime;
          EvaluateConstraint(x_prime, &y_prime);
          usable_threads.release();
          return y_prime;
        });
      }

      // Wait on results (note: 'std::future::get()' calls block until result is
      // available to this thread).

      // Get nominal constraint value.
      const VectorX<double> y_t = y_t_result.get();

      // Get perturbed constraint values and calculate derivative.
      MatrixX<double> dy_dx(m, n);
      for (int j = 0; j < n; ++j)
        dy_dx.col(j) = (y_prime_result[j].get() - y_t) * (1. / dx);

      // std::cout << dy_dx << std::endl;

      // Assign output.
      auto& y = *y_in;
      y = y_t;
      for (int i = 0; i < m; ++i) y(i).derivatives() = dy_dx.row(i);
    } else {
      const VectorX<double> y_t = y_t_result.get();
      auto& y = *y_in;
      y = y_t;
    }
  }

  // The format of the input to the eval() function is a vector containing {h,
  // input, state, next_state}.
  void DoEval(const Eigen::Ref<const VectorX<double>>& x,
              VectorX<double>* y) const override {
    AutoDiffVecXd y_t;
    EvaluateConstraintWithFiniteDiffDerivatives(x.cast<AutoDiffXd>(), &y_t,
                                                false);
    *y = math::ExtractValue(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    EvaluateConstraintWithFiniteDiffDerivatives(x, y, true);
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::logic_error(
        "DirectTranscriptionConstraint does not support symbolic evaluation.");
  }

  bool may_evaluate_in_parallel() const override { return true; }

 private:
  const SetContextFunction set_state_fn_;
  const SetContextFunction set_input_fn_;
  const GetStateFunction get_state_fn_;
  const int num_states_{0};
  const int num_inputs_{0};
  const ConstructSimulatorFunction construct_simulator_fn_;
  const double fixed_time_step_;
  const double evaluation_time_;
};
}  // namespace

ParallelGenericDirectTranscription::ParallelGenericDirectTranscription(
    const ConstructSimulatorFunction& construct_simulator_fn,
    const SetContextFunction& set_state_fn,
    const SetContextFunction& set_input_fn,
    const GetStateFunction& get_state_fn, int num_states, int num_inputs,
    int num_time_samples, double fixed_time_step)
    : planning::trajectory_optimization::MultipleShooting(
          num_inputs, num_states, num_time_samples, fixed_time_step) {
  // For N-1 time steps, add a constraint which depends on the breakpoint
  // along with the state and input vectors at that breakpoint and the
  // next.
  for (int i = 0; i < N() - 1; i++) {
    // Add the dynamic constraints.
    // Note that these constraints may be evaluated in parallel.
    const double evaluation_time = static_cast<double>(i) * fixed_time_step;
    auto constraint = std::make_shared<DirectTranscriptionConstraint>(
        construct_simulator_fn, set_state_fn, set_input_fn, get_state_fn,
        num_states, num_inputs, fixed_time_step, evaluation_time);
    prog().AddConstraint(constraint, {input(i), state(i), state(i + 1)});
  }
}

trajectories::PiecewisePolynomial<double>
ParallelGenericDirectTranscription::ReconstructInputTrajectory(
    const solvers::MathematicalProgramResult& result) const {
  VectorX<double> times = GetSampleTimes(result);
  std::vector<double> times_vec(N());
  std::vector<Eigen::MatrixXd> inputs(N());

  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    inputs[i] = result.GetSolution(input(i));
  }

  return trajectories::PiecewisePolynomial<double>::ZeroOrderHold(times_vec,
                                                                  inputs);
}

trajectories::PiecewisePolynomial<double>
ParallelGenericDirectTranscription::ReconstructStateTrajectory(
    const solvers::MathematicalProgramResult& result) const {
  VectorX<double> times = GetSampleTimes(result);
  std::vector<double> times_vec(N());
  std::vector<Eigen::MatrixXd> states(N());

  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    states[i] = result.GetSolution(state(i));
  }

  return trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
      times_vec, states);
}

void ParallelGenericDirectTranscription::DoAddRunningCost(
    const symbolic::Expression& g) {
  // Cost = \sum_n g(n,x[n],u[n]) dt
  for (int i = 0; i < N() - 1; i++) {
    prog().AddCost(SubstitutePlaceholderVariables(g * fixed_time_step(), i));
  }
}
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake