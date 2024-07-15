
#include "drake/planning/trajectory_optimization/parallel_generic_direct_transcription.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/implicit_euler_integrator.h"
#include "drake/systems/analysis/integrator_base.h"
#include "drake/systems/analysis/simulator.h"

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

constexpr size_t kThreeFor3D = 3;
constexpr size_t kSixSpatialDofs = kThreeFor3D * 2;
constexpr size_t kFourForQuaternion = 4;

// TODO(samzapo): Remove these hard coded values after Drake issue #12807 has
// been addressed.
constexpr int kNumFloatingBodyPositions = kThreeFor3D + kFourForQuaternion;
constexpr int kNumFloatingBodyVelocities = kSixSpatialDofs;

template <typename T>
using FloatingBodyStateVector =
    Eigen::Matrix<T, kNumFloatingBodyPositions + kNumFloatingBodyVelocities, 1>;

/// Poses are represented in vector form as a `Vector7` with structure `[qw, qx,
/// qy, qz, rx, ry, rz]` where the `q`s are orientation components and the `r`s
/// are translational components.
template <typename T>
using FloatingBodyPoseVector = Eigen::Matrix<T, kNumFloatingBodyPositions, 1>;

/// Spatial velocities are represented as a `Vector6` with structure `[angular
/// velocity, translational velocity]`.
template <typename T>
using FloatingBodySpatialVelocityVector =
    Eigen::Matrix<T, kNumFloatingBodyVelocities, 1>;

/**
 Creates a 7-dimensional vector stored in the format [qw qx qy qz rx ry rz]
 where q's are unit-quaternion components and r's are translational components.
 @warning The resulting quaternion is a (unit) quaternion with the scalar part
 as the first component.
 */
template <typename T>
FloatingBodyPoseVector<double> ToVector(const math::RigidTransform<double>& X) {
  // Checks if rotation matrix is valid.
  DRAKE_DEMAND(X.rotation().IsValid());

  const Eigen::Quaternion<double> quat = X.rotation().ToQuaternion();

  FloatingBodyPoseVector<double> v;
  v[0] = quat.w();
  v[1] = quat.x();
  v[2] = quat.y();
  v[3] = quat.z();
  v.template tail<kThreeFor3D>() = X.translation();
  return v;
}
/**
 Creates a RigidTransform from a 7-dimensional vector stored in the format [qw
 qx qy qz rx ry rz] where q's are quaternion components and r's are
 translational components. The quaternion components do not need to be
 normalized.
 @param v a vector with quaternion components that need not be normalized.
 @warning The quaternion included in the input vector has the scalar part as the
 first component.
 @warning Converting the rotation components of the RigidTransform to a
 quaternion representation will result in a normalized (unit) quaternion.
 */
template <typename T>
math::RigidTransform<double> ToRigidTransform(
    Eigen::Ref<const VectorX<double>> v) {
  DRAKE_DEMAND(v.size() == kNumFloatingBodyPositions);
  DRAKE_DEMAND(v.allFinite());
  DRAKE_DEMAND(!v.template head<kFourForQuaternion>().isZero());

  const Eigen::Quaternion<double> quat(v[0], v[1], v[2], v[3]);

  // Note that this RotationMatrix constructor does not require the quaternion
  // to be normalized.
  const math::RotationMatrix<double> R(quat);
  math::RigidTransform<double> X(R, v.template tail<kThreeFor3D>());
  return X;
}

std::unique_ptr<multibody::MultibodyPlant<double>> ConstructTestPlant(
    const std::string& body_name,
    double time_step = 0.0 /* continuous time is default */) {
  auto mbp = std::make_unique<multibody::MultibodyPlant<double>>(time_step);

  const Vector3<double> kDefaultGravityVector(0, 0, 0);
  mbp->mutable_gravity_field().set_gravity_vector(kDefaultGravityVector);

  const multibody::ModelInstanceIndex model_instance =
      mbp->HasModelInstanceNamed(body_name)
          ? mbp->GetModelInstanceByName(body_name)
          : mbp->AddModelInstance(body_name);

  const auto M_Bcm = multibody::SpatialInertia<double>::SolidCubeWithDensity(
      1000. /* water density, in kg/m³ */, 0.1 /* length */);

  mbp->AddRigidBody(body_name, model_instance, M_Bcm);

  mbp->Finalize();
  return mbp;
}

GTEST_TEST(ParallelGenericParallelGenericDirectTranscriptionTest,
           FixedTimestepTest) {
  using namespace std::chrono_literals;

  const std::string kBodyName = "cube";
  auto system = ConstructTestPlant(kBodyName);
  auto& plant = *system;

  const auto& body = plant.GetBodyByName(kBodyName);

  const auto construct_simulator_fn = [&]() {
    auto simulator = std::make_unique<systems::Simulator<double>>(plant);

    const double kMinTimeStep = std::chrono::duration<double>(1ns).count();
    const double kMaxTimeStep = std::chrono::duration<double>(1s).count();

    auto& integrator = simulator->template reset_integrator<
        systems::ImplicitEulerIntegrator<double>>();
    integrator.set_maximum_step_size(kMaxTimeStep);
    integrator.set_requested_minimum_step_size(kMinTimeStep);

    return simulator;
  };

  const int kNumStates = kNumFloatingBodyPositions + kNumFloatingBodyVelocities;
  const auto set_state_fn =
      [&](systems::Context<double>* context,
          const Eigen::Ref<const Eigen::VectorXd>& q_v_body) {
        DRAKE_DEMAND(q_v_body.rows() == kNumStates);

        // Get body state from input vector.
        const auto X_WB = ToRigidTransform<double>(
            q_v_body.template head<kNumFloatingBodyPositions>());

        const multibody::SpatialVelocity<double> V_WBo_W(
            q_v_body.template tail<kNumFloatingBodyVelocities>());

        // Get mutable state from plant.
        auto q_v_plant = dynamic_cast<systems::BasicVector<double>&>(
                             context->get_mutable_continuous_state_vector())
                             .get_mutable_value();
        auto q_plant = q_v_plant.template segment<kNumFloatingBodyPositions>(
            body.floating_positions_start());
        auto v_plant = q_v_plant.template segment<kNumFloatingBodyVelocities>(
            plant.num_positions() + body.floating_velocities_start_in_v());

        // Set mutable state.
        q_plant = ToVector<double>(X_WB);

        v_plant = V_WBo_W.get_coeffs();
      };

  const auto get_state_fn =
      [&body](const systems::Context<double>& context) -> Eigen::VectorXd {
    // Get body state from plant.
    const auto& X_WB = body.EvalPoseInWorld(context);
    const auto& V_WBo_W = body.EvalSpatialVelocityInWorld(context);

    // Construct output vector.
    FloatingBodyStateVector<double> q_v_body;
    auto q_body = q_v_body.template head<kNumFloatingBodyPositions>();
    auto v_body = q_v_body.template tail<kNumFloatingBodyVelocities>();
    DRAKE_DEMAND(q_v_body.rows() == kNumStates);

    // Set body state in output vector.
    q_body = ToVector<double>(X_WB);
    v_body = V_WBo_W.get_coeffs();

    return q_v_body;
  };

  constexpr int kNumInputs = kSixSpatialDofs;
  const auto set_input_fn = [&](systems::Context<double>* context,
                                const Eigen::Ref<const Eigen::VectorXd>& u) {
    DRAKE_DEMAND(u.rows() == kNumInputs);
    std::vector<multibody::ExternallyAppliedSpatialForce<double>> forces{
        {
            .body_index = body.index(),
            .p_BoBq_B = Vector3<double>::Zero(),
            .F_Bq_W = multibody::SpatialForce<double>(u),
        },
    };

    plant.get_applied_spatial_force_input_port().FixValue(context, forces);
  };

  const math::RigidTransform<double> X_WB_inital =
      math::RigidTransform<double>::Identity();

  const math::RigidTransform<double> X_WB_final(
      math::RotationMatrix<double>::Identity(),
      Vector3<double>(1., 1., 1.) /* p */);

  FloatingBodyStateVector<double> q_v_initial;
  q_v_initial.template head<kNumFloatingBodyPositions>() =
      ToVector<double>(X_WB_inital);
  q_v_initial.template tail<kNumFloatingBodyVelocities>() =
      FloatingBodySpatialVelocityVector<double>::Zero();

  FloatingBodyStateVector<double> q_v_final;
  q_v_final.template head<kNumFloatingBodyPositions>() =
      ToVector<double>(X_WB_final);
  q_v_final.template tail<kNumFloatingBodyVelocities>() =
      FloatingBodySpatialVelocityVector<double>::Zero();

  const int kNumSegments = 4;
  const int kNumTimeSamples = kNumSegments + 1;
  const int kUpdateInterval = std::chrono::duration<double>(1s).count();

  auto multiple_shooting = std::make_unique<ParallelGenericDirectTranscription>(
      construct_simulator_fn, set_state_fn, set_input_fn, get_state_fn,
      kNumStates, kNumInputs, kNumTimeSamples, kUpdateInterval);

  auto& prog = multiple_shooting->prog();

  const solvers::VectorXDecisionVariable& u = multiple_shooting->input();
  multiple_shooting->AddRunningCost(u.transpose() * u);

  // Set fixed end points.
  prog.AddLinearConstraint(multiple_shooting->initial_state() == q_v_initial);
  prog.AddLinearConstraint(multiple_shooting->final_state() == q_v_final);

  {
    std::vector<double> breaks(kNumTimeSamples);
    std::vector<MatrixX<double>> x_samples(kNumTimeSamples);
    std::vector<MatrixX<double>> u_samples(kNumTimeSamples);
    for (int i = 0; i <= kNumSegments; i++) {
      const double progress =
          (static_cast<double>(i) / static_cast<double>(kNumSegments));
      breaks[i] = kUpdateInterval * static_cast<double>(i);

      // Obviously not going to be a valid seed, due to quaternion.
      x_samples[i] = q_v_initial * (1.0 - progress) + q_v_final * progress;

      Vector<double, kNumInputs> u_sample = Vector<double, kNumInputs>::Zero();
      u_samples[i] = u_sample;
    }

    // Create an initial guess for the state trajectory.
    multiple_shooting->SetInitialTrajectory(
        trajectories::PiecewisePolynomial<double>::ZeroOrderHold(breaks,
                                                                 u_samples),
        trajectories::PiecewisePolynomial<double>::FirstOrderHold(breaks,
                                                                  x_samples));
  }

  log()->critical("Solving Mathematical Program...");

  solvers::IpoptSolver solver;
  solvers::SolverOptions options;
  options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);

  // Sets the default verbosity level for console output. The larger this value
  // the more detailed is the output. The valid range for this integer option is
  // 0 ≤ print_level ≤ 12 and its default value is 5.
  options.SetOption(solver.id(), "print_level", 5);
  options.SetOption(solver.id(), "output_file", "ipopt_logs.txt");
  options.SetOption(solver.id(), "check_derivatives_for_naninf", "yes");

  const auto result =
      solver.Solve(prog, std::nullopt /* initial guess*/, options);
  log()->critical("... Finished Solving Mathematical Program!");

  const auto& solver_details =
      result.template get_solver_details<solvers::IpoptSolver>();

  log()->critical("Status: {}", solver_details.ConvertStatusToString());

  const auto inputs = multiple_shooting->ReconstructInputTrajectory(result);
  const auto states = multiple_shooting->ReconstructStateTrajectory(result);
  const std::vector<double>& breaks = states.get_segment_times();
  const size_t N = breaks.size();

  for (size_t i = 0; i < N; ++i) {
    const double start_time = breaks[i];
    const Vector<double, kNumInputs> input = inputs.value(start_time);
    const Vector<double, kNumStates> state = states.value(start_time);

    std::cout << "@t=" << start_time << std::endl;
    std::cout << "\tu=" << input.transpose() << std::endl;
    std::cout << "\tx=" << state.transpose() << std::endl;
  }

  // Replay the final trajectory in the logs.
  if (result.is_success()) {
    log()->critical("SUCCESS!");

    // Replay the full roll-out.
    auto simulator = construct_simulator_fn();
    auto& context = simulator->get_mutable_context();
    context.SetTime(breaks[0]);
    simulator->Initialize();
    for (size_t i = 0; i < N - 1; ++i) {
      const double start_time = breaks[i];
      const double end_time = breaks[i + 1];

      const Vector<double, kNumInputs> input = inputs.value(start_time);
      set_input_fn(&context, input);

      simulator->AdvanceTo(end_time);
    }
  } else {
    log()->critical("FAILURE!");

    // Replay each segment.
    for (size_t i = 0; i < N - 1; ++i) {
      const double start_time = breaks[i];
      const double end_time = breaks[i + 1];
      auto simulator = construct_simulator_fn();
      auto& context = simulator->get_mutable_context();

      context.SetTime(start_time);

      const Vector<double, kNumInputs> input = inputs.value(start_time);
      const Vector<double, kNumStates> state = states.value(start_time);

      set_input_fn(&context, input);
      set_state_fn(&context, state);

      simulator->Initialize();

      try {
        simulator->AdvanceTo(end_time);
      } catch (const std::exception& e) {
        std::cout << "\t... Segment threw an exception: " << e.what()
                  << std::endl;
      } catch (...) {
        std::cout << "\t... Segment threw an unknown error." << std::endl;
      }
    }
  }

  EXPECT_TRUE(result.is_success());
}

}  // namespace
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake