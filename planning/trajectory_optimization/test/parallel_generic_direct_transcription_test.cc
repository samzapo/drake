
#include "drake/planning/trajectory_optimization/parallel_generic_direct_transcription.h"

#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {
namespace {

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
FloatingBodyPoseVector<T> ToVector(const math::RigidTransform<T>& X) {
  // Checks if rotation matrix is valid.
  DR_DEMAND(X.rotation().IsValid());

  const Eigen::Quaternion<T> quat = X.rotation().ToQuaternion();

  FloatingBodyPoseVector<T> v;
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
math::RigidTransform<T> ToRigidTransform(Eigen::Ref<const VectorX<T>> v) {
  DR_DEMAND(v.size() == kNumFloatingBodyPositions);
  DR_DEMAND(v.allFinite());
  DR_DEMAND(!v.template head<kFourForQuaternion>().isZero());

  const Eigen::Quaternion<T> quat(v[0], v[1], v[2], v[3]);

  // Note that this RotationMatrix constructor does not require the quaternion
  // to be normalized.
  const math::RotationMatrix<T> R(quat);
  math::RigidTransform<T> X(R, v.template tail<kThreeFor3D>());
  return X;
}

std::unique_ptr<multibody::MultibodyPlant<double>> ConstructTestPlant(
    const std::string& body_name,
    double time_step = 0.0 /* continuous time is default */) {
  auto mbp = std::make_unique<MultibodyPlant<double>>(time_step);

  const ModelInstanceIndex model_instance =
      mbp->HasModelInstanceNamed(body_name)
          ? mbp->GetModelInstanceByName(body_name)
          : mbp->AddModelInstance(body_name);

  const auto M_Bcm = SpatialInertia<double>::SolidCubeWithDensity(
      1000. /* water density, in kg/m³ */, 1.0 /* length */);

  const RigidBody<T>& rigid_body =
      mbp->AddRigidBody(body_name, model_instance, M_Bcm);

  mbp->Finalize();
  return mbp;
}

GTEST_TEST(ParallelGenericParallelGenericDirectTranscriptionTest,
           FixedTimestepTest) {
  const std::string kBodyName = "cube";
  auto plant_ptr = ConstructTestPlant(kBodyName);
  auto& plant = *plant_ptr;

  auto& body = plant.GetBodyByName(kBodyName);
  const auto construct_simulator_fn = [&]() {
    auto simulator = std::make_unique<systems::Simulator<T>>(plant);

    using namespace std::chrono_literals;
    const double kMinTimeStep = std::chrono::duration<double>(1ms).count();
    const double kMaxTimeStep = std::chrono::duration<double>(1s).count();

    auto& integrator =
        simulator
            ->template reset_integrator<systems::ImplicitEulerIntegrator<T>>();
    integrator.set_target_accuracy(1e-6);
    integrator.set_reuse(true);
    integrator.set_maximum_step_size(kMaxTimeStep);
    integrator.set_requested_minimum_step_size(kMinTimeStep);

    return simulator;
  };

  const auto set_state_fn =
      [&](systems::Context<T>* context,
          const Eigen::Ref<const Eigen::VectorXd>& q_v_body) {
        auto& q_v_plant = context->get_mutable_continuous_state_vector();
        q_v_plant.temaplte segment<kNumFloatingBodyPositions>(
            body.floating_positions_start()) =
            q_v_body.template head<kNumFloatingBodyPositions>();

        q_v_plant.temaplte segment<kNumFloatingBodyVelocities>(
            plant.num_positions() + body.floating_velocities_start_in_v()) =
            q_v_body.template tail<kNumFloatingBodyVelocities>();
      };

  const auto get_state_fn =
      [&body](const systems::Context<T>& context) -> Eigen::VectorXd {
    FloatingBodyStateVector<double> q_v;
    q_v.template head<kNumFloatingBodyPositions>() =
        body.EvalPoseInWorld(context);
    q_v.template tail<kNumFloatingBodyVelocities>() =
        body.EvalSpatialVelocityInWorld(context).get_coeffs();

    return q_v;
  };

  const auto set_input_fn = [&](systems::Context<T>* context,
                                const Eigen::Ref<const Eigen::VectorXd>& u) {
    std::vector<multibody::ExternallyAppliedSpatialForce<double>> forces{
        {
            .body_index = body.index(),
            .p_BoBq_B = Vector3<T>::Zero(),
            .F_Bq_W = multibody::SpatialForce<T>(u),
        },
    };

    plant.get_applied_spatial_force_input_port().FixValue(context, forces);
  };

  math::RigidTransform<double> X_WB_inital =
      math::RigidTransform<double>::Identity();

  math::RigidTransform<double> X_WB_final(
      math::RotationMatrix<double>::MakeYRotation(M_PI / 16) *
          math::RotationMatrix<double>::MakeZRotation(M_PI / 8),
      Vector3<double>(10., 10., 10.) /* p */);

  FloatingBodyStateVector<double> q_v_initial;
  q_v_initial.template head<kNumFloatingBodyPositions>() =
      ToVector(X_WB_inital);
  q_v_initial.template tail<kNumFloatingBodyVelocities>() =
      FloatingBodySpatialVelocityVector<double>::Zero();

  FloatingBodyStateVector<double> q_v_final;
  q_v_initial.template head<kNumFloatingBodyPositions>() = ToVector(X_WB_final);
  q_v_initial.template tail<kNumFloatingBodyVelocities>() =
      FloatingBodySpatialVelocityVector<double>::Zero();

  systems::BasicVector<double> initial_state(q_v_initial);
  systems::BasicVector<double> final_state(q_v_final);
  const int num_states = initial_state.get_value().rows();

  const int kNumSegments = 2;
  const int kUpdateInterval = std::chrono::duration<double>(1s).count();

  auto multiple_shooting =
      std::make_unique<ParallelGenericDirectTranscription<T>>(
          construct_simulator_fn, set_state_fn, set_input_fn, get_state_fn,
          num_states, kSixSpatialDofs, kNumSegments, kUpdateInterval);

  auto& prog = multiple_shooting->prog();

  const solvers::VectorXDecisionVariable& u = multiple_shooting->input();
  multiple_shooting->AddRunningCost(u * u.transpose());

  // Set fixed end points.
  prog.AddLinearConstraint(multiple_shooting->initial_state() ==
                           initial_state.get_value());
  prog.AddLinearConstraint(multiple_shooting->final_state() ==
                           final_state.get_value());

  {
    std::vector<double> breaks(num_segments + 1);
    std::vector<MatrixX<double>> x_samples(num_segments + 1);
    std::vector<MatrixX<double>> u_samples(num_segments + 1);
    for (int i = 0; i <= num_segments; i++) {
      const double progress =
          (static_cast<double>(i) / static_cast<double>(num_segments));
      breaks[i] = expected_voyage_duration * progress;

      x_samples[i] = initial_state.get_value() * (1.0 - progress) +
                     final_state.get_value() * progress;

      Vector<T, kNumInputs> u_sample = Vector<T, kNumInputs>::Zero();

      u_samples[i] = u_sample;
    }

    // Create an initial guess for the state trajectory.
    multiple_shooting->SetInitialTrajectory(
        trajectories::PiecewisePolynomial<double>::ZeroOrderHold(),
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
  const auto state = multiple_shooting->ReconstructStateTrajectory(result);
  const std::vector<double>& breaks = state.get_segment_times();

  // Replay the final trajectory in the logs.
  if (result.is_success()) {
    log()->critical("SUCCESS!");

    // Replay the full roll-out.
    auto simulator = construct_simulator_fn();
    auto& context = simulator->get_mutable_context();
    context.SetTime(breaks[0]);
    simulator->Initialize();
    const double dt = (inputs.end_time() - inputs.start_time()) /
                      (static_cast<double>(num_segments) * 100.);
    for (double t = inputs.start_time(); t <= inputs.end_time(); t += dt) {
      set_input_fn(&context, inputs.value(t));
      simulator->AdvanceTo(t);
    }
  } else {
    log()->critical("FAILURE!");

    // Replay each segment.
    for (size_t i = 0; i < breaks.size() - 1; ++i) {
      const double start_time = breaks[i];
      const double end_time = breaks[i + 1];
      auto simulator = construct_simulator_fn();
      auto& context = simulator->get_mutable_context();

      context.SetTime(start_time);

      set_state_fn(&context, state.value(start_time));
      set_input_fn(&context, inputs.value(start_time));

      simulator->Initialize();

      const double dt = (end_time - start_time) / 100.;
      for (double t = start_time; t <= end_time; t += dt) {
        try {
          simulator->AdvanceTo(t);
        } catch (...) {
          break;
        }
      }
    }
  }
}

}  // namespace
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake