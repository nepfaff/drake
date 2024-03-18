#include "drake/systems/controllers/motion_task.h"

#include <limits>
#include <utility>
#include <vector>

namespace drake {
namespace systems {
namespace controllers {

// number of terms to describe position
constexpr int kPositionDimensions = 3;
// number of terms to describe quaternion
constexpr int kOrientationDimensions = 4;
// number of terms to describe position (3 dof) & quaternion (4 dof)
constexpr int kCartesianPoseDimensions =
    kPositionDimensions + kOrientationDimensions;
// task dimensions for cartesian task (mechanical degrees of freedom)
constexpr int kTaskDimensions = 6;
// maximum velocity in cartesian space (m/s)
constexpr double kMaxCartesianVelocity = 0.5;

namespace {
template <typename T>
const Eigen::MatrixX<T> ComputeEMatrix(const Eigen::Vector4<T>& orientation) {
  Eigen::MatrixX<T> error_matrix(4, 3);
  error_matrix << -orientation(1), -orientation(2), -orientation(3),
      orientation(0), orientation(3), -orientation(2), -orientation(3),
      orientation(0), orientation(1), orientation(2), -orientation(1),
      orientation(0);
  error_matrix *= 0.5;
  // get the inverse of e matrix
  return 4 * error_matrix.transpose();
}

}  // namespace

template <typename T>
MotionTask<T>::MotionTask(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
    const drake::multibody::MultibodyPlant<T>* plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd,
    const int& num_operational_points, const bool& cap_max_velocity)
    : drake::systems::LeafSystem<T>(
          drake::systems::SystemTypeTag<MotionTask>{}),
      owned_plant_(std::move(owned_plant)),
      plant_(owned_plant_ ? owned_plant_.get() : plant),
      kp_(kp),
      kd_(kd),
      q_dim_(plant_->num_positions()),
      v_dim_(plant_->num_velocities()),
      num_operational_points_(num_operational_points),
      cap_max_velocity_(cap_max_velocity) {
  // Check that only one of owned_plant and plant was set.
  DRAKE_DEMAND(owned_plant_ == nullptr || plant == nullptr);
  DRAKE_DEMAND(plant_ != nullptr);
  DRAKE_DEMAND(plant_->is_finalized());
  DRAKE_DEMAND(kp_.size() == kd_.size());
  DRAKE_DEMAND(kp_.size() == kTaskDimensions);

  input_port_index_state_ =
      this->DeclareInputPort("estimated_state", drake::systems::kVectorValued,
                             q_dim_ + v_dim_)
          .get_index();

  input_port_index_current_pose_ =
      this->DeclareInputPort("operational_point_pose",
                             drake::systems::kVectorValued,
                             num_operational_points_ * kCartesianPoseDimensions)
          .get_index();
  input_port_index_desired_pose_ =
      this->DeclareInputPort("desired_pose", drake::systems::kVectorValued,
                             num_operational_points_ * kCartesianPoseDimensions)
          .get_index();

  input_port_index_desired_velocity_ =
      this->DeclareInputPort("desired_velocity", drake::systems::kVectorValued,
                             num_operational_points_ * kTaskDimensions)
          .get_index();

  input_port_index_desired_acceleration_ =
      this->DeclareInputPort("desired_acceleration",
                             drake::systems::kVectorValued,
                             num_operational_points_ * kTaskDimensions)
          .get_index();

  input_port_bias_spatial_acceleration_ =
      this->DeclareInputPort("bias_spatial_acceleration",
                             drake::systems::kVectorValued,
                             num_operational_points_ * kTaskDimensions)
          .get_index();

  input_port_index_jacobian_ =
      this->DeclareAbstractInputPort("jacobian",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  input_port_index_projected_jacobian_ =
      this->DeclareAbstractInputPort("projected_jacobian",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  input_port_index_taskspace_inertia_ =
      this->DeclareAbstractInputPort("taskspace_inertia",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  auto plant_context = plant_->CreateDefaultContext();

  // Declare cache entry for the multibody plant context.
  plant_context_cache_index_ =
      this->DeclareCacheEntry(
              "plant_context_cache", *plant_context,
              &MotionTask<T>::SetMultibodyContext,
              {this->input_port_ticket(
                  get_input_port_estimated_state().get_index())})
          .cache_index();

  output_port_index_force_ =
      this->DeclareVectorOutputPort("force", v_dim_,
                                    &MotionTask<T>::CalcOutputForce,
                                    {this->all_input_ports_ticket()})
          .get_index();
}

template <typename T>
MotionTask<T>::MotionTask(const drake::multibody::MultibodyPlant<T>* plant,
                          const Eigen::Ref<const Eigen::VectorXd>& kp,
                          const Eigen::Ref<const Eigen::VectorXd>& kd,
                          const int& num_operational_points,
                          const bool& cap_max_velocity)
    : MotionTask(nullptr, plant, kp, kd, num_operational_points,
                 cap_max_velocity) {}

template <typename T>
MotionTask<T>::MotionTask(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd,
    const int& num_operational_points, const bool& cap_max_velocity)
    : MotionTask(std::move(plant), nullptr, kp, kd, num_operational_points,
                 cap_max_velocity) {}

template <typename T>
template <typename U>
MotionTask<T>::MotionTask(const MotionTask<U>& other)
    : MotionTask(
          drake::systems::System<U>::template ToScalarType<T>(*other.plant_),
          other.kp_, other.kd_, other.num_operational_points_,
          other.cap_max_velocity_) {}

template <typename T>
MotionTask<T>::~MotionTask() = default;

template <typename T>
void MotionTask<T>::SetMultibodyContext(
    const drake::systems::Context<T>& context,
    drake::systems::Context<T>* plant_context) const {
  const Eigen::VectorX<T>& x = get_input_port_estimated_state().Eval(context);
  // Set the plant positions and velocities.
  plant_->SetPositionsAndVelocities(plant_context, x);
}

template <typename T>
void MotionTask<T>::CalcOutputForce(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  // Get the desired task-space pose.
  const auto& desired_pose = get_input_port_desired_poses().Eval(context);

  // Get the current task-space pose.
  const auto& current_pose = get_input_port_current_poses().Eval(context);

  const Eigen::MatrixX<T>& jacobian_matrix =
      get_input_port_jacobian().template Eval<Eigen::MatrixX<T>>(context);

  const Eigen::MatrixX<T>& projected_jacobian =
      get_input_port_projected_jacobian().template Eval<Eigen::MatrixX<T>>(
          context);

  const Eigen::MatrixX<T>& taskspace_inertia =
      get_input_port_taskspace_inertia().template Eval<Eigen::MatrixX<T>>(
          context);

  Eigen::VectorX<T> pd_control(num_operational_points_ * kTaskDimensions);
  for (int i = 0; i < num_operational_points_; i++) {
    // get current velocity
    const auto& linear_velocity =
        jacobian_matrix.block((i * kTaskDimensions), 0, (kTaskDimensions / 2),
                              q_dim_) *
        get_input_port_estimated_state().Eval(context).tail(q_dim_);

    const auto& angular_velocity =
        jacobian_matrix.block((i * kTaskDimensions) + (kTaskDimensions / 2), 0,
                              kTaskDimensions / 2, q_dim_) *
        get_input_port_estimated_state().Eval(context).tail(q_dim_);

    Eigen::VectorX<T> current_velocity(kTaskDimensions);
    current_velocity << linear_velocity, angular_velocity;

    Eigen::Vector4<T> current_orientation =
        current_pose.block((i * kCartesianPoseDimensions) + kPositionDimensions,
                           0, kOrientationDimensions, 1);

    Eigen::Vector3<T> current_pos = current_pose.block(
        (i * kCartesianPoseDimensions), 0, kPositionDimensions, 1);

    Eigen::Vector4<T> goal_orientation =
        desired_pose.block((i * kCartesianPoseDimensions) + kPositionDimensions,
                           0, kOrientationDimensions, 1);

    Eigen::Vector3<T> desired_pos = desired_pose.block(
        (i * kCartesianPoseDimensions), 0, kPositionDimensions, 1);

    const auto& dist1 = (current_orientation - goal_orientation).norm();
    const auto& dist2 = (-current_orientation - goal_orientation).norm();
    if (dist1 > dist2) {
      current_orientation = -current_orientation;
    }

    // Compute the motion control terms.
    Eigen::VectorX<T> pos_error(kTaskDimensions);
    pos_error.head(3) = desired_pos - current_pos;
    pos_error.tail(3) = ComputeEMatrix(current_orientation) * goal_orientation;

    Eigen::VectorX<T> vel_error(kTaskDimensions);
    vel_error.head(3) =
        get_input_port_desired_velocity().Eval(context).block(
            (i * kTaskDimensions), 0, (kTaskDimensions / 2), 1) -
        linear_velocity;
    vel_error.tail(3) = get_input_port_desired_velocity().Eval(context).block(
                            (i * kTaskDimensions) + (kTaskDimensions / 2), 0,
                            (kTaskDimensions / 2), 1) -
                        angular_velocity;

    Eigen::VectorX<T> pd_control_values =
        kp_.array() * pos_error.array() + kd_.array() * vel_error.array();

    if (cap_max_velocity_) {
      Eigen::Vector3<T> tmp_velocity;
      tmp_velocity = kp_.block(0, 0, kTaskDimensions / 2, 1).array() /
                     kd_.block(0, 0, kTaskDimensions / 2, 1).array() *
                     pos_error.head(kTaskDimensions / 2).array();
      auto ratio_pos = kMaxCartesianVelocity / tmp_velocity.norm();
      if (tmp_velocity.norm() < std::numeric_limits<T>::epsilon()) {
        ratio_pos = 1.0;
      }
      if (ratio_pos > 1.0) {
        ratio_pos = 1.0;
      }

      pd_control_values.block(0, 0, kTaskDimensions / 2, 1) =
          -kd_.block(0, 0, kTaskDimensions / 2, 1).array() *
          (linear_velocity - ratio_pos * tmp_velocity).array();
    }

    pd_control.block((i * kTaskDimensions), 0, kTaskDimensions, 1) =
        get_input_port_desired_acceleration().HasValue(context)
            ? pd_control_values +
                  get_input_port_desired_acceleration().Eval(context).block(
                      (i * kTaskDimensions), 0, kTaskDimensions, 1)
            : pd_control_values;

    pd_control.block((i * kTaskDimensions), 0, kTaskDimensions, 1) -=
        get_input_port_bias_spatial_acceleration().Eval(context).block(
            (i * kTaskDimensions), 0, kTaskDimensions, 1);
  }
  auto task_force = taskspace_inertia * pd_control;
  output->get_mutable_value() = projected_jacobian.transpose() * task_force;
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::MotionTask)
