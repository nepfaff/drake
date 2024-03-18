#include "drake/systems/controllers/spatial_motion_controller.h"

#include <memory>
#include <utility>

#include "drake/systems/controllers/motion_task.h"
#include "drake/systems/controllers/posture_task.h"
#include "drake/systems/controllers/taskspace_dynamics.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/shared_pointer_system.h"

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
void SpatialMotionController<T>::SetUp(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd,
    const Eigen::Ref<const Eigen::VectorXd>& kp_posture,
    const Eigen::Ref<const Eigen::VectorXd>& kd_posture,
    const int num_operational_points, const bool cap_max_velocity) {
  DRAKE_DEMAND(multibody_plant_for_control_->is_finalized());

  drake::systems::DiagramBuilder<T> builder;
  MotionTask<T>* motion_task{};
  PostureTask<T>* posture_task{};
  TaskSpaceDynamics<T>* taskspace_dynamics{};
  TaskSpaceDynamics<T>* taskspace_dynamics_secondary{};

  if (owned_plant) {
    builder.template AddSystem<drake::systems::SharedPointerSystem<T>>(
        std::move(owned_plant));
  }
  motion_task = builder.template AddSystem<MotionTask<T>>(
      multibody_plant_for_control_, kp, kd, num_operational_points,
      cap_max_velocity);
  posture_task = builder.template AddSystem<PostureTask<T>>(
      multibody_plant_for_control_, kp_posture, kd_posture);
  taskspace_dynamics = builder.template AddSystem<TaskSpaceDynamics<T>>(
      multibody_plant_for_control_, num_operational_points);
  taskspace_dynamics_secondary =
      builder.template AddSystem<TaskSpaceDynamics<T>>(
          multibody_plant_for_control_);

  const int num_positions = multibody_plant_for_control_->num_positions();
  const int num_velocities = multibody_plant_for_control_->num_velocities();
  const int num_actuators = multibody_plant_for_control_->num_actuators();

  DRAKE_DEMAND(num_positions == num_velocities);
  DRAKE_DEMAND(num_positions == num_actuators);

  // update taskspace dynamics of the primary task
  input_port_index_operational_points_ = builder.ExportInput(
      taskspace_dynamics->get_input_port_operational_points(),
      "operational_points");

  auto parent_nullspace =
      builder.template AddSystem<drake::systems::ConstantValueSource>(
          drake::Value<Eigen::MatrixX<T>>(
              Eigen::MatrixX<T>::Identity(num_positions, num_positions)));
  builder.Connect(parent_nullspace->get_output_port(),
                  taskspace_dynamics->get_input_port_projected_nullspace());

  input_port_index_estimated_state_ = builder.ExportInput(
      taskspace_dynamics->get_input_port_estimated_state(), "estimated_state");

  // connect taskspace dynamics to the primary task
  builder.Connect(taskspace_dynamics->get_output_port_operational_point_pose(),
                  motion_task->get_input_port_taskspace_inertia());

  builder.Connect(
      taskspace_dynamics->get_output_port_bias_spatial_acceleration(),
      motion_task->get_input_port_bias_spatial_acceleration());

  builder.Connect(taskspace_dynamics->get_output_port_jacobian(),
                  motion_task->get_input_port_jacobian());

  builder.Connect(taskspace_dynamics->get_output_port_taskspace_inertia(),
                  motion_task->get_input_port_taskspace_inertia());

  builder.Connect(taskspace_dynamics->get_output_port_projected_jacobian(),
                  motion_task->get_input_port_projected_jacobian());

  builder.ConnectInput(input_port_index_estimated_state_,
                       motion_task->get_input_port_estimated_state());

  input_port_index_desired_poses_ = builder.ExportInput(
      motion_task->get_input_port_desired_poses(), "desired_poses");

  input_port_index_desired_velocity_ = builder.ExportInput(
      motion_task->get_input_port_desired_velocity(), "estimated_velocity");

  input_port_index_desired_acceleration_ =
      builder.ExportInput(motion_task->get_input_port_desired_acceleration(),
                          "desired_acceleration");

  // update taskspace dynamics of the secondary task
  builder.Connect(
      taskspace_dynamics->get_output_port_projected_nullspace(),
      taskspace_dynamics_secondary->get_input_port_projected_nullspace());

  builder.ConnectInput(
      input_port_index_estimated_state_,
      taskspace_dynamics_secondary->get_input_port_estimated_state());

  // connect secondary taskspace dynamics to the secondary task
  builder.Connect(
      taskspace_dynamics_secondary->get_output_port_taskspace_inertia(),
      posture_task->get_input_port_taskspace_inertia());

  builder.Connect(
      taskspace_dynamics_secondary->get_output_port_projected_jacobian(),
      posture_task->get_input_port_projected_jacobian());

  builder.ConnectInput(input_port_index_estimated_state_,
                       posture_task->get_input_port_estimated_state());

  input_port_index_desired_state_ = builder.ExportInput(
      posture_task->get_input_port_desired_state(), "desired_state");

  // add primary and secondary taskspace outputs
  auto task_force_adder =
      builder.template AddSystem<drake::systems::Adder>(3, num_actuators);
  builder.Connect(motion_task->get_output_port_force(),
                  task_force_adder->get_input_port(0));
  builder.Connect(posture_task->get_output_port_force(),
                  task_force_adder->get_input_port(1));
  builder.Connect(taskspace_dynamics->get_output_port_bias(),
                  task_force_adder->get_input_port(2));

  // Exposes posture task's output force port.
  output_port_index_control_ =
      builder.ExportOutput(task_force_adder->get_output_port(), "force");

  // Exposes current operational point.
  output_port_index_operational_point_pose_ = builder.ExportOutput(
      taskspace_dynamics->get_output_port_operational_point_pose(),
      "operational_point_pose");

  builder.BuildInto(this);
}

template <typename T>
SpatialMotionController<T>::SpatialMotionController(
    const drake::multibody::MultibodyPlant<T>& plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd,
    const Eigen::Ref<const Eigen::VectorXd>& kp_posture,
    const Eigen::Ref<const Eigen::VectorXd>& kd_posture,
    const int num_operational_points, const bool cap_max_velocity)
    : multibody_plant_for_control_(&plant),
      num_operational_points_(num_operational_points) {
  SetUp(nullptr, kp, kd, kp_posture, kd_posture, num_operational_points,
        cap_max_velocity);
}

template <typename T>
SpatialMotionController<T>::SpatialMotionController(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd,
    const Eigen::Ref<const Eigen::VectorXd>& kp_posture,
    const Eigen::Ref<const Eigen::VectorXd>& kd_posture,
    const int num_operational_points, const bool cap_max_velocity)
    : multibody_plant_for_control_(plant.get()),
      num_operational_points_(num_operational_points) {
  SetUp(std::move(plant), kp, kd, kp_posture, kd_posture,
        num_operational_points, cap_max_velocity);
}

template <typename T>
SpatialMotionController<T>::~SpatialMotionController() = default;

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::SpatialMotionController)
