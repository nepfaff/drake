#include "drake/systems/controllers/joint_posture_controller.h"

#include <memory>
#include <utility>

#include "drake/systems/controllers/posture_task.h"
#include "drake/systems/controllers/taskspace_dynamics.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/shared_pointer_system.h"

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
void JointPostureController<T>::SetUp(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd) {
  DRAKE_DEMAND(multibody_plant_for_control_->is_finalized());

  drake::systems::DiagramBuilder<T> builder;
  PostureTask<T>* posture_task{};
  TaskSpaceDynamics<T>* taskspace_dynamics{};
  if (owned_plant) {
    builder.template AddSystem<drake::systems::SharedPointerSystem<T>>(
        std::move(owned_plant));
  }
  posture_task = builder.template AddSystem<PostureTask<T>>(
      multibody_plant_for_control_, kp, kd);
  taskspace_dynamics = builder.template AddSystem<TaskSpaceDynamics<T>>(
      multibody_plant_for_control_);
  const int num_positions = multibody_plant_for_control_->num_positions();
  const int num_velocities = multibody_plant_for_control_->num_velocities();
  const int num_actuators = multibody_plant_for_control_->num_actuators();
  const int dim = kp.size();
  DRAKE_DEMAND(num_positions == dim);
  DRAKE_DEMAND(num_positions == num_velocities);
  DRAKE_DEMAND(num_positions == num_actuators);

  // update taskspace dynamics of the primary task
  auto parent_nullspace =
      builder.template AddSystem<drake::systems::ConstantValueSource>(
          drake::Value<Eigen::MatrixX<T>>(
              Eigen::MatrixX<T>::Identity(dim, dim)));
  builder.Connect(parent_nullspace->get_output_port(),
                  taskspace_dynamics->get_input_port_projected_nullspace());

  input_port_index_estimated_state_ = builder.ExportInput(
      taskspace_dynamics->get_input_port_estimated_state(), "estimated_state");

  // connect taskspace dynamics to the primary task
  builder.ConnectInput(input_port_index_estimated_state_,
                       posture_task->get_input_port_estimated_state());

  input_port_index_desired_state_ = builder.ExportInput(
      posture_task->get_input_port_desired_state(), "desired_state");

  input_port_index_desired_acceleration_ =
      builder.ExportInput(posture_task->get_input_port_desired_acceleration(),
                          "desired_acceleration");

  builder.Connect(taskspace_dynamics->get_output_port_projected_jacobian(),
                  posture_task->get_input_port_projected_jacobian());

  builder.Connect(taskspace_dynamics->get_output_port_taskspace_inertia(),
                  posture_task->get_input_port_taskspace_inertia());

  // Exposes posture task's output force port.
  output_port_index_control_ =
      builder.ExportOutput(posture_task->get_output_port_force(), "force");

  builder.BuildInto(this);
}

template <typename T>
JointPostureController<T>::JointPostureController(
    const drake::multibody::MultibodyPlant<T>& plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd)
    : multibody_plant_for_control_(&plant) {
  SetUp(nullptr, kp, kd);
}

template <typename T>
JointPostureController<T>::JointPostureController(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd)
    : multibody_plant_for_control_(plant.get()) {
  SetUp(std::move(plant), kp, kd);
}

template <typename T>
JointPostureController<T>::~JointPostureController() = default;

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::JointPostureController)
