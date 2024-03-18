#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace controllers {

/**
 * Performs motion control for a single operational point.
 * The formulation is according to the paper "A unified approach for
 * motion control of robot manipulators: The operational
 * space formulation" (https://ieeexplore.ieee.org/document/1087068)
 *
 * @system
 * name: MotionTask
 * input_ports:
 * - estimated_state
 * - current_pose
 * - desired_pose
 * - desired_velocity
 * - desired_acceleration
 * - bias_spatial_acceleration
 * - jacobian
 * - projected_jacobian
 * - taskspace_inertia_matrix
 * output_ports:
 * - y0 (force)
 * @endsystem

 * Port `estimated_state` accepts system state;
 * Port `current_pose` accepts system current pose of the operational point;
 * Port `desired_pose` accepts system desired pose of the operational point;
 * Port `desired_velocity` accepts system desired velocity of the operational
 * point;
 * Port `desired_acceleration` accepts system desired acceleration of the
 * operational point;
 * Port `bias_spatial_acceleration` accepts system desired bias (non linear
 terms)
 * Port `jacobian` accepts system jacobian at the operational point;
 * Port `projected_jacobian` accepts system jacobian matrix from parent task;
 * Port `taskspace_inertia_matrix` accepts system taskspace inertia matrix to
 compute forces;
 * Port `force` emits generalized forces for control.
 *
 * @tparam_default_scalar
 * @ingroup control_systems
 */
template <typename T>
class MotionTask final : public drake::systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MotionTask)

  /**
   * Constructs the MotionTask system.
   *
   * @param plant Pointer to the multibody plant model. The life span of @p
   * plant must be longer than that of this instance.
   * @param kp Proportional position gain
   * @param kd Derivative position gain
   * @param num_operational_points Number of operational points to control
   * @param cap_max_velocity If true, caps the maximum velocity
   * @pre The plant must be finalized (i.e., plant.is_finalized() must return
   * `true`).
   */
  explicit MotionTask(const drake::multibody::MultibodyPlant<T>* plant,
                      const Eigen::Ref<const Eigen::VectorXd>& kp,
                      const Eigen::Ref<const Eigen::VectorXd>& kd,
                      const int& num_operational_points = 1,
                      const bool& cap_max_velocity = false);

  /**
   * Constructs the MotionTask system and takes the ownership of the
   * input `plant`.
   */
  explicit MotionTask(
      std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
      const Eigen::Ref<const Eigen::VectorXd>& kp,
      const Eigen::Ref<const Eigen::VectorXd>& kd,
      const int& num_operational_points = 1,
      const bool& cap_max_velocity = false);

  // Scalar-converting copy constructor.
  template <typename U>
  explicit MotionTask(const MotionTask<U>& other);

  ~MotionTask() override;

  /**
   * Returns the input port for the estimated state.
   */
  const drake::systems::InputPort<T>& get_input_port_estimated_state() const {
    return this->get_input_port(input_port_index_state_);
  }

  /**
   * Returns the input port for the current pose.
   */
  const drake::systems::InputPort<T>& get_input_port_current_poses() const {
    return this->get_input_port(input_port_index_current_pose_);
  }

  /**
   * Returns the input port for the desired state.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_poses() const {
    return this->get_input_port(input_port_index_desired_pose_);
  }

  /**
   * Returns the input port for the desired velocity.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_velocity() const {
    return this->get_input_port(input_port_index_desired_velocity_);
  }

  /**
   * Returns the input port for the desired acceleration.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_acceleration()
      const {
    return this->get_input_port(input_port_index_desired_acceleration_);
  }

  /**
   * Returns the input port for bias spatial acceleration.
   */
  const drake::systems::InputPort<T>& get_input_port_bias_spatial_acceleration()
      const {
    return this->get_input_port(input_port_bias_spatial_acceleration_);
  }

  /**
   * Returns the input port for the jacobian matrix of the operational point.
   */
  const drake::systems::InputPort<T>& get_input_port_jacobian() const {
    return this->get_input_port(input_port_index_jacobian_);
  }

  /**
   * Returns the input port for the parent jacobian matrix.
   */
  const drake::systems::InputPort<T>& get_input_port_projected_jacobian()
      const {
    return this->get_input_port(input_port_index_projected_jacobian_);
  }

  /**
   * Returns the input port for the taskspace inertia matrix to project the task
   * on.
   */
  const drake::systems::InputPort<T>& get_input_port_taskspace_inertia() const {
    return this->get_input_port(input_port_index_taskspace_inertia_);
  }

  /**
   * Returns the output port for the generalized forces that realize the desired
   * acceleration. The dimension of that force vector will be identical to the
   * dimensionality of the generalized velocities.
   */
  const drake::systems::OutputPort<T>& get_output_port_force() const {
    return this->get_output_port(output_port_index_force_);
  }

 private:
  // Other constructors delegate to this private constructor.
  MotionTask(std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
             const drake::multibody::MultibodyPlant<T>* plant,
             const Eigen::Ref<const Eigen::VectorXd>& kp,
             const Eigen::Ref<const Eigen::VectorXd>& kd,
             const int& num_operational_points, const bool& cap_max_velocity);

  template <typename>
  friend class MotionTask;

  // This is the calculator method for the output port.
  void CalcOutputForce(const drake::systems::Context<T>& context,
                       drake::systems::BasicVector<T>* force) const;

  // Method for updating multibody context.
  void SetMultibodyContext(const drake::systems::Context<T>&,
                           drake::systems::Context<T>*) const;

  const std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant_{};
  const drake::multibody::MultibodyPlant<T>* const plant_;

  int input_port_index_state_{0};
  int input_port_index_desired_pose_{0};
  int input_port_index_current_pose_{0};
  int input_port_index_desired_velocity_{0};
  int input_port_index_desired_acceleration_{0};
  int input_port_bias_spatial_acceleration_{0};
  int input_port_index_jacobian_{0};
  int input_port_index_projected_jacobian_{0};
  int input_port_index_taskspace_inertia_{0};
  int output_port_index_force_{0};

  const Eigen::VectorXd kp_, kd_;

  const int q_dim_{0};
  const int v_dim_{0};
  const int num_operational_points_{1};
  const bool cap_max_velocity_{false};

  drake::systems::CacheIndex plant_context_cache_index_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::MotionTask)
