#pragma once

#include <memory>
#include <stdexcept>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace controllers {

/**
 * Solves the joint posture (tracking joint angles and velocities)
 * given the taskspace inertia matrix(Λ) and the jacobian.
 * It outputs the 'τ_des'. PostureTask can be used independently of
 * other tasks or as a lower priority task. The task dof is the same
 * as the active dof on the robot.
 *
 * The formulation is:
 *      F = Λ * a_desired
 *      τ_des = Jᵀₜ|꜀ F
 *
 * TODO (Aditya.Bhat) : As of now the plant isn't being used. But this
 * will change when we start using tactile feedback.
 *
 * @system
 * name: PostureTask
 * input_ports:
 * - estimated_state
 * - desired_state
 * - <span style="color:gray">desired_acceleration</span>
 * - projected_jacobian
 * - taskspace_inertia_matrix
 *
 * output_ports:
 * - force
 * @endsystem
 *
 * Port `estimated_state` accepts system estimated state;
 * Port `desired_state` accepts system desired state;
 * Port `desired_acceleration` accepts system desired acceleration;
 * Port `projected_jacobian` accepts system jacobian from parent task;
 * Port `taskspace_inertia_matrix` accepts system taskspace inertia matrix;
 * Port `force` emits generalized forces.
 *
 * @tparam_default_scalar
 * @ingroup control_systems
 */
template <typename T>
class PostureTask final : public drake::systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PostureTask)

  /**
   * Constructs the PostureTask system.
   *
   * @param plant Pointer to the multibody plant model. The life span of @p
   * plant must be longer than that of this instance.
   * @param kp Proportional gain
   * @param kd Derivative gain
   * @pre The plant must be finalized (i.e., plant.is_finalized() must return
   * `true`).
   */
  PostureTask(const drake::multibody::MultibodyPlant<T>* plant,
              const Eigen::Ref<const Eigen::VectorXd>& kp,
              const Eigen::Ref<const Eigen::VectorXd>& kd);

  /**
   * Constructs the PostureTask system and takes the ownership of the
   * input `plant`.
   */
  PostureTask(std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
              const Eigen::Ref<const Eigen::VectorXd>& kp,
              const Eigen::Ref<const Eigen::VectorXd>& kd);

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit PostureTask(const PostureTask<U>& other);

  ~PostureTask() override;

  /**
   * Returns the input port for the estimated state.
   */
  const drake::systems::InputPort<T>& get_input_port_estimated_state() const {
    return this->get_input_port(input_port_index_state_);
  }

  /**
   * Returns the input port for the desired state.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_state() const {
    return this->get_input_port(input_port_index_desired_state_);
  }

  /**
   * Returns the input port for the desired acceleration.
   */
  const drake::systems::InputPort<T>& get_input_port_desired_acceleration()
      const {
    return this->get_input_port(input_port_index_desired_acceleration_);
  }

  /**
   * Returns the input port for the projected jacobian matrix.
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
   * state within the constraints of the parent task. The dimension of that
   * force vector will be identical to the dimensionality of the number of
   * positions.
   */
  const drake::systems::OutputPort<T>& get_output_port_force() const {
    return this->get_output_port(output_port_index_force_);
  }

 private:
  // Other constructors delegate to this private constructor.
  PostureTask(std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
              const drake::multibody::MultibodyPlant<T>* plant,
              const Eigen::Ref<const Eigen::VectorXd>& kp,
              const Eigen::Ref<const Eigen::VectorXd>& kd);

  template <typename>
  friend class PostureTask;

  // This is the calculator method for the output port.
  void CalcOutputForce(const drake::systems::Context<T>& context,
                       drake::systems::BasicVector<T>* force) const;

  // Methods for updating cache entries.
  void SetMultibodyContext(const drake::systems::Context<T>&,
                           drake::systems::Context<T>*) const;

  const std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant_{};
  const drake::multibody::MultibodyPlant<T>* const plant_;

  int input_port_index_state_{0};
  int input_port_index_desired_state_{0};
  int input_port_index_desired_acceleration_{0};
  int input_port_index_projected_jacobian_{0};
  int input_port_index_taskspace_inertia_{0};
  int output_port_index_force_{0};

  const Eigen::VectorXd kp_, kd_;

  const int q_dim_{0};
  const int v_dim_{0};

  drake::systems::CacheIndex plant_context_cache_index_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::controllers::PostureTask)
