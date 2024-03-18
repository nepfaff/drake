#pragma once

#include <memory>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/controllers/decomposition_inverse.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace controllers {

/**
 * Computes taskspace dynamics. Given the estimated state, operational point &
 * the parent nullspace matrix, this computes the operational point's pose,
 * jacobian, projected jacobian (and inverse), cartesian space inertia matrix
 * and the projected nullspace.
 *
 * All multi task controllers use this as the input to the tasks so that
 * the correct constrained jacobian is obtained and the right nullspace gets
 * projected to lower priority tasks.
 *
 * @system
 * name: TaskSpaceDynamics
 * input_ports:
 * - operational_point
 * - parent_nullspace
 * - estimated_state
 * output_ports:
 * - operational_point_pose
 * - jacobian
 * - projected_jacobian
 * - taskspace_inertia
 * - projected_jacobian_inverse
 * - projected_nullspace
 * @endsystem
 *
 * Port `operational_point` accepts the name of the of the operational point
 frame
 * Port `parent_nullspace` accepts the nullspace matrix of the previous task
 * Port `estimated_state` accepts the estimated state of the system

 * Port `operational_point_pose` emits the RigidTransform of the frame wrt the
 world frame
 * Port `jacobian` emits the jacobian matrix for the frame wrt world frame
 * Port `projected_jacobian` emits the projected jacobian matrix
 * Port `taskspace_inertia` emits the taskspace inertia matrix (cartesian space
 inertia)
 * Port `projected_jacobian_inverse` emits the projected jacobian inverse matrix
 (Jbar)
 * Port `projected_nullspace` emits the projected nullspace matrix
 *
 * @tparam_default_scalar
 * @ingroup control_systems
 */
template <typename T>
class TaskSpaceDynamics final : public drake::systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TaskSpaceDynamics)

  /**
   * Constructs the TaskSpaceDynamics system.
   *
   * @param plant Pointer to the multibody plant model. The life span of @p
   * plant must be longer than that of this instance.
   * @pre The plant must be finalized (i.e., plant.is_finalized() must return
   * `true`).
   */
  explicit TaskSpaceDynamics(const drake::multibody::MultibodyPlant<T>* plant,
                             int num_operational_points = 1);

  /**
   * Constructs the TaskSpaceDynamics system and takes the ownership of the
   * input `plant`.
   */
  explicit TaskSpaceDynamics(
      std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
      int num_operational_points = 1);

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit TaskSpaceDynamics(const TaskSpaceDynamics<U>& other);

  ~TaskSpaceDynamics() override;

  /**
   * Returns the input port for the operational point
   */
  const drake::systems::InputPort<T>& get_input_port_operational_points()
      const {
    return this->get_input_port(input_port_index_operational_points_);
  }

  /**
   * Returns the input port for the projected nullspace
   */
  const drake::systems::InputPort<T>& get_input_port_projected_nullspace()
      const {
    return this->get_input_port(input_port_index_projected_nullspace_);
  }

  /**
   * Returns the input port for the estimated state.
   */
  const drake::systems::InputPort<T>& get_input_port_estimated_state() const {
    return this->get_input_port(input_port_index_state_);
  }

  /**
   * Returns the output current operational point pose
   */
  const drake::systems::OutputPort<T>& get_output_port_operational_point_pose()
      const {
    return this->get_output_port(output_port_index_operational_point_pose_);
  }

  /**
   * Returns the output jacobian matrix for the frame
   */
  const drake::systems::OutputPort<T>& get_output_port_jacobian() const {
    return this->get_output_port(output_port_index_jacobian_);
  }

  /**
   * Returns the output jacobian dot velocity for the frame
   */
  const drake::systems::OutputPort<T>&
  get_output_port_bias_spatial_acceleration() const {
    return this->get_output_port(output_port_index_bias_spatial_acceleration_);
  }

  /**
   * Returns the output bias
   */
  const drake::systems::OutputPort<T>& get_output_port_bias() const {
    return this->get_output_port(output_port_index_bias_);
  }

  /**
   * Returns the projected jacobian matrix
   */
  const drake::systems::OutputPort<T>& get_output_port_projected_jacobian()
      const {
    return this->get_output_port(output_port_index_projected_jacobian_);
  }

  /**
   * Returns the taskspace inertia matrix
   */
  const drake::systems::OutputPort<T>& get_output_port_taskspace_inertia()
      const {
    return this->get_output_port(output_port_index_taskspace_inertia_);
  }

  /**
   * Returns the projected jacobian inverse matrix
   */
  const drake::systems::OutputPort<T>&
  get_output_port_projected_jacobian_inverse() const {
    return this->get_output_port(output_port_index_projected_jacobian_inverse_);
  }

  /**
   *  Returns the projected nullspace matrix
   */
  const drake::systems::OutputPort<T>& get_output_port_projected_nullspace()
      const {
    return this->get_output_port(output_port_index_projected_nullspace_);
  }

 private:
  // Other constructors delegate to this private constructor.
  TaskSpaceDynamics(
      std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
      const drake::multibody::MultibodyPlant<T>* plant,
      int num_operational_points);

  template <typename>
  friend class TaskSpaceDynamics;

  // This is the calculator method for the output port.
  void CalcOutputOperationalPointPose(
      const drake::systems::Context<T>& context,
      drake::systems::BasicVector<T>* output) const;

  // This is the calculator method for the output port.
  void CalcOutputJacobian(const drake::systems::Context<T>& context,
                          Eigen::MatrixX<T>* output) const;

  // This is the calculator method for the output port.
  void CalcBiasSpatialAcceleration(
      const drake::systems::Context<T>& context,
      drake::systems::BasicVector<T>* output) const;

  // This is the calculator method for the output port.
  void CalcBias(const drake::systems::Context<T>& context,
                drake::systems::BasicVector<T>* output) const;

  // This is the calculator method for the output port.
  void CalcOutputProjectedJacobian(const drake::systems::Context<T>& context,
                                   Eigen::MatrixX<T>* output) const;

  // This is the calculator method for the output port.
  void CalcOutputTaskspaceInertia(const drake::systems::Context<T>& context,
                                  Eigen::MatrixX<T>* output) const;

  //  This is the calculator method for the output port.
  void CalcOutputProjectedJacobianInverse(
      const drake::systems::Context<T>& context,
      Eigen::MatrixX<T>* output) const;

  // This is the calculator method for the output port.
  void CalcOutputProjectedNullspace(const drake::systems::Context<T>& context,
                                    Eigen::MatrixX<T>* output) const;

  // Methods for updating cache entries.
  void SetMultibodyContext(const drake::systems::Context<T>&,
                           drake::systems::Context<T>*) const;

  const std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant_{};
  const drake::multibody::MultibodyPlant<T>* const plant_;

  int input_port_index_operational_points_{0};
  int input_port_index_projected_nullspace_{0};
  int input_port_index_state_{0};
  int output_port_index_operational_point_pose_{0};
  int output_port_index_jacobian_{0};
  int output_port_index_bias_spatial_acceleration_{0};
  int output_port_index_bias_{0};
  int output_port_index_projected_jacobian_{0};
  int output_port_index_taskspace_inertia_{0};
  int output_port_index_projected_jacobian_inverse_{0};
  int output_port_index_projected_nullspace_{0};

  const int q_dim_{0};
  const int v_dim_{0};
  const int num_operational_points_{0};
  const SvdDecompositionInverseParam svd_inverse_param_{};

  drake::systems::CacheIndex plant_context_cache_index_{0};
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::TaskSpaceDynamics)
