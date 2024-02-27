#pragma once

#include <memory>
#include <stdexcept>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace controllers {
/**
 * Estimates the joint friction based on a model
 *
 * Computes the feedforward force `τ_friction` that needs to be applied so that
 * the friction in the joints is overcome. That is, `τ_friction` is the result
 * of a system ID based on constant velocity tests.
 *
 * The friction compensator outputs a force based on a scaled tanh function
 * (scaled by columb friction and has a slope based on the viscous friction) .
 * There is a deadband at zero velocity that can be tuned based on the noise
 * in the zero velocity estimate on the robot.
 *
 * The estimator is of the form:
 *   τ_friction = fc * tanh(estimated_velocity/deadband_vel) + fv *
 * estimated_velocity
 *
 * τ_friction is multiplied with the sign of the commanded force to get the
 * estimate in the correct direction.
 *
 * In this implementation, the value of τ_friction is capped at a default value.
 *
 *
 * @system
 * name: JointFrictionCompensation
 * input_ports:
 * - estimated_state
 * - input_torque
 * output_ports:
 * - friction_force
 * @endsystem
 *
 * Port `estimated_state` accepts system estimated state;
 * Port `input_torque` accepts system torque;
 * Port `friction_force` emits generalized forces that
 * compensate friction.
 *
 * @tparam_default_scalar
 * @ingroup control_systems
 */
template <typename T>
class JointFrictionCompensation final : public drake::systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JointFrictionCompensation)

  /**
   * Constructs the JointFrictionCompensation system.
   *
   * @param fc Columb friction
   * @param fv Viscous friction
   * @param v0 Velocity around 0
   */
  explicit JointFrictionCompensation(
      const Eigen::Ref<const Eigen::VectorXd>& fc,
      const Eigen::Ref<const Eigen::VectorXd>& fv, const double& v0);

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit JointFrictionCompensation(const JointFrictionCompensation<U>& other);

  ~JointFrictionCompensation() override;

  /**
   * Returns the input port for the estimated state.
   */
  const drake::systems::InputPort<T>& get_input_port_estimated_state() const {
    return this->get_input_port(input_port_index_state_);
  }

  /**
   * Returns the input port for the torque.
   */
  const drake::systems::InputPort<T>& get_input_port_torque() const {
    return this->get_input_port(input_port_index_torque_);
  }

  /**
   * Returns the output port for the generalized forces that compensate
   * for friction.
   */
  const drake::systems::OutputPort<T>& get_output_port_force() const {
    return this->get_output_port(output_port_index_force_);
  }

 private:
  template <typename>
  friend class JointFrictionCompensation;

  // coulomb and viscous friction
  const Eigen::VectorXd fc_, fv_;

  // velocity around 0
  const double v0_;

  // This is the calculator method for the output port.
  void CalcOutputForce(const drake::systems::Context<T>& context,
                       drake::systems::BasicVector<T>* force) const;

  int input_port_index_state_{0};
  int input_port_index_torque_{0};
  int output_port_index_force_{0};

  const int v_dim_{0};
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake

namespace drake {
namespace systems {
namespace scalar_conversion {
template <>
struct Traits<drake::systems::controllers::JointFrictionCompensation>
    : public systems::scalar_conversion::NonSymbolicTraits {};
}  // namespace scalar_conversion
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::JointFrictionCompensation)
