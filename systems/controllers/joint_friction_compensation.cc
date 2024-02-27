#include "drake/systems/controllers/joint_friction_compensation.h"

#include <utility>
#include <vector>

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
JointFrictionCompensation<T>::JointFrictionCompensation(
    const Eigen::Ref<const Eigen::VectorXd>& fc,
    const Eigen::Ref<const Eigen::VectorXd>& fv, const double& v0)
    : drake::systems::LeafSystem<T>(
          drake::systems::SystemTypeTag<JointFrictionCompensation>{}),
      fc_(fc),
      fv_(fv),
      v0_(v0),
      v_dim_(fc.size()) {
  input_port_index_state_ =
      this->DeclareInputPort("estimated_state", drake::systems::kVectorValued,
                             v_dim_ + v_dim_)
          .get_index();
  input_port_index_torque_ =
      this->DeclareInputPort("input_torque", drake::systems::kVectorValued,
                             v_dim_)
          .get_index();

  output_port_index_force_ =
      this->DeclareVectorOutputPort(
              "friction_force", v_dim_,
              &JointFrictionCompensation<T>::CalcOutputForce,
              {this->all_input_ports_ticket()})
          .get_index();
}

template <typename T>
template <typename U>
JointFrictionCompensation<T>::JointFrictionCompensation(
    const JointFrictionCompensation<U>& other)
    : JointFrictionCompensation(other.fc_, other.fv_, other.v0_) {}

template <typename T>
JointFrictionCompensation<T>::~JointFrictionCompensation() = default;

template <typename T>
void JointFrictionCompensation<T>::CalcOutputForce(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  const Eigen::VectorX<T>& input_velocity =
      get_input_port_estimated_state().Eval(context).tail(v_dim_);

  const Eigen::VectorX<T>& torque_values =
      get_input_port_torque().Eval(context);

  const Eigen::VectorX<T>& torque_sign = torque_values.array().sign();
  const Eigen::VectorX<T>& friction_torque =
      fc_ * (input_velocity.cwiseAbs().array() / v0_).array().tanh().matrix() +
      fv_ * input_velocity.cwiseAbs();

  // TODO(AdityaBhat): See if this number holds up
  constexpr double friction_cap = 1.5;
  const Eigen::VectorX<T>& capped_friction_torque =
      friction_torque.array()
          .min(friction_cap * Eigen::VectorX<T>::Ones(v_dim_).array())
          .max(-friction_cap * Eigen::VectorX<T>::Ones(v_dim_).array())
          .matrix();
  output->get_mutable_value() =
      (capped_friction_torque.array() * torque_sign.array()).matrix();
}
}  // namespace controllers
}  // namespace systems
}  // namespace anzu

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::JointFrictionCompensation)