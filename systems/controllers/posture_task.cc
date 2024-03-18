#include "drake/systems/controllers/posture_task.h"

#include <utility>
#include <vector>

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
PostureTask<T>::PostureTask(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
    const drake::multibody::MultibodyPlant<T>* plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd)
    : drake::systems::LeafSystem<T>(
          drake::systems::SystemTypeTag<PostureTask>{}),
      owned_plant_(std::move(owned_plant)),
      plant_(owned_plant_ ? owned_plant_.get() : plant),
      kp_(kp),
      kd_(kd),
      q_dim_(plant_->num_positions()),
      v_dim_(plant_->num_velocities()) {
  // Check that only one of owned_plant and plant where set.
  DRAKE_DEMAND(owned_plant_ == nullptr || plant == nullptr);
  DRAKE_DEMAND(plant_ != nullptr);
  DRAKE_DEMAND(plant_->is_finalized());
  DRAKE_DEMAND(kp_.size() == kd_.size());
  DRAKE_DEMAND(kp_.size() == v_dim_);
  DRAKE_DEMAND(q_dim_ == v_dim_);

  input_port_index_state_ =
      this->DeclareInputPort("estimated_state", drake::systems::kVectorValued,
                             q_dim_ + v_dim_)
          .get_index();

  input_port_index_desired_state_ =
      this->DeclareInputPort("desired_state", drake::systems::kVectorValued,
                             q_dim_ + v_dim_)
          .get_index();

  input_port_index_desired_acceleration_ =
      this->DeclareInputPort("desired_acceleration",
                             drake::systems::kVectorValued, v_dim_)
          .get_index();

  input_port_index_projected_jacobian_ =
      this->DeclareAbstractInputPort("projected_jacobian",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  input_port_index_taskspace_inertia_ =
      this->DeclareAbstractInputPort("taskspace_inertia",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  output_port_index_force_ =
      this->DeclareVectorOutputPort("force", v_dim_,
                                    &PostureTask<T>::CalcOutputForce,
                                    {this->all_input_ports_ticket()})
          .get_index();

  auto plant_context = plant_->CreateDefaultContext();

  // Declare cache entry for the multibody plant context.
  plant_context_cache_index_ =
      this->DeclareCacheEntry(
              "plant_context_cache", *plant_context,
              &PostureTask<T>::SetMultibodyContext,
              {this->input_port_ticket(
                  get_input_port_estimated_state().get_index())})
          .cache_index();
}

template <typename T>
PostureTask<T>::PostureTask(const drake::multibody::MultibodyPlant<T>* plant,
                            const Eigen::Ref<const Eigen::VectorXd>& kp,
                            const Eigen::Ref<const Eigen::VectorXd>& kd)
    : PostureTask(nullptr, plant, kp, kd) {}

template <typename T>
PostureTask<T>::PostureTask(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
    const Eigen::Ref<const Eigen::VectorXd>& kp,
    const Eigen::Ref<const Eigen::VectorXd>& kd)
    : PostureTask(std::move(plant), nullptr, kp, kd) {}

template <typename T>
template <typename U>
PostureTask<T>::PostureTask(const PostureTask<U>& other)
    : PostureTask(
          drake::systems::System<U>::template ToScalarType<T>(*other.plant_),
          other.kp_, other.kd_) {}

template <typename T>
PostureTask<T>::~PostureTask() = default;

template <typename T>
void PostureTask<T>::SetMultibodyContext(
    const drake::systems::Context<T>& context,
    drake::systems::Context<T>* plant_context) const {
  const drake::VectorX<T>& x = get_input_port_estimated_state().Eval(context);
  // Set the plant positions and velocities.
  plant_->SetPositionsAndVelocities(plant_context, x);
}

template <typename T>
void PostureTask<T>::CalcOutputForce(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  // PD control action
  const drake::VectorX<T>& estimated_state =
      get_input_port_estimated_state().Eval(context);
  const drake::VectorX<T>& desired_state =
      get_input_port_desired_state().Eval(context);

  const drake::VectorX<T>& pd_control =
      (kp_.array() *
       (desired_state.head(q_dim_) - estimated_state.head(q_dim_)).array()) +
      (kd_.array() *
       (desired_state.tail(q_dim_) - estimated_state.tail(q_dim_)).array());

  const auto& projected_jacobian_matrix =
      get_input_port_projected_jacobian().template Eval<Eigen::MatrixX<T>>(
          context);

  const auto& taskspace_inertia_matrix =
      get_input_port_taskspace_inertia().template Eval<Eigen::MatrixX<T>>(
          context);

  const drake::VectorX<T>& task_acceleration =
      get_input_port_desired_acceleration().HasValue(context)
          ? pd_control + get_input_port_desired_acceleration().Eval(context)
          : pd_control;

  output->get_mutable_value() = projected_jacobian_matrix.transpose() *
                                taskspace_inertia_matrix * task_acceleration;
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::controllers::PostureTask)
