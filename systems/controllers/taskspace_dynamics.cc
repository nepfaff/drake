#include "drake/systems/controllers/taskspace_dynamics.h"

#include <string>
#include <utility>
#include <vector>

#include "drake/systems/controllers/decomposition_inverse.h"
#include "drake/systems/controllers/scoped_names.h"

namespace drake {
namespace systems {
namespace controllers {

// number of terms to describe position & quaternion
constexpr int kCartesianPoseDimensions = 7;
// task dimensions for cartesian task
constexpr int kTaskDimensions = 6;
template <typename T>
TaskSpaceDynamics<T>::TaskSpaceDynamics(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_plant,
    const drake::multibody::MultibodyPlant<T>* plant,
    int num_operational_points)
    : drake::systems::LeafSystem<T>(
          drake::systems::SystemTypeTag<TaskSpaceDynamics>{}),
      owned_plant_(std::move(owned_plant)),
      plant_(owned_plant_ ? owned_plant_.get() : plant),
      q_dim_(plant_->num_positions()),
      v_dim_(plant_->num_velocities()),
      num_operational_points_(num_operational_points) {
  // Check that only one of owned_plant and plant where set.
  DRAKE_DEMAND(owned_plant_ == nullptr || plant == nullptr);
  DRAKE_DEMAND(plant_ != nullptr);
  DRAKE_DEMAND(plant_->is_finalized());

  input_port_index_operational_points_ =
      this->DeclareAbstractInputPort("operational_points",
                                     drake::Value<std::vector<std::string>>{})
          .get_index();

  input_port_index_projected_nullspace_ =
      this->DeclareAbstractInputPort("projected_nullspace",
                                     drake::Value<Eigen::MatrixX<T>>{})
          .get_index();

  input_port_index_state_ =
      this->DeclareInputPort("estimated_state", drake::systems::kVectorValued,
                             q_dim_ + v_dim_)
          .get_index();

  auto plant_context = plant_->CreateDefaultContext();

  // Declare cache entry for the multibody plant context.
  plant_context_cache_index_ =
      this->DeclareCacheEntry(
              "plant_context_cache", *plant_context,
              &TaskSpaceDynamics<T>::SetMultibodyContext,
              {this->input_port_ticket(
                  get_input_port_estimated_state().get_index())})
          .cache_index();

  output_port_index_operational_point_pose_ =
      this->DeclareVectorOutputPort(
              "operational_point_pose",
              kCartesianPoseDimensions * num_operational_points_,
              &TaskSpaceDynamics<T>::CalcOutputOperationalPointPose,
              {this->input_port_ticket(
                   get_input_port_estimated_state().get_index()),
               this->input_port_ticket(
                   get_input_port_operational_points().get_index()),
               this->cache_entry_ticket(plant_context_cache_index_)})
          .get_index();

  const drake::systems::LeafOutputPort<T>& jacobian =
      this->DeclareAbstractOutputPort(
          "jacobian", &TaskSpaceDynamics<T>::CalcOutputJacobian,
          {this->input_port_ticket(
               get_input_port_estimated_state().get_index()),
           this->input_port_ticket(
               get_input_port_operational_points().get_index()),
           this->cache_entry_ticket(plant_context_cache_index_)});
  output_port_index_jacobian_ = jacobian.get_index();

  const drake::systems::LeafOutputPort<T>& bias_spatial_acceleration =
      this->DeclareVectorOutputPort(
          "bias_spatial_acceleration",
          kTaskDimensions * num_operational_points_,
          &TaskSpaceDynamics<T>::CalcBiasSpatialAcceleration,
          {this->input_port_ticket(
               get_input_port_estimated_state().get_index()),
           this->input_port_ticket(
               get_input_port_operational_points().get_index()),
           this->cache_entry_ticket(plant_context_cache_index_)});

  output_port_index_bias_spatial_acceleration_ =
      bias_spatial_acceleration.get_index();

  const drake::systems::LeafOutputPort<T>& bias_term =
      this->DeclareVectorOutputPort(
          "bias_term", v_dim_, &TaskSpaceDynamics<T>::CalcBias,
          {this->input_port_ticket(
               get_input_port_estimated_state().get_index()),
           this->cache_entry_ticket(plant_context_cache_index_)});
  output_port_index_bias_ = bias_term.get_index();

  const drake::systems::LeafOutputPort<T>& projected_jacobian =
      this->DeclareAbstractOutputPort(
          "projected_jacobian",
          &TaskSpaceDynamics<T>::CalcOutputProjectedJacobian,
          {this->input_port_ticket(
               get_input_port_projected_nullspace().get_index()),
           jacobian.cache_entry().ticket()});
  output_port_index_projected_jacobian_ = projected_jacobian.get_index();

  const drake::systems::LeafOutputPort<T>& taskspace_inertia =
      this->DeclareAbstractOutputPort(
          "taskspace_inertia",
          &TaskSpaceDynamics<T>::CalcOutputTaskspaceInertia,
          {projected_jacobian.cache_entry().ticket(),
           this->cache_entry_ticket(plant_context_cache_index_)});
  output_port_index_taskspace_inertia_ = taskspace_inertia.get_index();

  const drake::systems::LeafOutputPort<T>& projected_jacobian_inverse =
      this->DeclareAbstractOutputPort(
          "projected_jacobian_inverse",
          &TaskSpaceDynamics<T>::CalcOutputProjectedJacobianInverse,
          {projected_jacobian.cache_entry().ticket(),
           taskspace_inertia.cache_entry().ticket()});
  output_port_index_projected_jacobian_inverse_ =
      projected_jacobian_inverse.get_index();

  output_port_index_projected_nullspace_ =
      this->DeclareAbstractOutputPort(
              "projected_nullspace",
              &TaskSpaceDynamics<T>::CalcOutputProjectedNullspace,
              {projected_jacobian.cache_entry().ticket(),
               projected_jacobian_inverse.cache_entry().ticket()})
          .get_index();
}

template <typename T>
TaskSpaceDynamics<T>::TaskSpaceDynamics(
    const drake::multibody::MultibodyPlant<T>* plant,
    int num_operational_points)
    : TaskSpaceDynamics(nullptr, plant, num_operational_points) {}

template <typename T>
TaskSpaceDynamics<T>::TaskSpaceDynamics(
    std::unique_ptr<drake::multibody::MultibodyPlant<T>> plant,
    int num_operational_points)
    : TaskSpaceDynamics(std::move(plant), nullptr, num_operational_points) {}

template <typename T>
template <typename U>
TaskSpaceDynamics<T>::TaskSpaceDynamics(const TaskSpaceDynamics<U>& other)
    : TaskSpaceDynamics(
          drake::systems::System<U>::template ToScalarType<T>(*other.plant_),
          other.num_operational_points_) {}

template <typename T>
TaskSpaceDynamics<T>::~TaskSpaceDynamics() = default;

template <typename T>
void TaskSpaceDynamics<T>::SetMultibodyContext(
    const drake::systems::Context<T>& context,
    drake::systems::Context<T>* plant_context) const {
  const Eigen::VectorX<T>& x = get_input_port_estimated_state().Eval(context);
  // Set the plant positions and velocities.
  plant_->SetPositionsAndVelocities(plant_context, x);
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputOperationalPointPose(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);
  Eigen::VectorX<T> operation_point_poses(kCartesianPoseDimensions *
                                          num_operational_points_);
  if (!get_input_port_operational_points().HasValue(context)) {
    operation_point_poses
        << drake::math::RigidTransform<T>::Identity().translation(),
        drake::math::RotationMatrix<T>::Identity().ToQuaternionAsVector4();
  } else {
    std::vector<std::string> operational_points =
        get_input_port_operational_points()
            .template Eval<std::vector<std::string>>(context);
    DRAKE_DEMAND(num_operational_points_ ==
                 static_cast<int>(operational_points.size()));
    for (int i = 0; i < num_operational_points_; i++) {
      const drake::multibody::Frame<T>& operational_point_frame =
          GetScopedFrameByName(*plant_, operational_points[i]);
      drake::math::RigidTransform<T> operational_pose =
          operational_point_frame.CalcPoseInWorld(plant_context);
      operation_point_poses.block(i * kCartesianPoseDimensions, 0,
                                  kCartesianPoseDimensions, 1)
          << operational_pose.translation(),
          operational_pose.rotation().ToQuaternionAsVector4();
    }
  }

  output->get_mutable_value() = operation_point_poses;
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputJacobian(
    const drake::systems::Context<T>& context,
    Eigen::MatrixX<T>* output) const {
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);

  if (get_input_port_operational_points().HasValue(context)) {
    std::vector<std::string> operational_points =
        get_input_port_operational_points()
            .template Eval<std::vector<std::string>>(context);
    Eigen::MatrixX<T> stacked_jacobian_matrix(
        operational_points.size() * kTaskDimensions, v_dim_);
    for (int i = 0; i < num_operational_points_; i++) {
      const drake::multibody::Frame<T>& operational_point_frame =
          GetScopedFrameByName(*plant_, operational_points[i]);

      // point on the body frame
      const drake::Vector3<T> point_on_frame = drake::Vector3<T>::Zero();
      Eigen::MatrixX<T> jacobian_position_matrix(point_on_frame.size(), v_dim_);
      plant_->CalcJacobianTranslationalVelocity(
          plant_context, drake::multibody::JacobianWrtVariable::kV,
          operational_point_frame, point_on_frame, plant_->world_frame(),
          plant_->world_frame(), &jacobian_position_matrix);

      Eigen::MatrixX<T> jacobian_orientation_matrix(point_on_frame.size(),
                                                    v_dim_);
      plant_->CalcJacobianAngularVelocity(
          plant_context, drake::multibody::JacobianWrtVariable::kV,
          operational_point_frame, plant_->world_frame(), plant_->world_frame(),
          &jacobian_orientation_matrix);

      Eigen::MatrixX<T> jacobian_matrix(
          jacobian_position_matrix.rows() + jacobian_orientation_matrix.rows(),
          jacobian_position_matrix.cols());
      jacobian_matrix << jacobian_position_matrix, jacobian_orientation_matrix;
      stacked_jacobian_matrix.block(i * jacobian_matrix.rows(), 0,
                                    jacobian_matrix.rows(),
                                    jacobian_matrix.cols()) = jacobian_matrix;
    }
    *output = stacked_jacobian_matrix;
  } else {
    *output = Eigen::MatrixX<T>::Identity(v_dim_, v_dim_);
  }
}

template <typename T>
void TaskSpaceDynamics<T>::CalcBiasSpatialAcceleration(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);
  Eigen::VectorX<T> bias_acceleration(kTaskDimensions *
                                      num_operational_points_);

  if (get_input_port_operational_points().HasValue(context)) {
    std::vector<std::string> operational_points =
        get_input_port_operational_points()
            .template Eval<std::vector<std::string>>(context);

    for (int i = 0; i < num_operational_points_; i++) {
      const drake::multibody::Frame<T>& operational_point_frame =
          GetScopedFrameByName(*plant_, operational_points[i]);

      // point on the body frame
      const drake::Vector3<T> point_on_frame = drake::Vector3<T>::Zero();
      bias_acceleration.block(i * kTaskDimensions, 0, kTaskDimensions, 1)
          << plant_
                 ->CalcBiasSpatialAcceleration(
                     plant_context, drake::multibody::JacobianWrtVariable::kV,
                     operational_point_frame, point_on_frame,
                     plant_->world_frame(), plant_->world_frame())
                 .get_coeffs();
    }
  } else {
    bias_acceleration =
        Eigen::VectorX<T>::Zero(kTaskDimensions * num_operational_points_);
  }
  output->get_mutable_value() = bias_acceleration;
}

template <typename T>
void TaskSpaceDynamics<T>::CalcBias(
    const drake::systems::Context<T>& context,
    drake::systems::BasicVector<T>* output) const {
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);
  Eigen::VectorX<T> bias(v_dim_);
  plant_->CalcBiasTerm(plant_context, &bias);
  output->get_mutable_value() = bias;
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputProjectedJacobian(
    const drake::systems::Context<T>& context,
    Eigen::MatrixX<T>* output) const {
  const Eigen::MatrixX<T>& stacked_jacobian_matrix =
      this->get_output_port_jacobian().template Eval<Eigen::MatrixX<T>>(
          context);
  const Eigen::MatrixX<T>& nullspace_parent =
      get_input_port_projected_nullspace().template Eval<Eigen::MatrixX<T>>(
          context);
  *output = stacked_jacobian_matrix * nullspace_parent.transpose();
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputTaskspaceInertia(
    const drake::systems::Context<T>& context,
    Eigen::MatrixX<T>* output) const {
  const auto& projected_jacobian =
      this->get_output_port_projected_jacobian()
          .template Eval<Eigen::MatrixX<T>>(context);
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);

  Eigen::MatrixX<T> mass_matrix = Eigen::MatrixX<T>::Zero(v_dim_, v_dim_);
  plant_->CalcMassMatrixViaInverseDynamics(plant_context, &mass_matrix);

  const Eigen::MatrixX<T> taskspace_inertia_inverse =
      projected_jacobian * CholeskyDecompositionInverse(mass_matrix) *
      projected_jacobian.transpose();
  Eigen::MatrixX<T> taskspace_inertia(taskspace_inertia_inverse.rows(),
                                      taskspace_inertia_inverse.cols());
  taskspace_inertia =
      SvdDecompositionInverse(taskspace_inertia_inverse, svd_inverse_param_);
  *output = taskspace_inertia;
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputProjectedJacobianInverse(
    const drake::systems::Context<T>& context,
    Eigen::MatrixX<T>* output) const {
  const auto& plant_context =
      this->get_cache_entry(plant_context_cache_index_)
          .template Eval<drake::systems::Context<T>>(context);
  const auto& projected_jacobian_matrix =
      this->get_output_port_projected_jacobian()
          .template Eval<Eigen::MatrixX<T>>(context);
  const auto& taskspace_inertia =
      this->get_output_port_taskspace_inertia()
          .template Eval<Eigen::MatrixX<T>>(context);
  // dynamically consistent inverse
  Eigen::MatrixX<T> mass_matrix = Eigen::MatrixX<T>::Zero(v_dim_, v_dim_);
  plant_->CalcMassMatrixViaInverseDynamics(plant_context, &mass_matrix);

  *output = CholeskyDecompositionInverse(mass_matrix) *
            projected_jacobian_matrix.transpose() * taskspace_inertia;
}

template <typename T>
void TaskSpaceDynamics<T>::CalcOutputProjectedNullspace(
    const drake::systems::Context<T>& context,
    Eigen::MatrixX<T>* output) const {
  const auto& projected_jacobian =
      this->get_output_port_projected_jacobian()
          .template Eval<Eigen::MatrixX<T>>(context);

  const auto& projected_jacobian_inverse =
      this->get_output_port_projected_jacobian_inverse()
          .template Eval<Eigen::MatrixX<T>>(context);

  *output =
      Eigen::MatrixX<T>::Identity(q_dim_, q_dim_) -
      projected_jacobian.transpose() * projected_jacobian_inverse.transpose();
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::controllers::TaskSpaceDynamics)
