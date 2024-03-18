#include "drake/systems/controllers/scoped_names.h"

#include "drake/multibody/parsing/scoped_names.h"
#include "drake/multibody/tree/scoped_name.h"

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
const drake::multibody::Frame<T>& GetScopedFrameByName(
    const drake::multibody::MultibodyPlant<T>& plant,
    const std::string& full_name) {
  if (full_name == "world") return plant.world_frame();
  auto scoped_name = drake::multibody::ScopedName::Parse(full_name);
  if (!scoped_name.get_namespace().empty()) {
    auto instance = plant.GetModelInstanceByName(scoped_name.get_namespace());
    return plant.GetFrameByName(scoped_name.get_element(), instance);
  } else {
    return plant.GetFrameByName(scoped_name.get_element());
  }
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&GetScopedFrameByName<T>))

}  // namespace controllers
}  // namespace systems
}  // namespace drake
