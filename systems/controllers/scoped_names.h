#pragma once

#include <string>

#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace systems {
namespace controllers {

template <typename T>
const drake::multibody::Frame<T>& GetScopedFrameByName(
    const drake::multibody::MultibodyPlant<T>& plant,
    const std::string& full_name);

}  // namespace controllers
}  // namespace systems
}  // namespace drake