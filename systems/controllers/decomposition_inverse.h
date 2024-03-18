#pragma once

#include <Eigen/Dense>

#include "drake/common/name_value.h"

namespace drake {
namespace systems {
namespace controllers {

// Mass matrix inverse is computed using Cholesky decomposition as mass matrix
// is always positive semidifinite. This approach is faster than regular inverse
// computation.
// https://stackoverflow.com/questions/38640563/eigen-efficient-inverse-of-symmetric-positive-definite-matrix
template <typename T>
Eigen::MatrixX<T> CholeskyDecompositionInverse(
    const Eigen::MatrixX<T>& matrix) {
  return matrix.llt().solve(
      Eigen::MatrixX<T>::Identity(matrix.rows(), matrix.cols()));
}

/// Parameters for SvdDecompositionInverse.
struct SvdDecompositionInverseParam {
  double svd_min{0.001};
  double svd_range{0.00005};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(svd_min));
    a->Visit(DRAKE_NVP(svd_range));
  }
};

/// Uses SVD to perform matrix inversion, with regularization on singular values
/// approaching 0 to first have C⁰ blending and then be saturated to zero to
/// avoid large inverse singular values. This is adapted from the symmetric
/// regularization approach as mentioned in
/// https://journals.sagepub.com/doi/10.1177/0278364917698748
/// Note that this does not use the beta scaling term, but instead does linear
/// interpolation on singular values within the prescribed range.
template <typename T>
Eigen::MatrixX<T> SvdDecompositionInverse(
    const Eigen::MatrixX<T>& matrix,
    const SvdDecompositionInverseParam& param) {
  Eigen::MatrixX<T> invSigma =
      Eigen::MatrixX<T>::Zero(matrix.cols(), matrix.rows());

  Eigen::JacobiSVD<Eigen::MatrixX<T>> svdHolder;
  svdHolder.compute(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

  const Eigen::VectorX<T>& singular_values = svdHolder.singularValues();
  for (int i = 0; i < singular_values.size(); i++) {
    if (singular_values(i) > param.svd_min + param.svd_range) {
      invSigma(i, i) = 1.0 / singular_values(i);
    } else {
      if (singular_values(i) < param.svd_min) {
        invSigma(i, i) = 0;
      } else {
        auto ratio = (singular_values(i) - param.svd_min) / param.svd_range;
        invSigma(i, i) = ratio / singular_values(i);
      }
    }
  }
  return svdHolder.matrixV() * invSigma * svdHolder.matrixU().transpose();
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake