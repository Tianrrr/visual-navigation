/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

#include <basalt/imu/imu_types.h>
#include <basalt/imu/preintegration.h>

namespace visnav {

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 2: implement the rest of the functor
    Eigen::Matrix<T, 3, 1> p_cam = T_i_c.inverse() * T_w_i.inverse() * p_3d;
    Eigen::Matrix<T, 2, 1> p = cam->project(p_cam);
    residuals = p - p_2d;

    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model,
                                          const Sophus::SE3d& T_i_c,
                                          const double scale)
      : p_2d(p_2d), cam_model(cam_model), T_i_c(T_i_c), scale(scale) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 4: Compute reprojection error
    Eigen::Matrix<T, 3, 1> p_cam = T_i_c.inverse() * T_w_i.inverse() * p_3d_w;
    Eigen::Matrix<T, 2, 1> p = cam->project(p_cam);
    residuals = T(scale) * (p - p_2d.cast<T>());

    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
  Sophus::SE3d T_i_c;
  double scale;
};

struct PreImuCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PreImuCostFunctor(const basalt::IntegratedImuMeasurement<double>::Ptr& iim,
                    const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                    const Eigen::Vector3d& gw)
      : iim(iim), bg(bg), ba(ba), gw(gw) {}

  template <class T>
  bool operator()(const T* const sv0, const T* const sT0, const T* const sv1,
                  const T* const sT1, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T0(sT0);
    Eigen::Map<Sophus::SE3<T> const> const T1(sT1);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const v0(sv0);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const v1(sv1);

    Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals(sResiduals);
    residuals = iim->get_sqrt_cov_inv() *
                iim->simple_residual(v0, T0, v1, T1, gw, bg, ba);

    return true;
  }

  basalt::IntegratedImuMeasurement<double>::Ptr iim;
  const Eigen::Vector3d bg;
  const Eigen::Vector3d ba;
  const Eigen::Vector3d gw;
};

}  // namespace visnav
