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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  // UNUSED(xi);
  T norm_xi = xi.norm();
  if (norm_xi < 1e-10) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    Eigen::Matrix<T, 3, 3> xi_hat;
    xi_hat << T(0), -xi(2, 0), xi(1, 0), xi(2, 0), T(0), -xi(0, 0), -xi(1, 0),
        xi(0, 0), T(0);

    Eigen::Matrix<T, 3, 3> Rot =
        Eigen::Matrix<T, 3, 3>::Identity() + sin(norm_xi) / norm_xi * xi_hat +
        (T(1) - cos(norm_xi)) / (norm_xi * norm_xi) * xi_hat * xi_hat;

    return Rot;
  }
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  // UNUSED(mat);
  T norm_xi = acos((mat.trace() - T(1)) / T(2));
  if (norm_xi < 1e-10) {
    Eigen::Matrix<T, 3, 1> xi;
    xi << T(0), T(0), T(0);
    return xi;
  } else {
    Eigen::Matrix<T, 3, 1> temp;
    temp << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);
    Eigen::Matrix<T, 3, 1> xi = norm_xi / (T(2) * sin(norm_xi)) * temp;
    return xi;
  }
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  // UNUSED(xi);
  Eigen::Matrix<T, 4, 4> RT = Eigen::Matrix<T, 4, 4>::Identity();
  Eigen::Matrix<T, 3, 1> w, v;
  v = xi.block(0, 0, 3, 1);
  w << xi(3, 0), xi(4, 0), xi(5, 0);
  T norm_w = w.norm();

  if (norm_w < 1e-10) {
    Eigen::Matrix<T, 3, 3> Rot = Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T, 3, 3> J = Eigen::Matrix<T, 3, 3>::Identity();
    RT.block(0, 0, 3, 3) = Rot;
    RT.block(0, 3, 3, 1) = J * v;
  } else {
    Eigen::Matrix<T, 3, 3> w_hat;
    w_hat << 0, -xi(5, 0), xi(4, 0), xi(5, 0), 0, -xi(3, 0), -xi(4, 0),
        xi(3, 0), 0;

    Eigen::Matrix<T, 3, 3> Rot =
        Eigen::Matrix<T, 3, 3>::Identity() + sin(norm_w) / norm_w * w_hat +
        (T(1) - cos(norm_w)) / (norm_w * norm_w) * w_hat * w_hat;
    Eigen::Matrix<T, 3, 3> J =
        Eigen::Matrix<T, 3, 3>::Identity() +
        (T(1) - cos(norm_w)) / (norm_w * norm_w) * w_hat +
        (norm_w - sin(norm_w)) / (norm_w * norm_w * norm_w) * w_hat * w_hat;
    RT.block(0, 0, 3, 3) = Rot;
    RT.block(0, 3, 3, 1) = J * v;
  }

  return RT;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  // UNUSED(mat);
  Eigen::Matrix<T, 6, 1> twist;

  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);
  T norm_w = acos((R.trace() - T(1)) / T(2));
  if (norm_w < 1e-10) {
    twist << t(0, 0), t(1, 0), t(2, 0), 0, 0, 0;
    return twist;
  } else {
    Eigen::Matrix<T, 3, 1> temp;
    temp << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    Eigen::Matrix<T, 3, 1> w = norm_w / (T(2) * sin(norm_w)) * temp;
    Eigen::Matrix<T, 3, 3> w_hat;
    w_hat << 0, -w(2, 0), w(1, 0), w(2, 0), 0, -w(0, 0), -w(1, 0), w(0, 0), 0;
    Eigen::Matrix<T, 3, 3> J_inv =
        Eigen::Matrix<T, 3, 3>::Identity() - T(1) / T(2) * w_hat +
        (T(1) / (norm_w * norm_w) -
         (T(1) + cos(norm_w)) / (2 * norm_w * sin(norm_w))) *
            w_hat * w_hat;
    twist.block(3, 0, 3, 1) = w;
    twist.block(0, 0, 3, 1) = J_inv * t;

    return twist;
  }
}

}  // namespace visnav
