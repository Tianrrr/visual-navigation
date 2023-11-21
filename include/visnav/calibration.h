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

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

/// CalibCornerData stores the locations and ids for all detected calibration
/// grid corners of one image
struct CalibCornerData {
  /// Detected (or sometimes reprojected) 2d corner location for every detected
  /// april grid corner in the image
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      corners;
  /// Corner id of the detected corners. This has the same size as `corners` and
  /// the corner id is the index in AprilGrid::aprilgrid_corner_pos_3d
  std::vector<int> corner_ids;
};

/// CalibInitPoseData is used only for loading initialization for the
/// calibration from file.
struct CalibInitPoseData {
  /// transform camera-to-aprilgrid, i.e. pose of the camera
  Sophus::SE3d T_a_c;

  size_t num_inliers;

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      reprojected_corners;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <class Scalar, class CamT>
struct LoadCalibration {
  static constexpr int N = CamT::N;

  LoadCalibration() {}

  // transformations from cameras to body (IMU)
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> T_i_c;

  // Camera intrinsics
  std::vector<CamT, Eigen::aligned_allocator<CamT>> intrinsics;
};

struct Calibration {
  typedef std::shared_ptr<Calibration> Ptr;

  Calibration() {}

  // transformations from cameras to body (IMU)
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> T_i_c;

  // Camera intrinsics
  std::vector<std::shared_ptr<AbstractCamera<double>>> intrinsics;
};

struct cam_intrinsics {
  cam_intrinsics() {
    intr_map["fx"] = 0;
    intr_map["yx"] = 0;
    intr_map["cx"] = 0;
    intr_map["cy"] = 0;
    intr_map["xi"] = 0;
    intr_map["alpha"] = 0;
  }
  std::string camera_type;
  std::map<std::string, double> intr_map;
};

struct IMU_CAM {
  IMU_CAM(){};

  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> T_imu_cam;

  std::vector<std::shared_ptr<AbstractCamera<double>>> intrinsics;
  std::vector<cam_intrinsics> cam_imu_intr;

  // 反序列化后，将calib_accel_bais和calib_gyro_bias的size变为3，resize（）
  std::vector<double> calib_accel_bias;
  std::vector<double> calib_gyro_bias;

  std::vector<double> gyro_noise_std;
  std::vector<double> accel_noise_std;

  std::vector<double> gyro_bias_std;
  std::vector<double> accel_bias_std;

  std::vector<std::vector<int>> resolution;
};

}  // namespace visnav
