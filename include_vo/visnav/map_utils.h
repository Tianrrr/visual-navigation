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

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map
  // UNUSED(calib_cam);
  // UNUSED(feature_corners);
  // UNUSED(cameras);
  // UNUSED(landmarks);

  opengv::bearingVectors_t bearingVectors1;
  opengv::bearingVectors_t bearingVectors2;

  Sophus::SE3d T = cameras.at(fcid0).T_w_c.inverse() * cameras.at(fcid1).T_w_c;
  Eigen::Matrix<double, 3, 1> translation = T.matrix().block(0, 3, 3, 1);
  Eigen::Matrix<double, 3, 3> rotation = T.matrix().block(0, 0, 3, 3);

  for (const TrackId& id : shared_track_ids) {
    if (landmarks.count(id) == 0) {
      FeatureId p_id = feature_tracks.at(id).at(fcid0);
      FeatureId q_id = feature_tracks.at(id).at(fcid1);

      Eigen::Vector2d p = feature_corners.at(fcid0).corners[p_id];
      Eigen::Vector2d q = feature_corners.at(fcid1).corners[q_id];

      bearingVectors1.emplace_back(
          calib_cam.intrinsics[fcid0.cam_id]->unproject(p));
      bearingVectors2.emplace_back(
          calib_cam.intrinsics[fcid1.cam_id]->unproject(q));

      new_track_ids.emplace_back(id);
    }
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors1, bearingVectors2, translation, rotation);
  for (size_t id = 0; id < new_track_ids.size(); id++) {
    opengv::point_t point = opengv::triangulation::triangulate(adapter, id);
    Eigen::Vector3d p_world = cameras.at(fcid0).T_w_c * point;

    Landmark l;
    l.p = p_world;

    for (const auto& kv : feature_tracks.at(new_track_ids[id])) {
      if (cameras.find(kv.first) != cameras.end())
        l.obs[kv.first] = feature_tracks.at(new_track_ids[id]).at(kv.first);
    }
    landmarks[new_track_ids[id]] = l;
  }
  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  // UNUSED(calib_cam)
  // UNUSED(feature_corners);
  // UNUSED(feature_tracks);
  // UNUSED(cameras);
  // UNUSED(landmarks);

  Camera c_left, c_right;

  Eigen::Matrix4d identity_matrix = Eigen::Matrix4d::Identity();
  Sophus::SE3d se3(identity_matrix);
  c_left.T_w_c = se3;
  c_right.T_w_c = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  cameras[fcid0] = c_left;
  cameras[fcid1] = c_right;

  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  //   UNUSED(fcid);
  //   UNUSED(shared_track_ids);
  //   UNUSED(calib_cam);
  //   UNUSED(feature_corners);
  //   UNUSED(feature_tracks);
  //   UNUSED(landmarks);
  //   UNUSED(T_w_c);
  //   UNUSED(reprojection_error_pnp_inlier_threshold_pixel);

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (const TrackId& id : shared_track_ids) {
    points.emplace_back(landmarks.at(id).p);

    FeatureId pixel_id = feature_tracks.at(id).at(fcid);
    Eigen::Vector2d pixel = feature_corners.at(fcid).corners[pixel_id];
    bearingVectors.emplace_back(
        calib_cam.intrinsics[fcid.cam_id]->unproject(pixel));
  }

  // create the central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan2(reprojection_error_pnp_inlier_threshold_pixel, 500.0));
  ransac.max_iterations_ = 1000;
  ransac.computeModel();

  Eigen::Matrix<double, 3, 3> R = ransac.model_coefficients_.block(0, 0, 3, 3);
  Eigen::Matrix<double, 3, 1> T = ransac.model_coefficients_.block(0, 3, 3, 1);

  adapter.sett(T);
  adapter.setR(R);

  Eigen::Matrix<double, 3, 4> refined_T =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
  std::vector<int> refined_inliers;
  ransac.sac_model_->selectWithinDistance(refined_T, ransac.threshold_,
                                          refined_inliers);
  Eigen::Matrix<double, 3, 3> Rotation = refined_T.block(0, 0, 3, 3);
  Eigen::Matrix<double, 3, 1> Translation = refined_T.block(0, 3, 3, 1);

  // Eigen::Matrix<double, 3, 3> Rotation =
  //     ransac.model_coefficients_.block(0, 0, 3, 3);
  // Eigen::Matrix<double, 3, 1> Translation =
  //     ransac.model_coefficients_.block(0, 3, 3, 1);

  Sophus::SE3d refined_pose(Rotation, Translation);
  T_w_c = refined_pose;

  std::cout << refined_inliers.size() << std::endl;

  for (size_t i = 0; i < refined_inliers.size(); i++) {
    inlier_track_ids.emplace_back(shared_track_ids[refined_inliers[i]]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  // UNUSED(feature_corners);
  // UNUSED(options);
  // UNUSED(fixed_cameras);
  // UNUSED(calib_cam);
  // UNUSED(cameras);
  // UNUSED(landmarks);

  for (auto& cam : cameras) {
    FrameCamId fcid = cam.first;
    std::string cam_model = calib_cam.intrinsics[fcid.cam_id]->name();

    for (auto& l : landmarks) {
      if (l.second.obs.count(fcid) != 0 &&
          l.second.outlier_obs.count(fcid) == 0) {
        FeatureId id = l.second.obs[fcid];
        Eigen::Vector2d pixel = feature_corners.at(fcid).corners[id];

        problem.AddParameterBlock(cam.second.T_w_c.data(),
                                  Sophus::SE3d::num_parameters,
                                  new Sophus::test::LocalParameterizationSE3);
        problem.AddParameterBlock(l.second.p.data(), 3);

        if (fixed_cameras.count(fcid) != 0)
          problem.SetParameterBlockConstant(cam.second.T_w_c.data());

        problem.AddParameterBlock(calib_cam.intrinsics[fcid.cam_id]->data(), 8);

        if (!options.optimize_intrinsics) {
          problem.SetParameterBlockConstant(
              calib_cam.intrinsics[fcid.cam_id]->data());
        }

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
            BundleAdjustmentReprojectionCostFunctor, 2,
            Sophus::SE3d::num_parameters, 3, 8>(
            new BundleAdjustmentReprojectionCostFunctor(pixel, cam_model));

        if (!options.use_huber) {
          problem.AddResidualBlock(cost_function, NULL, cam.second.T_w_c.data(),
                                   l.second.p.data(),
                                   calib_cam.intrinsics[fcid.cam_id]->data());
        } else {
          problem.AddResidualBlock(
              cost_function, new ceres::HuberLoss(options.huber_parameter),
              cam.second.T_w_c.data(), l.second.p.data(),
              calib_cam.intrinsics[fcid.cam_id]->data());
        }
      }
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
