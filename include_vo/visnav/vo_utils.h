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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  // UNUSED(current_pose);
  // UNUSED(cam);
  // UNUSED(landmarks);
  // UNUSED(cam_z_threshold);
  for (auto& l : landmarks) {
    Eigen::Vector3d l_cam = current_pose.inverse() * l.second.p;

    if (l_cam.z() < cam_z_threshold) continue;

    Eigen::Vector2d p_2d = cam->project(l_cam);

    if (p_2d.x() > cam->width() || p_2d.y() > cam->height() || p_2d.x() < 0 ||
        p_2d.y() < 0)
      continue;

    projected_points.emplace_back(p_2d);
    projected_track_ids.emplace_back(l.first);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  // UNUSED(kdl);
  // UNUSED(landmarks);
  // UNUSED(feature_corners);
  // UNUSED(projected_points);
  // UNUSED(projected_track_ids);
  // UNUSED(match_max_dist_2d);
  // UNUSED(feature_match_threshold);
  // UNUSED(feature_match_dist_2_best);
  for (size_t key_id = 0; key_id < kdl.corners.size(); key_id++) {
    std::bitset<256> key_descriptor = kdl.corner_descriptors[key_id];
    Eigen::Vector2d keypoint = kdl.corners[key_id];

    int best_match_dist = 256;
    int sec_best_match_dist = 256;
    TrackId best_track_idx;

    for (size_t index = 0; index < projected_track_ids.size(); index++) {
      double match_dist = (projected_points[index] - keypoint).norm();
      if (match_dist >= match_max_dist_2d) continue;

      int current_dist = 256;
      TrackId current_trackid = -1;

      TrackId trackid = projected_track_ids[index];

      for (auto& lm : landmarks.at(trackid).obs) {
        FrameCamId fcid = lm.first;
        FeatureId featid = lm.second;
        std::bitset<256> lm_descriptor =
            feature_corners.at(fcid).corner_descriptors[featid];

        int dist = (key_descriptor ^ lm_descriptor).count();
        if (dist < current_dist) {
          current_dist = dist;
          current_trackid = trackid;
        }
      }

      if (current_dist < best_match_dist) {
        sec_best_match_dist = best_match_dist;
        best_match_dist = current_dist;
        best_track_idx = current_trackid;
      } else if (current_dist < sec_best_match_dist) {
        sec_best_match_dist = current_dist;
      }
    }

    if (best_match_dist < feature_match_threshold &&
        sec_best_match_dist >= feature_match_dist_2_best * best_match_dist) {
      md.matches.emplace_back(key_id, best_track_idx);
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.
  // UNUSED(cam);
  // UNUSED(kdl);
  // UNUSED(landmarks);
  // UNUSED(reprojection_error_pnp_inlier_threshold_pixel);

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (auto& m : md.matches) {
    Eigen::Vector2d p_2d = kdl.corners[m.first];
    bearingVectors.emplace_back(cam->unproject(p_2d));
    points.emplace_back(landmarks.at(m.second).p);
  }

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

  Sophus::SE3d refined_pose(Rotation, Translation);
  md.T_w_c = refined_pose;

  for (size_t i = 0; i < refined_inliers.size(); i++) {
    md.inliers.emplace_back(md.matches[refined_inliers[i]]);
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.
  // UNUSED(fcidl);
  // UNUSED(fcidr);
  // UNUSED(kdl);
  // UNUSED(kdr);
  // UNUSED(calib_cam);
  // UNUSED(md_stereo);
  // UNUSED(md);
  // UNUSED(landmarks);
  // UNUSED(next_landmark_id);
  // UNUSED(t_0_1);
  // UNUSED(R_0_1);

  for (auto& kv : md.inliers) {
    landmarks[kv.second].obs[fcidl] = kv.first;
    for (auto& stereo : md_stereo.inliers) {
      if (stereo.first == kv.first) {
        landmarks[kv.second].obs[fcidr] = stereo.second;
        break;
      }
    }
  }

  opengv::bearingVectors_t bearingVectors1;
  opengv::bearingVectors_t bearingVectors2;
  std::vector<std::pair<FeatureId, FeatureId>> v;

  for (auto& stereo : md_stereo.inliers) {
    bool exist = 0;
    for (auto& kv : md.inliers) {
      if (stereo.first == kv.first) {
        exist = 1;
        break;
      }
    }
    if (exist) continue;

    Eigen::Vector2d p = kdl.corners[stereo.first];
    Eigen::Vector2d q = kdr.corners[stereo.second];

    bearingVectors1.emplace_back(
        calib_cam.intrinsics[fcidl.cam_id]->unproject(p));
    bearingVectors2.emplace_back(
        calib_cam.intrinsics[fcidr.cam_id]->unproject(q));
    v.emplace_back(stereo.first, stereo.second);
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors1, bearingVectors2, t_0_1, R_0_1);

  for (size_t id = 0; id < bearingVectors1.size(); id++) {
    opengv::point_t point = opengv::triangulation::triangulate(adapter, id);

    Eigen::Vector3d p_world = md.T_w_c * point;
    Landmark l;
    l.p = p_world;
    l.obs[fcidl] = v[id].first;
    l.obs[fcidr] = v[id].second;
    next_landmark_id++;
    landmarks[next_landmark_id] = l;
  }
}

// void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
//                           Cameras& cameras, Landmarks& landmarks,
//                           Landmarks& old_landmarks,
//                           std::set<FrameId>& kf_frames) {
//   kf_frames.emplace(fcidl.frame_id);

//   // TODO SHEET 5: Remove old cameras and observations if the number of
//   keyframe
//   // pairs (left and right image is a pair) is larger than max_num_kfs. The
//   ids
//   // of all the keyframes that are currently in the optimization should be
//   // stored in kf_frames. Removed keyframes should be removed from cameras
//   and
//   // landmarks with no left observations should be moved to old_landmarks.
//   // UNUSED(max_num_kfs);
//   // UNUSED(cameras);
//   // UNUSED(landmarks);
//   // UNUSED(old_landmarks);

//   while ((int)kf_frames.size() > max_num_kfs) {
//     FrameId frame = *kf_frames.begin();
//     kf_frames.erase(frame);
//     cameras.erase(FrameCamId(frame, 0));
//     cameras.erase(FrameCamId(frame, 1));
//     for (auto& l : landmarks) {
//       if (l.second.obs.count(FrameCamId(frame, 0)) != 0)
//         l.second.obs.erase(FrameCamId(frame, 0));
//       if (l.second.obs.count(FrameCamId(frame, 1)) != 0)
//         l.second.obs.erase(FrameCamId(frame, 1));
//     }
//   }

//   for (auto it = landmarks.begin(); it != landmarks.end();) {
//     if (it->second.obs.size() == 0) {
//       old_landmarks[it->first] = it->second;
//       it = landmarks.erase(it);
//     } else {
//       ++it;
//     }
//   }
// }
}  // namespace visnav
