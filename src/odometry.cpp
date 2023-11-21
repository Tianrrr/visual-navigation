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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

#include <basalt/imu/imu_types.h>
#include <basalt/imu/preintegration.h>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
bool next_step();
void optimize();
void compute_projections();

void load_data_with_imu(const std::string& dataset_path,
                        const std::string& calib_path);

void preintegrate(FrameId fi, FrameId fj);

Sophus::SE3d predict_pose(FrameId i, FrameId j,
                          basalt::PoseVelState<double> Rvp_i);

void FrameManager(const bool take_keyframe);
void bundle_adjustment(const BundleAdjustmentOptions& options,
                       const FrameId& fixed_frame);
void save_pose();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////
int prev_frame = 0;
int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = false;
TrackId next_landmark_id = 0;

/// intrinsic calibration
Calibration calib_cam;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;

std::vector<basalt::IntegratedImuMeasurement<double>::Ptr> v_iim;

Sophus::SE3d T_0_1;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// landmark positions that were removed from the current map
Landmarks old_landmarks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 20);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// euroc imu io
///////////////////////////////////////////////////////////////////////////////

std::map<FrameId, Sophus::SE3d> key_pose;
// std::vector<FrameId> keys;
// std::unordered_map<Timestamp, std::vector<double>> groundtruth;
std::vector<Timestamp> s_timestamps;

std::vector<Timestamp> image_timestamps;

std::vector<Timestamp> imu_timestamps;
std::vector<Timestamp> imu_timestamps2;

tbb::concurrent_unordered_map<int64_t, std::string> image_path;

std::vector<basalt::ImuData<double>,
            Eigen::aligned_allocator<basalt::ImuData<double>>>
    imu0_data;
std::vector<basalt::ImuData<double>,
            Eigen::aligned_allocator<basalt::ImuData<double>>>
    imu0_data2;

IMU_CAM calib_cam_imu;

Sophus::SE3d T_i_cam0;

Sophus::SE3d T_i_cam1;

Sophus::SE3d T_w_imu0;

Eigen::Vector3d bg;
Eigen::Vector3d ba;

const Eigen::Vector3d gw(0., 0., -9.81);

Eigen::Vector3d accel_noise_std;

Eigen::Vector3d gyro_noise_std;

std::vector<FrameId> key_frames;
std::vector<FrameId> f_frames;
int max_keyNum = 7;
int max_fNum = 3;

States states;
double scale = 1;

std::map<FrameCamId, LandmarkMatchData> fcl_md;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/MH_01_easy/mav0";
  // std::string cam_calib = "opt_calib.json";
  std::string cam_calib = "calibration.json";

  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--scale", scale, "scale. Default: 1.0");
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  std::cout << "scale: " << scale << std::endl;

  load_data_with_imu(dataset_path, cam_calib);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        auto cam_id = static_cast<CamId>(show_cam1);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        auto cam_id = static_cast<CamId>(show_cam2);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
    }
    save_pose();
  }

  return 0;
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  auto frame_id =
      static_cast<FrameId>(view_id == 0 ? show_frame1 : show_frame2);
  auto cam_id = static_cast<CamId>(view_id == 0 ? show_cam1 : show_cam2);

  FrameCamId fcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(fcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(fcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(fcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(fcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(fcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(fcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (0 == fcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (cam.first == fcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == fcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.cam_id == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
    }
    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const bool in_cam_1 = kv_lm.second.obs.count(fcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(fcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(fcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(fcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  if (show_old_points3d && old_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : old_landmarks) {
      glColor3ubv(color_old_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available
// load images, cam_calibration, imu_calibration, imu_measurements
void load_data_with_imu(const std::string& dataset_path,
                        const std::string& calib_path) {
  // read data from csv

  // 1. read imag timestep
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        image_timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  s_timestamps = image_timestamps;

  // 2. read imu data
  {
    std::string line;
    line.clear();
    const std::string imu_path = dataset_path + "/imu0/";

    // accel_data.clear();
    // gyro_data.clear();
    imu0_data.clear();

    std::ifstream fi(imu_path + "data.csv");

    while (std::getline(fi, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Vector3d gyro, accel;

      ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
          tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];

      basalt::ImuData<double> cur_imu;
      cur_imu.t_ns = timestamp;
      cur_imu.accel = accel;
      cur_imu.gyro = gyro;

      imu0_data.push_back(cur_imu);
      imu_timestamps.push_back(timestamp);
    }
  }

  imu0_data2 = imu0_data;
  imu_timestamps2 = imu_timestamps;

  std::cout << "Loaded " << imu0_data.size() << " imudata" << std::endl;

  // 3. load cam and imu calibrations
  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);

      archive(calib_cam_imu);

      calib_cam_imu.intrinsics.clear();
      calib_cam_imu.calib_accel_bias.resize(3);
      calib_cam_imu.calib_gyro_bias.resize(3);

      std::cout << "Loaded camera from " << calib_path << " with models ";

      int index = 0;
      for (auto& cam_imu : calib_cam_imu.cam_imu_intr) {
        Eigen::Matrix<double, 8, 1> intr;
        std::string cam_type = cam_imu.camera_type;
        intr << cam_imu.intr_map["fx"], cam_imu.intr_map["fy"],
            cam_imu.intr_map["cx"], cam_imu.intr_map["cy"],
            cam_imu.intr_map["xi"], cam_imu.intr_map["alpha"], 0, 0;

        std::shared_ptr<AbstractCamera<double>> cam =
            AbstractCamera<double>::from_data(cam_type, intr.data());
        cam->width() = calib_cam_imu.resolution[index][0];
        cam->height() = calib_cam_imu.resolution[index][1];

        calib_cam_imu.intrinsics.push_back(cam);
        index++;
      }
      std::cout << std::endl;

      T_i_cam0 = calib_cam_imu.T_imu_cam[0];
      T_i_cam1 = calib_cam_imu.T_imu_cam[1];
      T_0_1 = T_i_cam0.inverse() * T_i_cam1;

      ba << calib_cam_imu.calib_accel_bias[0],
          calib_cam_imu.calib_accel_bias[1], calib_cam_imu.calib_accel_bias[2];

      bg << calib_cam_imu.calib_gyro_bias[0], calib_cam_imu.calib_gyro_bias[1],
          calib_cam_imu.calib_gyro_bias[2];

      accel_noise_std << calib_cam_imu.accel_noise_std[0],
          calib_cam_imu.accel_noise_std[1], calib_cam_imu.accel_noise_std[2];

      gyro_noise_std << calib_cam_imu.gyro_noise_std[0],
          calib_cam_imu.gyro_noise_std[1], calib_cam_imu.gyro_noise_std[2];

      Eigen::Vector3d a_imu_0 = imu0_data[0].accel - ba;

      T_w_imu0.setQuaternion(Eigen::Quaternion<double>::FromTwoVectors(
          a_imu_0, Eigen::Vector3d::UnitZ()));
      T_w_imu0.translation() = Eigen::Vector3d::Zero();

      current_pose = T_w_imu0 * T_i_cam0;

      std::cout << "finish loading data" << std::endl;

    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  // std::cout << imu0_data.size() << std::endl;
  calib_cam.intrinsics = calib_cam_imu.intrinsics;
  calib_cam.T_i_c = calib_cam_imu.T_imu_cam;

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  if (current_frame >= int(images.size()) / NUM_CAMS) {
    for (auto& fid : key_frames) {
      key_pose[fid] = states[fid].T_w_i;
    }
    return false;
  }

  if (current_frame == 0) {
    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "KF Projected " << projected_track_ids.size() << " points."
              << std::endl;

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[fcidr]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << "stereo-matches."
              << std::endl;

    feature_corners[fcidl] = kdl;
    feature_corners[fcidr] = kdr;
    feature_matches[std::make_pair(fcidl, fcidr)] = md_stereo;

    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    Eigen::Matrix<double, 3, 1> v_w_imu0;
    v_w_imu0 << 0, 0, 0;
    basalt::PoseVelState<double> state(
        image_timestamps[0], current_pose * T_i_cam0.inverse(), v_w_imu0);

    // states存放imu到世界坐标的转换
    states[current_frame] = state;

    add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md,
                      landmarks, next_landmark_id);

    key_frames.emplace_back(current_frame);

    for (const auto& state : states) {
      cameras[FrameCamId(state.first, 0)].T_w_c = current_pose;
      cameras[FrameCamId(state.first, 1)].T_w_c = current_pose * T_0_1;
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    compute_projections();

    prev_frame = current_frame;

    current_frame++;

    return true;
  } else {
    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);
    std::cout << "#############current_frame: " << current_frame
              << "####################" << std::endl;

    f_frames.emplace_back(current_frame);

    if ((f_frames.size() + key_frames.size()) > 1) {
      Sophus::SE3d pre_T_w_i =
          predict_pose(prev_frame, current_frame, states[prev_frame]);
      current_pose = pre_T_w_i * T_i_cam0;
    }

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "Projected " << projected_track_ids.size() << " points."
              << std::endl;

    KeypointsData kdl;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);

    feature_corners[fcidl] = kdl;

    LandmarkMatchData md;
    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    std::cout << "Found " << md.inliers.size() << " match inliers."
              << std::endl;

    // 第一个md是F1的
    fcl_md[fcidl] = md;

    current_pose = md.T_w_c;

    basalt::PoseVelState<double> state(image_timestamps[current_frame],
                                       current_pose * T_i_cam0.inverse(),
                                       states[current_frame - 1].vel_w_i);

    std::cout << "vel: " << state.vel_w_i.transpose() << std::endl;

    // states存放imu到世界坐标的状态
    states[current_frame] = state;

    cameras[FrameCamId(current_frame, 0)].T_w_c = current_pose;
    cameras[FrameCamId(current_frame, 1)].T_w_c = current_pose * T_0_1;

    if (current_frame > 1) {
      if (f_frames.size() > 1)
        preintegrate(f_frames[f_frames.size() - 2],
                     f_frames[f_frames.size() - 1]);

      optimize();
      std::cout << "-----------------------------------------------------"
                << std::endl;
      std::cout << "-----------------------------------------------------"
                << std::endl;
      current_pose = cameras[fcidl].T_w_c;
    }

    // 判断需不需要关键帧，如果不需要，就运行manager去除第一个f_frame帧
    if (int(md.inliers.size()) < new_kf_min_inliers &&
        std::find(key_frames.begin(), key_frames.end(), prev_frame) ==
            key_frames.end()) {
      // if (int(fcl_md[FrameCamId(*f_frames.begin(), 0)].inliers.size()) <
      //     new_kf_min_inliers) {
      take_keyframe = true;
      // 需要关键帧，将f_frame第一帧变成关键帧

      FrameManager(take_keyframe);

      take_keyframe = false;

    } else {
      FrameManager(take_keyframe);
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    prev_frame = current_frame;
    current_frame++;
    return true;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  std::cout << "--------optimization---------" << std::endl;
  std::cout << "keyframes: ";
  for (auto& k : key_frames) {
    std::cout << k << " ";
  }
  std::cout << "\nfree_frames: ";
  for (auto& f : f_frames) {
    std::cout << f << " ";
  }
  std::cout << std::endl;

  size_t num_obs = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with "
            << (2 * key_frames.size() + f_frames.size()) << " cameras, "
            << landmarks.size() << " points and " << num_obs << " observations."
            << std::endl;
  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole
  // second
  // camera constant is a bit suboptimal, since we only need 1 DoF, but it's
  // simple and the initial poses should be good from calibration.
  FrameId fixed_fid = *(key_frames.begin());
  // std::cout << "fid " << fid << std::endl;

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 50;
  ba_options.verbosity_level = ba_verbose;

  bundle_adjustment(ba_options, fixed_fid);

  for (const auto& state : states) {
    cameras[FrameCamId(state.first, 0)].T_w_c = state.second.T_w_i * T_i_cam0;
    cameras[FrameCamId(state.first, 1)].T_w_c = state.second.T_w_i * T_i_cam1;
  }

  // Update project info cache
  compute_projections();
}

void preintegrate(FrameId fi, FrameId fj) {
  Timestamp ti = image_timestamps[fi];
  Timestamp tj = image_timestamps[fj];

  auto it_i = std::find(imu_timestamps.begin(), imu_timestamps.end(), ti);
  int index_i = std::distance(imu_timestamps.begin(), it_i);

  auto it_j = std::find(imu_timestamps.begin(), imu_timestamps.end(), tj);
  int index_j = std::distance(imu_timestamps.begin(), it_j);

  basalt::IntegratedImuMeasurement<double>::Ptr iim{
      new basalt::IntegratedImuMeasurement<double>(imu0_data[index_i].t_ns, bg,
                                                   ba)};

  Eigen::Vector3d accel_cov = accel_noise_std.array() * accel_noise_std.array();
  Eigen::Vector3d gyro_cov = gyro_noise_std.array() * gyro_noise_std.array();

  for (int id = index_i + 1; id < index_j + 1; id++) {
    iim->integrate(imu0_data[id], accel_cov, gyro_cov);
  }
  v_iim.emplace_back(iim);
  if (v_iim.size() != (f_frames.size() - 1)) {
    std::cerr << " v_iim 的size不对" << std::endl;
  }

  imu_timestamps.erase(imu_timestamps.begin(),
                       imu_timestamps.begin() + index_j);
  imu0_data.erase(imu0_data.begin(), imu0_data.begin() + index_j);
}

void FrameManager(const bool take_keyframe) {
  FrameId F1 = *f_frames.begin();

  if (take_keyframe) {
    key_frames.emplace_back(F1);
    f_frames.erase(f_frames.begin());
    if (v_iim.size() > 0) {
      v_iim.erase(v_iim.begin());
      if (v_iim.size() != (f_frames.size() - 1)) {
        std::cerr << " v_iim 的size不对" << std::endl;
      }
    }

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    FrameCamId F1_0(F1, 0), F1_1(F1, 1);

    // 拿到kdl和kdr做匹配
    kdl = feature_corners[F1_0];
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[F1_1]);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    feature_corners[F1_1] = kdr;
    feature_matches[std::make_pair(F1_0, F1_1)] = md_stereo;

    LandmarkMatchData md = fcl_md[F1_0];

    add_new_landmarks(F1_0, F1_1, kdl, kdr, calib_cam, md_stereo, md, landmarks,
                      next_landmark_id);

    fcl_md.erase(FrameCamId(F1, 0));

    if ((int)key_frames.size() > max_keyNum) {
      FrameId K1 = *key_frames.begin();
      key_pose[K1] = states[K1].T_w_i;
      key_frames.erase(key_frames.begin());
      cameras.erase(FrameCamId(K1, 0));
      cameras.erase(FrameCamId(K1, 1));
      states.erase(K1);
      if (fcl_md.count(FrameCamId(K1, 0)) > 0) fcl_md.erase(FrameCamId(K1, 0));
      for (auto& l : landmarks) {
        if (l.second.obs.count(FrameCamId(K1, 0)) != 0)
          l.second.obs.erase(FrameCamId(K1, 0));
        if (l.second.obs.count(FrameCamId(K1, 1)) != 0)
          l.second.obs.erase(FrameCamId(K1, 1));
      }
      for (auto it = landmarks.begin(); it != landmarks.end();) {
        if (it->second.obs.size() == 0) {
          old_landmarks[it->first] = it->second;
          it = landmarks.erase(it);
        } else {
          ++it;
        }
      }
    }

    for (FrameId& f : f_frames) {
      FrameCamId fcidl(f, 0), fcidr(f, 1);
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          projected_points;
      std::vector<TrackId> projected_track_ids;

      Sophus::SE3d pose = states[f].T_w_i * T_i_cam0;

      project_landmarks(pose, calib_cam.intrinsics[0], landmarks,
                        cam_z_threshold, projected_points, projected_track_ids);

      LandmarkMatchData md;
      find_matches_landmarks(feature_corners[fcidl], landmarks, feature_corners,
                             projected_points, projected_track_ids,
                             match_max_dist_2d, feature_match_max_dist,
                             feature_match_test_next_best, md);

      localize_camera(pose, calib_cam.intrinsics[0], feature_corners[fcidl],
                      landmarks, reprojection_error_pnp_inlier_threshold_pixel,
                      md);

      // 新的md.inliers
      fcl_md[fcidl] = md;
    }

    compute_projections();

  } else {
    if ((int)f_frames.size() > max_fNum - 1) {
      f_frames.erase(f_frames.begin());
      cameras.erase(FrameCamId(F1, 0));
      cameras.erase(FrameCamId(F1, 1));
      states.erase(F1);
      fcl_md.erase(FrameCamId(F1, 0));
      if (v_iim.size() > 0) {
        v_iim.erase(v_iim.begin());
        if (v_iim.size() != (f_frames.size() - 1)) {
          std::cerr << " v_iim 的size不对" << std::endl;
        }
      }
    }
  }
}

void bundle_adjustment(const BundleAdjustmentOptions& options,
                       const FrameId& fixed_frame) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  // UNUSED(feature_corners);
  // UNUSED(options);
  // UNUSED(fixed_cameras);
  // UNUSED(calib_cam);
  // UNUSED(cameras);
  // UNUSED(landmarks);

  for (auto& k_frame : key_frames) {
    FrameCamId fcidl(k_frame, 0);
    FrameCamId fcidr(k_frame, 1);
    std::vector<FrameCamId> v_fcid;
    v_fcid.emplace_back(fcidl);
    v_fcid.emplace_back(fcidr);

    int cam_lr = 0;
    for (auto& fcid : v_fcid) {
      std::string cam_model = calib_cam.intrinsics[fcid.cam_id]->name();

      for (auto& l : landmarks) {
        if (l.second.obs.count(fcid) != 0 &&
            l.second.outlier_obs.count(fcid) == 0) {
          FeatureId id = l.second.obs[fcid];
          Eigen::Vector2d pixel = feature_corners.at(fcid).corners[id];

          problem.AddParameterBlock(states[k_frame].T_w_i.data(),
                                    Sophus::SE3d::num_parameters,
                                    new Sophus::test::LocalParameterizationSE3);
          problem.AddParameterBlock(l.second.p.data(), 3);

          if (k_frame == fixed_frame)
            problem.SetParameterBlockConstant(states[k_frame].T_w_i.data());

          problem.AddParameterBlock(calib_cam.intrinsics[fcid.cam_id]->data(),
                                    8);

          if (!options.optimize_intrinsics) {
            problem.SetParameterBlockConstant(
                calib_cam.intrinsics[fcid.cam_id]->data());
          }

          ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
              BundleAdjustmentReprojectionCostFunctor, 2,
              Sophus::SE3d::num_parameters, 3, 8>(
              new BundleAdjustmentReprojectionCostFunctor(
                  pixel, cam_model, calib_cam.T_i_c[cam_lr], scale));

          if (!options.use_huber) {
            problem.AddResidualBlock(
                cost_function, NULL, states[k_frame].T_w_i.data(),
                l.second.p.data(), calib_cam.intrinsics[fcid.cam_id]->data());
          } else {
            problem.AddResidualBlock(
                cost_function, new ceres::HuberLoss(options.huber_parameter),
                states[k_frame].T_w_i.data(), l.second.p.data(),
                calib_cam.intrinsics[fcid.cam_id]->data());
          }
        }
      }
      cam_lr++;
    }
  }

  for (FrameId frame : f_frames) {
    FrameCamId fcid(frame, 0);
    LandmarkMatchData md = fcl_md[fcid];
    std::string cam_model = calib_cam.intrinsics[fcid.cam_id]->name();

    for (auto& match : md.inliers) {
      FeatureId id = match.first;
      TrackId tid = match.second;
      if (landmarks.count(tid) > 0) {
        Eigen::Vector2d pixel = feature_corners.at(fcid).corners[id];
        problem.AddParameterBlock(states[frame].T_w_i.data(),
                                  Sophus::SE3d::num_parameters,
                                  new Sophus::test::LocalParameterizationSE3);

        problem.AddParameterBlock(landmarks.at(tid).p.data(), 3);

        problem.AddParameterBlock(calib_cam.intrinsics[fcid.cam_id]->data(), 8);

        if (!options.optimize_intrinsics) {
          problem.SetParameterBlockConstant(
              calib_cam.intrinsics[fcid.cam_id]->data());
        }

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
            BundleAdjustmentReprojectionCostFunctor, 2,
            Sophus::SE3d::num_parameters, 3, 8>(
            new BundleAdjustmentReprojectionCostFunctor(
                pixel, cam_model, calib_cam.T_i_c[0], scale));

        if (!options.use_huber) {
          problem.AddResidualBlock(cost_function, NULL,
                                   states[frame].T_w_i.data(),
                                   landmarks.at(tid).p.data(),
                                   calib_cam.intrinsics[fcid.cam_id]->data());
        } else {
          problem.AddResidualBlock(
              cost_function, new ceres::HuberLoss(options.huber_parameter),
              states[frame].T_w_i.data(), landmarks.at(tid).p.data(),
              calib_cam.intrinsics[fcid.cam_id]->data());
        }
      }
    }
  }

  for (size_t i = 0; i < v_iim.size(); i++) {
    auto& s0 = states[f_frames[i]];
    auto& s1 = states[f_frames[i + 1]];

    problem.AddParameterBlock(s0.vel_w_i.data(), 3);
    // if (i == 0) problem.SetParameterBlockConstant(s0.vel_w_i.data());

    problem.AddParameterBlock(s0.T_w_i.data(), Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    problem.AddParameterBlock(s1.vel_w_i.data(), 3);
    problem.AddParameterBlock(s1.T_w_i.data(), Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<PreImuCostFunctor, 9, 3,
                                        Sophus::SE3d::num_parameters, 3,
                                        Sophus::SE3d::num_parameters>(
            new PreImuCostFunctor(v_iim[i], bg, ba, gw));

    if (!options.use_huber) {
      problem.AddResidualBlock(cost_function, NULL, s0.vel_w_i.data(),
                               s0.T_w_i.data(), s1.vel_w_i.data(),
                               s1.T_w_i.data());
    } else {
      problem.AddResidualBlock(cost_function,
                               new ceres::HuberLoss(options.huber_parameter),
                               s0.vel_w_i.data(), s0.T_w_i.data(),
                               s1.vel_w_i.data(), s1.T_w_i.data());
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

void save_pose() {
  // save keypose
  {
    std::ofstream ofs;
    ofs.open("/home/parallels/visnav/keypose.txt", std::ios::out);

    for (auto& frame_pose : key_pose) {
      Eigen::Vector4d quaternion =
          frame_pose.second.unit_quaternion().coeffs().transpose();
      Eigen::Vector3d translation = frame_pose.second.translation().transpose();

      double t = double(s_timestamps[frame_pose.first]) * 1e-9;
      ofs << std::fixed << std::setprecision(9) << t;

      ofs << " " << translation[0] << " " << translation[1] << " "
          << translation[2] << " " << quaternion[1] << " " << quaternion[2]
          << " " << quaternion[3] << " " << quaternion[0] << std::endl;
    }
    ofs.close();
  }
}

Sophus::SE3d predict_pose(FrameId fi, FrameId fj,
                          basalt::PoseVelState<double> Rvp_i) {
  Timestamp ti = image_timestamps[fi];
  Timestamp tj = image_timestamps[fj];

  auto it_i = std::find(imu_timestamps.begin(), imu_timestamps.end(), ti);
  int index_i = std::distance(imu_timestamps.begin(), it_i);

  auto it_j = std::find(imu_timestamps.begin(), imu_timestamps.end(), tj);
  int index_j = std::distance(imu_timestamps.begin(), it_j);

  basalt::IntegratedImuMeasurement<double>::Ptr iim{
      new basalt::IntegratedImuMeasurement<double>(imu0_data[index_i].t_ns, bg,
                                                   ba)};

  for (int id = index_i + 1; id < index_j + 1; id++) {
    iim->integrate(imu0_data[id], accel_noise_std, gyro_noise_std);
  }

  basalt::PoseVelState<double> Rvp_j;

  iim->predictState(Rvp_i, gw, Rvp_j);

  return Rvp_j.T_w_i;
}
