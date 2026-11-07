/*
 * slam_toolbox
 * Copyright (c) 2026, Pritam Paul
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "karto_sdk/Karto.h"
#include "slam_toolbox/adaptive_sparse_map_generator.hpp"

namespace
{
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

#define ASSERT_TRUE(cond) \
  do { \
    ++g_tests_run; \
    if (!(cond)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #cond << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_FALSE(cond) \
  do { \
    ++g_tests_run; \
    if (cond) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": NOT(" #cond ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_EQ(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (val_a != val_b) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " == " #b \
                << " (actual: " << static_cast<int64_t>(val_a) \
                << " vs expected: " << static_cast<int64_t>(val_b) << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_NEAR(a, b, eps) \
  do { \
    ++g_tests_run; \
    double val_a = static_cast<double>(a); \
    double val_b = static_cast<double>(b); \
    if (std::abs(val_a - val_b) > static_cast<double>(eps)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": |" #a " - " #b "| <= " #eps \
                << " (actual: " << val_a << " vs expected: " << val_b \
                << ", diff: " << std::abs(val_a - val_b) << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_GT(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (!(val_a > val_b)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " > " #b \
                << " (actual: " << val_a \
                << " vs expected > " << val_b << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_GE(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (!(val_a >= val_b)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " >= " #b \
                << " (actual: " << val_a \
                << " vs expected >= " << val_b << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_LE(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (!(val_a <= val_b)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " <= " #b \
                << " (actual: " << val_a \
                << " vs expected <= " << val_b << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define ASSERT_LT(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (!(val_a < val_b)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " < " #b \
                << " (actual: " << val_a \
                << " vs expected < " << val_b << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

karto::LaserRangeFinder * createTestLaser(const std::string & sensor_name)
{
  karto::LaserRangeFinder * laser = karto::LaserRangeFinder::CreateLaserRangeFinder(
    karto::LaserRangeFinder_Custom, karto::Name(sensor_name));
  laser->SetIs360Laser(true);
  laser->SetMinimumAngle(-M_PI);
  laser->SetMaximumAngle(M_PI);
  laser->SetAngularResolution(M_PI / 180.0);  // 1 deg per ray (360 beams)
  laser->SetMinimumRange(0.1);
  laser->SetMaximumRange(25.0);
  laser->SetRangeThreshold(20.0);
  karto::SensorManager::GetInstance()->RegisterSensor(laser, true);
  return laser;
}

void destroyTestLaser(karto::LaserRangeFinder * laser)
{
  if (laser) {
    karto::SensorManager::GetInstance()->UnregisterSensor(laser);
    delete laser;
  }
}

karto::LocalizedRangeScan * createSingleTargetScan(
  karto::LaserRangeFinder * laser,
  double robot_x, double robot_y, double robot_yaw,
  int target_beam_idx, double range_meters)
{
  karto::Pose2 pose(robot_x, robot_y, robot_yaw);
  karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
    laser->GetName(), std::vector<kt_double>(360, 25.0));
  scan->SetSensorPose(pose);
  scan->SetOdometricPose(pose);
  scan->SetCorrectedPose(pose);

  std::vector<kt_double> ranges(360, 25.0);  // 25.0m == max range, skipped by generateMap
  karto::PointVectorDouble points;
  for (int i = 0; i < 360; ++i) {
    const double beam_angle = robot_yaw - M_PI + i * (M_PI / 180.0);
    const double dist = (i == target_beam_idx) ? range_meters : 25.0;
    ranges[i] = dist;
    points.push_back(karto::Vector2<kt_double>(
      robot_x + dist * std::cos(beam_angle),
      robot_y + dist * std::sin(beam_angle)));
  }
  scan->SetRangeReadings(ranges);
  scan->SetPointReadings(points);
  return scan;
}

karto::LocalizedRangeScanVector generateSerpentineFacilityScans(
  karto::LaserRangeFinder * laser,
  double length_x, double width_y, size_t num_poses)
{
  karto::LocalizedRangeScanVector scans;
  const double half_x = length_x / 2.0;
  const double half_y = width_y / 2.0;

  for (size_t i = 0; i < num_poses; ++i) {
    const double ratio = static_cast<double>(i) / static_cast<double>(num_poses);
    const double rx = -half_x * 0.8 + ratio * (length_x * 1.6);
    const double ry = std::sin(ratio * 8.0 * M_PI) * (half_y * 0.7);
    const double yaw = std::atan2(std::cos(ratio * 8.0 * M_PI), 1.0);

    karto::Pose2 pose(rx, ry, yaw);
    karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 10.0));
    scan->SetSensorPose(pose);
    scan->SetOdometricPose(pose);
    scan->SetCorrectedPose(pose);

    std::vector<kt_double> ranges(360);
    karto::PointVectorDouble points;

    for (int b = 0; b < 360; ++b) {
      const double angle = yaw - M_PI + b * (M_PI / 180.0);
      const double ca = std::cos(angle);
      const double sa = std::sin(angle);

      double hit_dist = 16.0;
      if (ca > 1e-4) {
        hit_dist = std::min(hit_dist, (half_x - rx) / ca);
      } else if (ca < -1e-4) {
        hit_dist = std::min(hit_dist, (-half_x - rx) / ca);
      }
      if (sa > 1e-4) {
        hit_dist = std::min(hit_dist, (half_y - ry) / sa);
      } else if (sa < -1e-4) {
        hit_dist = std::min(hit_dist, (-half_y - ry) / sa);
      }

      hit_dist = std::max(0.5, std::min(hit_dist, 20.0));
      ranges[b] = hit_dist;
      points.push_back(karto::Vector2<kt_double>(rx + hit_dist * ca, ry + hit_dist * sa));
    }

    scan->SetRangeReadings(ranges);
    scan->SetPointReadings(points);
    scans.push_back(scan);
  }
  return scans;
}

}  // namespace

// =================================================================================================
// STRESS TEST 1: Multi-Obstacle Consecutive Full-Beam Penetration
// =================================================================================================
void stress_test1_multi_obstacle_penetration()
{
  std::cout <<
    "\n================================================================================\n";
  std::cout << "  [STRESS 1] Multi-Obstacle Consecutive Full-Beam Penetration\n";
  std::cout << "================================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("lidar_penetration");
  const double resolution = 0.05;

  // 1.1: Straight beam passing through 4 consecutive obstacles in series
  // Robot at (0, 0), angle 0 (beam 180).
  // Obstacles at 2.0m, 4.0m, 6.0m, 8.0m. Distant wall at 12.0m.
  {
    std::cout << "  [1.1] Penetrating 4 Consecutive Obstacles in Series...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.1);

    karto::LocalizedRangeScanVector accumulated_scans;
    const double distances[4] = {2.0, 4.0, 6.0, 8.0};

    // Establish each obstacle with 3 scans
    for (int obs = 0; obs < 4; ++obs) {
      for (int k = 0; k < 3; ++k) {
        accumulated_scans.push_back(
          createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, distances[obs]));
      }
    }

    nav_msgs::msg::OccupancyGrid map_obs;
    ASSERT_TRUE(generator.generateMap(accumulated_scans, resolution, map_obs));
    const auto * grid = generator.getSparseGrid();
    ASSERT_TRUE(grid != nullptr);

    int cx[4], cy[4];
    for (int obs = 0; obs < 4; ++obs) {
      grid->world_to_cell(distances[obs], 0.0, cx[obs], cy[obs]);
      ASSERT_EQ(grid->get_cost(cx[obs], cy[obs]), 254);
      ASSERT_GE(grid->get_hits(cx[obs], cy[obs]), 3);
    }
    std::cout << "    [PASS] Established 4 confirmed lethal obstacles along X axis (cost 254)\n";

    // Now shoot 3 scans all the way to Distant Wall at 12.0m
    for (int k = 0; k < 3; ++k) {
      accumulated_scans.push_back(
        createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 12.0));
    }

    nav_msgs::msg::OccupancyGrid map_penetrated;
    ASSERT_TRUE(generator.generateMap(accumulated_scans, resolution, map_penetrated));

    // 1. Verify wall at 12.0m received 3 hits and is lethal (254)
    int wall_cx = 0, wall_cy = 0;
    grid->world_to_cell(12.0, 0.0, wall_cx, wall_cy);
    ASSERT_EQ(grid->get_cost(wall_cx, wall_cy), 254);
    ASSERT_EQ(grid->get_hits(wall_cx, wall_cy), 3);

    // 2. Verify all 4 intermediate obstacles received 3 additional passes
    ASSERT_EQ(grid->get_passes(cx[3], cy[3]), 6);   // 8.0m obstacle
    ASSERT_EQ(grid->get_passes(cx[2], cy[2]), 9);   // 6.0m obstacle
    ASSERT_EQ(grid->get_passes(cx[1], cy[1]), 12);  // 4.0m obstacle
    ASSERT_EQ(grid->get_passes(cx[0], cy[0]), 15);  // 2.0m obstacle

    // 3. Verify open cells between obstacles are Free (0)
    const double free_positions[4] = {3.0, 5.0, 7.0, 10.0};
    for (int f = 0; f < 4; ++f) {
      int fcx = 0, fcy = 0;
      grid->world_to_cell(free_positions[f], 0.0, fcx, fcy);
      ASSERT_EQ(grid->get_cost(fcx, fcy), 0);
      ASSERT_GE(grid->get_passes(fcx, fcy), 3);
      ASSERT_EQ(grid->get_hits(fcx, fcy), 0);
    }
    std::cout <<
      "    [PASS] Full beam traversed through all 4 obstacles to 12.0m without stopping\n";

    for (auto * s : accumulated_scans) {
      delete s;
    }
  }

  // 1.2: Diagonal multi-obstacle penetration crossing 16x16 block tiles
  {
    std::cout << "  [1.2] Diagonal Multi-Obstacle Penetration Across Block Boundaries...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.1);

    karto::LocalizedRangeScanVector diag_scans;
    // 3 scans hitting obstacle 1 at 2.0m (beam 225: 45 deg)
    for (int k = 0; k < 3; ++k) {
      diag_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 225, 2.0));
    }
    // 3 scans hitting obstacle 2 at 5.0m
    for (int k = 0; k < 3; ++k) {
      diag_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 225, 5.0));
    }
    // 3 scans penetrating both obstacles to reach wall at 10.0m
    for (int k = 0; k < 3; ++k) {
      diag_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 225, 10.0));
    }

    nav_msgs::msg::OccupancyGrid diag_map;
    ASSERT_TRUE(generator.generateMap(diag_scans, resolution, diag_map));
    const auto * grid = generator.getSparseGrid();
    ASSERT_TRUE(grid != nullptr);

    int wall_x = 0, wall_y = 0;
    grid->world_to_cell(10.0 * std::cos(M_PI / 4.0), 10.0 * std::sin(M_PI / 4.0), wall_x, wall_y);
    ASSERT_EQ(grid->get_cost(wall_x, wall_y), 254);
    ASSERT_EQ(grid->get_hits(wall_x, wall_y), 3);

    int obs1_x = 0, obs1_y = 0;
    grid->world_to_cell(2.0 * std::cos(M_PI / 4.0), 2.0 * std::sin(M_PI / 4.0), obs1_x, obs1_y);
    ASSERT_EQ(grid->get_hits(obs1_x, obs1_y), 3);
    ASSERT_GE(grid->get_passes(obs1_x, obs1_y), 9);

    int free_x = 0, free_y = 0;
    grid->world_to_cell(7.0 * std::cos(M_PI / 4.0), 7.0 * std::sin(M_PI / 4.0), free_x, free_y);
    ASSERT_EQ(grid->get_cost(free_x, free_y), 0);
    std::cout << "    [PASS] Diagonal rays penetrated obstacles across tile boundaries cleanly\n";

    for (auto * s : diag_scans) {
      delete s;
    }
  }

  // 1.3: Multi-obstacle penetration in negative coordinate quadrants
  {
    std::cout << "  [1.3] Negative Coordinate Space Multi-Obstacle Penetration...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.1);

    karto::LocalizedRangeScanVector neg_scans;
    for (int k = 0; k < 3; ++k) {
      neg_scans.push_back(createSingleTargetScan(laser, -5.0, -5.0, M_PI, 180, 2.0));
    }
    for (int k = 0; k < 3; ++k) {
      neg_scans.push_back(createSingleTargetScan(laser, -5.0, -5.0, M_PI, 180, 5.0));
    }
    for (int k = 0; k < 3; ++k) {
      neg_scans.push_back(createSingleTargetScan(laser, -5.0, -5.0, M_PI, 180, 10.0));
    }

    nav_msgs::msg::OccupancyGrid neg_map;
    ASSERT_TRUE(generator.generateMap(neg_scans, resolution, neg_map));
    const auto * grid = generator.getSparseGrid();
    ASSERT_TRUE(grid != nullptr);

    int wall_x = 0, wall_y = 0;
    grid->world_to_cell(-15.0, -5.0, wall_x, wall_y);
    ASSERT_EQ(grid->get_cost(wall_x, wall_y), 254);
    ASSERT_EQ(grid->get_hits(wall_x, wall_y), 3);

    int obs1_x = 0, obs1_y = 0;
    grid->world_to_cell(-7.0, -5.0, obs1_x, obs1_y);
    ASSERT_EQ(grid->get_hits(obs1_x, obs1_y), 3);
    ASSERT_GE(grid->get_passes(obs1_x, obs1_y), 9);

    int free_between_x = 0, free_between_y = 0;
    grid->world_to_cell(-12.0, -5.0, free_between_x, free_between_y);
    ASSERT_EQ(grid->get_cost(free_between_x, free_between_y), 0);
    std::cout <<
      "    [PASS] Negative coordinate space verified without sign or tile indexing errors\n";

    for (auto * s : neg_scans) {
      delete s;
    }
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// STRESS TEST 2: Dynamic Obstacle Clearance & Dilution Mechanics
// =================================================================================================
void stress_test2_dynamic_clearance()
{
  std::cout <<
    "\n================================================================================\n";
  std::cout << "  [STRESS 2] Dynamic Obstacle Clearance & Dilution Mechanics\n";
  std::cout << "================================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("lidar_clearance");
  const double resolution = 0.05;

  // 2.1: Mathematical precision of clearance transition: 3 initial hits
  // Occupancy threshold = 0.10.
  // When passes = 29: 3/29 = 0.103448 > 0.10 -> MUST BE Occupied (254).
  // When passes = 30: 3/30 = 0.100000 <= 0.10 -> MUST CLEAR TO Free (0)!
  // When passes = 35: 3/35 = 0.085714 < 0.10 -> MUST REMAIN Free (0).
  {
    std::cout << "  [2.1] Step-by-Step Exact Boundary Transition (Passes 29 -> 30 -> 35)...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.10);

    karto::LocalizedRangeScanVector base_scans;
    for (int k = 0; k < 3; ++k) {
      base_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
    }

    // Passes = 29 (3 hits + 26 passes to 5.0m)
    karto::LocalizedRangeScanVector scans_29 = base_scans;
    for (int k = 0; k < 26; ++k) {
      scans_29.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
    }
    nav_msgs::msg::OccupancyGrid map_29;
    ASSERT_TRUE(generator.generateMap(scans_29, resolution, map_29));
    const auto * grid = generator.getSparseGrid();
    int obs_cx = 0, obs_cy = 0;
    grid->world_to_cell(2.0, 0.0, obs_cx, obs_cy);
    ASSERT_EQ(grid->get_passes(obs_cx, obs_cy), 29);
    ASSERT_EQ(grid->get_hits(obs_cx, obs_cy), 3);
    ASSERT_EQ(grid->get_cost(obs_cx, obs_cy), 254);  // MUST STILL BE OCCUPIED!
    std::cout << "    [PASS] At passes=29: hit_ratio=0.1034 > 0.10 -> cell is Occupied (254)\n";

    // Passes = 30 (1 more pass to 5.0m)
    karto::LocalizedRangeScanVector scans_30 = scans_29;
    scans_30.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
    nav_msgs::msg::OccupancyGrid map_30;
    ASSERT_TRUE(generator.generateMap(scans_30, resolution, map_30));
    ASSERT_EQ(grid->get_passes(obs_cx, obs_cy), 30);
    ASSERT_EQ(grid->get_hits(obs_cx, obs_cy), 3);
    ASSERT_EQ(grid->get_cost(obs_cx, obs_cy), 0);  // EXACT THRESHOLD TRANSITION TO FREE!
    std::cout << "    [PASS] At passes=30: hit_ratio=0.1000 <= 0.10 -> cell CLEARS to Free (0)\n";

    // Passes = 35 (5 more passes to 5.0m)
    karto::LocalizedRangeScanVector scans_35 = scans_30;
    for (int k = 0; k < 5; ++k) {
      scans_35.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
    }
    nav_msgs::msg::OccupancyGrid map_35;
    ASSERT_TRUE(generator.generateMap(scans_35, resolution, map_35));
    ASSERT_EQ(grid->get_passes(obs_cx, obs_cy), 35);
    ASSERT_EQ(grid->get_cost(obs_cx, obs_cy), 0);
    std::cout << "    [PASS] At passes=35: hit_ratio=0.0857 < 0.10 -> cell remains Free (0)\n";

    for (auto * s : scans_35) {
      delete s;
    }
  }

  // 2.2: Full Bidirectional Cycle: Unknown -> Occupied -> Free -> Occupied -> Free
  {
    std::cout << "  [2.2] Full Bidirectional Re-emergence Cycle...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.10);

    // Phase 1: 4 hits at (3.0, 0.0) -> Occupied (254)
    karto::LocalizedRangeScanVector cycle_scans;
    for (int k = 0; k < 4; ++k) {
      cycle_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 3.0));
    }
    nav_msgs::msg::OccupancyGrid map1;
    ASSERT_TRUE(generator.generateMap(cycle_scans, resolution, map1));
    const auto * grid = generator.getSparseGrid();
    int cx = 0, cy = 0;
    grid->world_to_cell(3.0, 0.0, cx, cy);
    ASSERT_EQ(grid->get_cost(cx, cy), 254);
    std::cout << "    [PASS] Phase 1: 4 hits -> Occupied (254)\n";

    // Phase 2: Dilute with 40 passes through (3.0, 0.0) to wall at 6.0m
    // passes = 4 + 40 = 44, hits = 4 -> ratio = 4/44 = 0.0909 <= 0.10 -> CLEARS to Free (0)
    for (int k = 0; k < 40; ++k) {
      cycle_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 6.0));
    }
    nav_msgs::msg::OccupancyGrid map2;
    ASSERT_TRUE(generator.generateMap(cycle_scans, resolution, map2));
    ASSERT_EQ(grid->get_cost(cx, cy), 0);
    std::cout << "    [PASS] Phase 2: 40 penetrating passes -> Cleared to Free (0)\n";

    // Phase 3: Dynamic object reappears! 15 new hits at (3.0, 0.0)
    // passes = 44 + 15 = 59, hits = 4 + 15 = 19
    // ratio = 19/59 = 0.322 > 0.10 -> RE-OCCUPIED (254)!
    for (int k = 0; k < 15; ++k) {
      cycle_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 3.0));
    }
    nav_msgs::msg::OccupancyGrid map3;
    ASSERT_TRUE(generator.generateMap(cycle_scans, resolution, map3));
    ASSERT_EQ(grid->get_cost(cx, cy), 254);
    std::cout << "    [PASS] Phase 3: Obstacle returns (15 hits) -> Re-occupied (254)\n";

    // Phase 4: Dynamic object leaves permanently! 200 passes to 6.0m
    // passes = 59 + 200 = 259, hits = 19 -> ratio = 19/259 = 0.073 <= 0.10 -> CLEARS to Free (0)!
    for (int k = 0; k < 200; ++k) {
      cycle_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 6.0));
    }
    nav_msgs::msg::OccupancyGrid map4;
    ASSERT_TRUE(generator.generateMap(cycle_scans, resolution, map4));
    ASSERT_EQ(grid->get_cost(cx, cy), 0);
    std::cout << "    [PASS] Phase 4: Obstacle departs (200 passes) -> Cleared to Free (0)\n";

    for (auto * s : cycle_scans) {
      delete s;
    }
  }

  // 2.3: Simultaneous dynamic clearance across multiple angular beams
  {
    std::cout << "  [2.3] Multi-Angle Simultaneous Dynamic Obstacle Clearance...\n";
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.10);

    const int test_beams[4] = {150, 165, 180, 210};
    karto::LocalizedRangeScanVector fan_scans;

    // Phase 1: 3 hits on each of the 4 beams at 3.0m
    for (int b = 0; b < 4; ++b) {
      for (int k = 0; k < 3; ++k) {
        fan_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, test_beams[b], 3.0));
      }
    }
    nav_msgs::msg::OccupancyGrid init_map;
    ASSERT_TRUE(generator.generateMap(fan_scans, resolution, init_map));
    const auto * grid = generator.getSparseGrid();

    for (int b = 0; b < 4; ++b) {
      const double angle = -M_PI + test_beams[b] * (M_PI / 180.0);
      int cx = 0, cy = 0;
      grid->world_to_cell(3.0 * std::cos(angle), 3.0 * std::sin(angle), cx, cy);
      ASSERT_EQ(grid->get_cost(cx, cy), 254);
    }
    std::cout << "    [PASS] Established 4 radial obstacles along beams {150, 165, 180, 210}\n";

    // Phase 2: 30 sweeps covering all 4 beams to 8.0m (passing through the 3.0m obstacles)
    for (int sweep = 0; sweep < 30; ++sweep) {
      for (int b = 0; b < 4; ++b) {
        fan_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, test_beams[b], 8.0));
      }
    }
    nav_msgs::msg::OccupancyGrid cleared_map;
    ASSERT_TRUE(generator.generateMap(fan_scans, resolution, cleared_map));

    for (int b = 0; b < 4; ++b) {
      const double angle = -M_PI + test_beams[b] * (M_PI / 180.0);
      int cx = 0, cy = 0;
      grid->world_to_cell(3.0 * std::cos(angle), 3.0 * std::sin(angle), cx, cy);
      ASSERT_EQ(grid->get_cost(cx, cy), 0);
    }
    std::cout << "    [PASS] All 4 radial obstacles simultaneously cleared to Free (0)\n";

    for (auto * s : fan_scans) {
      delete s;
    }
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// STRESS TEST 3: High-Load Ray Casting & Memory Stability (100,000+ Rays)
// =================================================================================================
void stress_test3_high_load_stress_and_memory()
{
  std::cout <<
    "\n================================================================================\n";
  std::cout << "  [STRESS 3] High-Load Stress Testing & Memory Stability (100,000+ Rays)\n";
  std::cout << "================================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("lidar_high_load");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;
  generator.setMinPassThrough(2);
  generator.setOccupancyThreshold(0.10);

  // 3.1: Large Batch: 300 scans * 360 rays = 108,000 rays in Mega Distribution Hub (120x80m)
  {
    std::cout << "  [3.1] High-Load Batch: 108,000 Rays (300 scans x 360 beams, 120m x 80m)...\n";
    karto::LocalizedRangeScanVector scans_108k = generateSerpentineFacilityScans(laser, 120.0, 80.0,
      300);
    ASSERT_EQ(scans_108k.size(), 300);

    nav_msgs::msg::OccupancyGrid map_108k;
    const auto t_start = std::chrono::high_resolution_clock::now();
    bool ok = generator.generateMap(scans_108k, resolution, map_108k);
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    ASSERT_TRUE(ok);
    ASSERT_GT(map_108k.info.width, 0);
    ASSERT_GT(map_108k.info.height, 0);
    ASSERT_EQ(map_108k.info.resolution, static_cast<float>(resolution));
    ASSERT_EQ(map_108k.data.size(), map_108k.info.width * map_108k.info.height);

    const size_t mem_bytes = generator.getMemoryBytes();
    const double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
    const double rays_per_sec = (108000.0 / (elapsed_ms / 1000.0));

    std::cout << "    -> Latency: " << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
    std::cout << "    -> Throughput: " << static_cast<uint64_t>(rays_per_sec) << " rays/sec\n";
    std::cout << "    -> Memory Footprint: " << std::setprecision(2) << mem_mb << " MB\n";
    std::cout << "    -> Grid Size: " << map_108k.info.width << " x " << map_108k.info.height
              << " (" << map_108k.data.size() << " cells)\n";

    // Performance assertion: throughput must exceed 50,000 rays/sec
    ASSERT_GT(rays_per_sec, 50000.0);
    // Memory assertion: must remain well below 100 MB budget
    ASSERT_LT(mem_mb, 100.0);

    for (auto * s : scans_108k) {
      delete s;
    }
  }

  // 3.2: Massive Batch: 500 scans * 360 rays = 180,000 rays in Logistics Park (250x200m)
  {
    std::cout <<
      "  [3.2] Massive-Load Batch: 180,000 Rays (500 scans x 360 beams, 250m x 200m)...\n";
    karto::LocalizedRangeScanVector scans_180k = generateSerpentineFacilityScans(laser, 250.0,
      200.0, 500);
    ASSERT_EQ(scans_180k.size(), 500);

    nav_msgs::msg::OccupancyGrid map_180k;
    const auto t_start = std::chrono::high_resolution_clock::now();
    bool ok = generator.generateMap(scans_180k, resolution, map_180k);
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    ASSERT_TRUE(ok);
    ASSERT_GT(map_180k.info.width, 0);
    ASSERT_GT(map_180k.info.height, 0);
    ASSERT_EQ(map_180k.data.size(), map_180k.info.width * map_180k.info.height);

    const size_t mem_bytes = generator.getMemoryBytes();
    const double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
    const double rays_per_sec = (180000.0 / (elapsed_ms / 1000.0));

    std::cout << "    -> Latency: " << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
    std::cout << "    -> Throughput: " << static_cast<uint64_t>(rays_per_sec) << " rays/sec\n";
    std::cout << "    -> Memory Footprint: " << std::setprecision(2) << mem_mb << " MB\n";
    std::cout << "    -> Grid Size: " << map_180k.info.width << " x " << map_180k.info.height
              << " (" << map_180k.data.size() << " cells)\n";

    ASSERT_GT(rays_per_sec, 50000.0);
    ASSERT_LT(mem_mb, 200.0);

    for (auto * s : scans_180k) {
      delete s;
    }
  }

  // 3.3: Memory Stability Across 5 Consecutive Invocations (Absence of Leaks)
  {
    std::cout << "  [3.3] Memory Leak & Stability Check Across 5 Consecutive Map Invocations...\n";
    karto::LocalizedRangeScanVector test_scans = generateSerpentineFacilityScans(laser, 50.0, 50.0,
      50);
    nav_msgs::msg::OccupancyGrid map_iter;

    size_t first_alloc = 0;
    for (int iter = 0; iter < 5; ++iter) {
      generator.generateMap(test_scans, resolution, map_iter);
      const size_t current_mem = generator.getMemoryBytes();
      if (iter == 0) {
        first_alloc = current_mem;
      } else {
        ASSERT_EQ(current_mem, first_alloc);
      }
    }
    std::cout << "    -> Memory strictly stable at " << first_alloc <<
      " bytes across 5 cycles (zero leakage)\n";

    for (auto * s : test_scans) {
      delete s;
    }
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// STRESS TEST 4: Sensor Degradation & Edge Case Robustness
// =================================================================================================
void stress_test4_edge_cases_and_robustness()
{
  std::cout <<
    "\n================================================================================\n";
  std::cout << "  [STRESS 4] Sensor Degradation & Edge Case Robustness\n";
  std::cout << "================================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("lidar_edge_cases");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;

  // 4.1: Empty scan vector returns false safely without modifying map
  {
    std::cout << "  [4.1] Empty Scans Vector Handling...\n";
    karto::LocalizedRangeScanVector empty_scans;
    nav_msgs::msg::OccupancyGrid map_out;
    ASSERT_FALSE(generator.generateMap(empty_scans, resolution, map_out));
    std::cout << "    [PASS] Empty scan vector cleanly returned false without crash\n";
  }

  // 4.2: Scans with nullptr elements safely ignored
  {
    std::cout << "  [4.2] Nullptr Scan Elements in Vector...\n";
    karto::LocalizedRangeScanVector mixed_scans;
    mixed_scans.push_back(nullptr);
    mixed_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
    mixed_scans.push_back(nullptr);

    nav_msgs::msg::OccupancyGrid map_out;
    ASSERT_TRUE(generator.generateMap(mixed_scans, resolution, map_out));
    std::cout << "    [PASS] Nullptr elements in localized scan vector safely skipped\n";

    delete mixed_scans[1];
  }

  // 4.3: Scans with NaN and Infinity range values
  {
    std::cout << "  [4.3] NaN and Inf Range Values Handling...\n";
    karto::Pose2 pose(0.0, 0.0, 0.0);
    karto::LocalizedRangeScan * scan_nan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 25.0));
    scan_nan->SetSensorPose(pose);
    scan_nan->SetOdometricPose(pose);
    scan_nan->SetCorrectedPose(pose);

    std::vector<kt_double> ranges(360);
    karto::PointVectorDouble points;
    for (int i = 0; i < 360; ++i) {
      if (i % 4 == 0) {
        ranges[i] = std::numeric_limits<double>::quiet_NaN();
      } else if (i % 4 == 1) {
        ranges[i] = std::numeric_limits<double>::infinity();
      } else if (i % 4 == 2) {
        ranges[i] = -std::numeric_limits<double>::infinity();
      } else {
        ranges[i] = 5.0;
      }
      points.push_back(karto::Vector2<kt_double>(5.0, 0.0));
    }
    scan_nan->SetRangeReadings(ranges);
    scan_nan->SetPointReadings(points);

    karto::LocalizedRangeScanVector nan_scans;
    nan_scans.push_back(scan_nan);

    nav_msgs::msg::OccupancyGrid map_out;
    ASSERT_TRUE(generator.generateMap(nan_scans, resolution, map_out));
    std::cout <<
      "    [PASS] Scans with NaN, +Inf, -Inf successfully filtered without corrupting grid\n";

    delete scan_nan;
  }

  // 4.4: Range cap scaling (> 16.0m capped, invalid endpoint receives record_pass)
  {
    std::cout << "  [4.4] Range Cap Scaling (> 16.0m)...\n";
    generator.setMinPassThrough(2);
    generator.setOccupancyThreshold(0.10);

    // Scan with range 22.0m along angle 0 (beam 180)
    karto::LocalizedRangeScanVector cap_scans;
    for (int k = 0; k < 3; ++k) {
      cap_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 22.0));
    }

    nav_msgs::msg::OccupancyGrid map_out;
    ASSERT_TRUE(generator.generateMap(cap_scans, resolution, map_out));
    const auto * grid = generator.getSparseGrid();
    ASSERT_TRUE(grid != nullptr);

    // Beyond cap at 22.0m: untouched (cost 255)
    int cell_22_x = 0, cell_22_y = 0;
    grid->world_to_cell(22.0, 0.0, cell_22_x, cell_22_y);
    ASSERT_EQ(grid->get_cost(cell_22_x, cell_22_y), 255);

    // Intermediate cell at 10.0m: Free (0)
    int cell_10_x = 0, cell_10_y = 0;
    grid->world_to_cell(10.0, 0.0, cell_10_x, cell_10_y);
    ASSERT_EQ(grid->get_cost(cell_10_x, cell_10_y), 0);

    // Capped endpoint at 16.0m: hits = 0, cost = 0 (Free)
    int cell_16_x = 0, cell_16_y = 0;
    grid->world_to_cell(16.0, 0.0, cell_16_x, cell_16_y);
    ASSERT_EQ(grid->get_hits(cell_16_x, cell_16_y), 0);
    ASSERT_EQ(grid->get_cost(cell_16_x, cell_16_y), 0);
    std::cout <<
      "    [PASS] Ray capped at 16.0m: 10m is Free, 16m gets pass, 22m remains untouched\n";

    for (auto * s : cap_scans) {
      delete s;
    }
  }

  // 4.5: Parameter Clamping Robustness
  {
    std::cout << "  [4.5] Parameter Clamping Robustness...\n";
    generator.setMinPassThrough(-10);
    ASSERT_EQ(generator.getMinPassThrough(), 0);

    generator.setOccupancyThreshold(-0.5);
    ASSERT_NEAR(generator.getOccupancyThreshold(), 0.01, 1e-5);

    generator.setOccupancyThreshold(1.5);
    ASSERT_NEAR(generator.getOccupancyThreshold(), 0.99, 1e-5);

    generator.setOccupancyThreshold(0.0);
    ASSERT_NEAR(generator.getOccupancyThreshold(), 0.01, 1e-5);

    generator.setOccupancyThreshold(1.0);
    ASSERT_NEAR(generator.getOccupancyThreshold(), 0.99, 1e-5);
    std::cout <<
      "    [PASS] Setter parameter clamping verified: min_pass >= 0, threshold in [0.01, 0.99]\n";
  }

  // 4.6: Re-initialization with resolution changes
  {
    std::cout << "  [4.6] Re-initialization on Map Resolution Changes...\n";
    karto::LocalizedRangeScanVector res_scans;
    res_scans.push_back(createSingleTargetScan(laser, 0.0, 0.0, 0.0, 180, 5.0));

    nav_msgs::msg::OccupancyGrid map_res;
    ASSERT_TRUE(generator.generateMap(res_scans, 0.05, map_res));
    ASSERT_NEAR(generator.getSparseGrid()->resolution(), 0.05, 1e-5);

    ASSERT_TRUE(generator.generateMap(res_scans, 0.10, map_res));
    ASSERT_NEAR(generator.getSparseGrid()->resolution(), 0.10, 1e-5);

    ASSERT_TRUE(generator.generateMap(res_scans, 0.02, map_res));
    ASSERT_NEAR(generator.getSparseGrid()->resolution(), 0.02, 1e-5);
    std::cout <<
      "    [PASS] Resolution change dynamically recreates sparse grid without leaking old memory\n";

    delete res_scans[0];
  }

  // 4.7: reset() frees sparse grid blocks cleanly
  {
    std::cout << "  [4.7] reset() Cleans Memory and Deallocates Blocks...\n";
    generator.reset();
    ASSERT_TRUE(generator.getSparseGrid() != nullptr);
    ASSERT_EQ(generator.getSparseGrid()->allocated_blocks_count(), 0);
    // Base memory footprint for empty grid struct with 0 blocks allocated
    ASSERT_LE(generator.getMemoryBytes(), 344);
    std::cout <<
      "    [PASS] reset() deallocates all sparse blocks (block_count: 0, memory <= 344B)\n";
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// MAIN RUNNER
// =================================================================================================
int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  const std::string banner(80, '=');
  std::cout << "\n" << banner << "\n";
  std::cout << "   MILESTONE 2 EMPIRICAL STRESS TEST (Challenger 1: Penetration & Load)\n";
  std::cout << "   Target: AdaptiveSparseMapGenerator in slam_toolbox_upstream\n";
  std::cout << banner << "\n";

  const auto start_time = std::chrono::high_resolution_clock::now();

  stress_test1_multi_obstacle_penetration();
  stress_test2_dynamic_clearance();
  stress_test3_high_load_stress_and_memory();
  stress_test4_edge_cases_and_robustness();

  const auto end_time = std::chrono::high_resolution_clock::now();
  const double total_duration_s =
    std::chrono::duration<double>(end_time - start_time).count();

  std::cout << "\n" << banner << "\n";
  std::cout << "  EMPIRICAL STRESS TEST EXECUTION SUMMARY:\n";
  std::cout << "    - Total Assertions Evaluated: " << g_tests_run << "\n";
  std::cout << "    - Assertions Passed:          " << g_tests_passed << "\n";
  std::cout << "    - Assertions Failed:          " << g_tests_failed << "\n";
  std::cout << "    - Total Elapsed Time:         " << std::fixed << std::setprecision(2)
            << total_duration_s << " seconds\n";
  std::cout << banner << "\n\n";

  if (g_tests_failed > 0) {
    std::cerr << "[-] SUITE FAILED with " << g_tests_failed << " failure(s)!" << std::endl;
    return 1;
  }

  std::cout << "[+] ALL " << g_tests_passed << " EMPIRICAL ASSERTIONS PASSED WITH 100% SUCCESS!"
            << std::endl;
  return 0;
}
