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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "karto_sdk/Karto.h"
#include "slam_toolbox/adaptive_sparse_map_generator.hpp"
#include "slam_toolbox/dense_karto_map_generator.hpp"

namespace
{
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

#define STRESS_ASSERT(cond) \
  do { \
    ++g_tests_run; \
    if (!(cond)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #cond << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define STRESS_ASSERT_MSG(cond, msg) \
  do { \
    ++g_tests_run; \
    if (!(cond)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #cond " (" << msg << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

karto::LaserRangeFinder * createTestLaser(const std::string & name)
{
  karto::LaserRangeFinder * laser = karto::LaserRangeFinder::CreateLaserRangeFinder(
    karto::LaserRangeFinder_Custom, karto::Name(name));
  laser->SetMinimumAngle(-M_PI);
  laser->SetMaximumAngle(M_PI);
  laser->SetAngularResolution(M_PI / 180.0);  // 360 rays
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

karto::LocalizedRangeScan * createSingleRayScan(
  karto::LaserRangeFinder * laser,
  double sensor_x, double sensor_y, double sensor_yaw,
  int target_beam_idx, double range_meters)
{
  karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
    laser->GetName(), std::vector<kt_double>(360, 28.0));
  karto::Pose2 pose(sensor_x, sensor_y, sensor_yaw);
  scan->SetSensorPose(pose);
  scan->SetOdometricPose(pose);
  scan->SetCorrectedPose(pose);

  std::vector<kt_double> ranges(360, 28.0);
  ranges[target_beam_idx] = range_meters;

  karto::PointVectorDouble points;
  for (int b = 0; b < 360; ++b) {
    const double angle = sensor_yaw - M_PI + b * (M_PI / 180.0);
    const double r = ranges[b];
    points.push_back(karto::Vector2<kt_double>(
      sensor_x + r * std::cos(angle),
      sensor_y + r * std::sin(angle)));
  }

  scan->SetRangeReadings(ranges);
  scan->SetPointReadings(points);
  return scan;
}

}  // namespace

// =================================================================================================
// Test Suite 1: Numerical Boundary Conditions (H/P == T vs H/P == T +/- eps)
// =================================================================================================
void testNumericalBoundaries()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 1: Numerical Boundary Conditions (H/P == T, +/- eps)\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("boundary_lidar");
  const double resolution = 0.05;

  struct BoundaryTestCase
  {
    int hits;
    int passes;
    double threshold;
    uint8_t expected_cost;
    std::string description;
  };

  std::vector<BoundaryTestCase> test_cases = {
    // Exact equality cases (H/P == T) -> Must strictly evaluate to Free (0)
    {1, 100, 0.01, 0, "Exact Equality 1/100 == 0.01"},
    {5, 100, 0.05, 0, "Exact Equality 5/100 == 0.05"},
    {1, 10, 0.10, 0, "Exact Equality 1/10 == 0.10"},
    {2, 20, 0.10, 0, "Exact Equality 2/20 == 0.10"},
    {5, 50, 0.10, 0, "Exact Equality 5/50 == 0.10"},
    {10, 100, 0.10, 0, "Exact Equality 10/100 == 0.10"},
    {2, 10, 0.20, 0, "Exact Equality 2/10 == 0.20"},
    {20, 100, 0.20, 0, "Exact Equality 20/100 == 0.20"},
    {1, 4, 0.25, 0, "Exact Equality 1/4 == 0.25"},
    {25, 100, 0.25, 0, "Exact Equality 25/100 == 0.25"},
    {1, 2, 0.50, 0, "Exact Equality 1/2 == 0.50"},
    {5, 10, 0.50, 0, "Exact Equality 5/10 == 0.50"},
    {50, 100, 0.50, 0, "Exact Equality 50/100 == 0.50"},
    {3, 5, 0.60, 0, "Exact Equality 3/5 == 0.60"},
    {3, 4, 0.75, 0, "Exact Equality 3/4 == 0.75"},
    {4, 5, 0.80, 0, "Exact Equality 4/5 == 0.80"},
    {9, 10, 0.90, 0, "Exact Equality 9/10 == 0.90"},
    {19, 20, 0.95, 0, "Exact Equality 19/20 == 0.95"},
    {99, 100, 0.99, 0, "Exact Equality 99/100 == 0.99"},

    // Above-threshold cases (H/P == T + eps) -> Must evaluate to Occupied (254)
    {2, 100, 0.01, 254, "Above Threshold 2/100 (0.02 > 0.01)"},
    {6, 100, 0.05, 254, "Above Threshold 6/100 (0.06 > 0.05)"},
    {11, 100, 0.10, 254, "Above Threshold 11/100 (0.11 > 0.10)"},
    {101, 1000, 0.10, 254, "Above Threshold 101/1000 (0.101 > 0.10)"},
    {21, 100, 0.20, 254, "Above Threshold 21/100 (0.21 > 0.20)"},
    {26, 100, 0.25, 254, "Above Threshold 26/100 (0.26 > 0.25)"},
    {51, 100, 0.50, 254, "Above Threshold 51/100 (0.51 > 0.50)"},
    {61, 100, 0.60, 254, "Above Threshold 61/100 (0.61 > 0.60)"},
    {76, 100, 0.75, 254, "Above Threshold 76/100 (0.76 > 0.75)"},
    {81, 100, 0.80, 254, "Above Threshold 81/100 (0.81 > 0.80)"},
    {91, 100, 0.90, 254, "Above Threshold 91/100 (0.91 > 0.90)"},
    {96, 100, 0.95, 254, "Above Threshold 96/100 (0.96 > 0.95)"},
    {100, 100, 0.99, 254, "Above Threshold 100/100 (1.0 > 0.99)"},

    // Below-threshold cases (H/P == T - eps) -> Must evaluate to Free (0)
    {4, 100, 0.05, 0, "Below Threshold 4/100 (0.04 < 0.05)"},
    {9, 100, 0.10, 0, "Below Threshold 9/100 (0.09 < 0.10)"},
    {99, 1000, 0.10, 0, "Below Threshold 99/1000 (0.099 < 0.10)"},
    {19, 100, 0.20, 0, "Below Threshold 19/100 (0.19 < 0.20)"},
    {24, 100, 0.25, 0, "Below Threshold 24/100 (0.24 < 0.25)"},
    {49, 100, 0.50, 0, "Below Threshold 49/100 (0.49 < 0.50)"},
    {59, 100, 0.60, 0, "Below Threshold 59/100 (0.59 < 0.60)"},
    {74, 100, 0.75, 0, "Below Threshold 74/100 (0.74 < 0.75)"},
    {79, 100, 0.80, 0, "Below Threshold 79/100 (0.79 < 0.80)"},
    {89, 100, 0.90, 0, "Below Threshold 89/100 (0.89 < 0.90)"},
    {94, 100, 0.95, 0, "Below Threshold 94/100 (0.94 < 0.95)"},
    {98, 100, 0.99, 0, "Below Threshold 98/100 (0.98 < 0.99)"}
  };

  for (const auto & tc : test_cases) {
    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.setMinPassThrough(0);  // Allow immediate classification
    generator.setOccupancyThreshold(tc.threshold);

    // Target point at x = 2.0m, y = 0.0m (beam 180)
    const double target_x = 2.0;
    const double far_x = 5.0;

    karto::LocalizedRangeScanVector scans;

    // First produce hits (tc.hits times ray ends at 2.0m)
    for (int h = 0; h < tc.hits; ++h) {
      scans.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, target_x));
    }

    // Next produce clean passes (tc.passes - tc.hits times ray passes through 2.0m to 5.0m)
    const int passes_only = tc.passes - tc.hits;
    for (int p = 0; p < passes_only; ++p) {
      scans.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, far_x));
    }

    nav_msgs::msg::OccupancyGrid out_map;
    bool gen_ok = generator.generateMap(scans, resolution, out_map);
    STRESS_ASSERT(gen_ok);

    const auto * grid = generator.getSparseGrid();
    STRESS_ASSERT(grid != nullptr);

    int cell_x = 0;
    int cell_y = 0;
    grid->world_to_cell(target_x, 0.0, cell_x, cell_y);

    const uint32_t actual_passes = grid->get_passes(cell_x, cell_y);
    const uint32_t actual_hits = grid->get_hits(cell_x, cell_y);
    const uint8_t actual_cost = grid->get_cost(cell_x, cell_y);

    STRESS_ASSERT_MSG(actual_passes == static_cast<uint32_t>(tc.passes),
      tc.description + " passes mismatch: " + std::to_string(actual_passes) + " vs " +
      std::to_string(tc.passes));
    STRESS_ASSERT_MSG(actual_hits == static_cast<uint32_t>(tc.hits),
      tc.description + " hits mismatch: " + std::to_string(actual_hits) + " vs " +
      std::to_string(tc.hits));
    STRESS_ASSERT_MSG(actual_cost == tc.expected_cost,
      tc.description + " cost mismatch: got " + std::to_string(actual_cost) + " expected " +
      std::to_string(tc.expected_cost));

    if (actual_cost == tc.expected_cost) {
      std::cout << "  [PASS] " << tc.description << " -> Passes: " << actual_passes
                << ", Hits: " << actual_hits << ", Cost: " << static_cast<int>(actual_cost) << "\n";
    }

    for (auto * s : scans) {
      delete s;
    }
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// Test Suite 2: Extreme Values of min_pass_through (0, 1, 10, 50) and occupancy_threshold
// =================================================================================================
void testExtremeParameters()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 2: Extreme Values of min_pass_through and occupancy_threshold\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("extreme_lidar");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;

  // 1. Clamping tests
  generator.setMinPassThrough(-10);
  STRESS_ASSERT_MSG(generator.getMinPassThrough() == 0, "setMinPassThrough(-10) must clamp to 0");

  generator.setOccupancyThreshold(-0.5);
  STRESS_ASSERT_MSG(std::abs(generator.getOccupancyThreshold() - 0.01) < 1e-6,
    "setOccupancyThreshold(-0.5) must clamp to 0.01");

  generator.setOccupancyThreshold(1.5);
  STRESS_ASSERT_MSG(std::abs(generator.getOccupancyThreshold() - 0.99) < 1e-6,
    "setOccupancyThreshold(1.5) must clamp to 0.99");

  std::cout << "  [PASS] Parameter clamping verified: min_pass >= 0, threshold in [0.01, 0.99]\n";

  // 2. Testing min_pass_through values in {0, 1, 10, 50}
  const std::vector<int> min_pass_values = {0, 1, 10, 50};
  for (int mp : min_pass_values) {
    generator.setMinPassThrough(mp);
    generator.setOccupancyThreshold(0.10);
    STRESS_ASSERT(generator.getMinPassThrough() == mp);

    // Test noise filtration: provide exactly mp hits.
    // If mp == 0: 0 hits -> passes=0, cell is unknown 255.
    // If mp > 0: provide mp hits -> passes=mp <= mp -> must remain unknown 255!
    if (mp > 0) {
      karto::LocalizedRangeScanVector scans_mp;
      for (int i = 0; i < mp; ++i) {
        scans_mp.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
      }
      nav_msgs::msg::OccupancyGrid map_mp;
      generator.generateMap(scans_mp, resolution, map_mp);
      const auto * grid = generator.getSparseGrid();
      int cx = 0, cy = 0;
      grid->world_to_cell(2.0, 0.0, cx, cy);

      STRESS_ASSERT_MSG(grid->get_passes(cx, cy) == static_cast<uint32_t>(mp),
        "Passes mismatch for mp=" + std::to_string(mp));
      STRESS_ASSERT_MSG(grid->get_cost(cx, cy) == 255,
        "Cell should remain unknown (255) when passes == min_pass_through (" + std::to_string(mp) +
        ")");

      std::cout << "  [PASS] min_pass_through=" << mp << " filters " << mp
                << " observations as noise (cost remains 255)\n";

      // Now add 1 more hit -> passes = mp + 1 > mp -> must classify as Occupied (254)!
      scans_mp.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
      generator.generateMap(scans_mp, resolution, map_mp);
      STRESS_ASSERT_MSG(grid->get_passes(cx, cy) == static_cast<uint32_t>(mp + 1),
        "Passes should be mp+1");
      STRESS_ASSERT_MSG(grid->get_cost(cx, cy) == 254,
        "Cell should classify as Occupied (254) when passes > min_pass_through (" +
        std::to_string(mp) + ")");

      std::cout << "  [PASS] min_pass_through=" << mp << " classifies on " << (mp + 1)
                << "th observation (cost becomes 254)\n";

      for (auto * s : scans_mp) {
        delete s;
      }
    } else {
      // mp == 0: immediate classification on 1st observation!
      karto::LocalizedRangeScanVector scans_0;
      scans_0.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
      nav_msgs::msg::OccupancyGrid map_0;
      generator.generateMap(scans_0, resolution, map_0);
      const auto * grid = generator.getSparseGrid();
      int cx = 0, cy = 0;
      grid->world_to_cell(2.0, 0.0, cx, cy);

      STRESS_ASSERT_MSG(grid->get_passes(cx, cy) == 1, "Passes should be 1");
      STRESS_ASSERT_MSG(grid->get_cost(cx, cy) == 254,
        "mp=0 classifies on 1st hit as Occupied (254)");

      std::cout <<
        "  [PASS] min_pass_through=0 immediately classifies 1st observation as Occupied (254)\n";

      for (auto * s : scans_0) {
        delete s;
      }
    }
  }

  // 3. Testing occupancy_threshold values in {0.01, 0.05, 0.20, 0.50, 0.99}
  struct ThresholdBoundary
  {
    double threshold;
    int passes;
    int hits_for_free;
    int hits_for_occupied;
  };

  std::vector<ThresholdBoundary> threshold_tests = {
    {0.01, 100, 1, 2},   // 1/100 = 0.01 (Free), 2/100 = 0.02 (Occ)
    {0.05, 100, 5, 6},   // 5/100 = 0.05 (Free), 6/100 = 0.06 (Occ)
    {0.20, 10, 2, 3},    // 2/10 = 0.20 (Free), 3/10 = 0.30 (Occ)
    {0.50, 10, 5, 6},    // 5/10 = 0.50 (Free), 6/10 = 0.60 (Occ)
    {0.99, 100, 99, 100}  // 99/100 = 0.99 (Free), 100/100 = 1.0 (Occ)
  };

  for (const auto & tt : threshold_tests) {
    generator.setMinPassThrough(0);
    generator.setOccupancyThreshold(tt.threshold);

    // Test Free boundary
    {
      karto::LocalizedRangeScanVector scans_f;
      for (int h = 0; h < tt.hits_for_free; ++h) {
        scans_f.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
      }
      for (int p = 0; p < (tt.passes - tt.hits_for_free); ++p) {
        scans_f.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
      }
      nav_msgs::msg::OccupancyGrid map_f;
      generator.generateMap(scans_f, resolution, map_f);
      int cx = 0, cy = 0;
      generator.getSparseGrid()->world_to_cell(2.0, 0.0, cx, cy);
      STRESS_ASSERT_MSG(generator.getSparseGrid()->get_cost(cx, cy) == 0,
        "Threshold " + std::to_string(tt.threshold) + " with " + std::to_string(tt.hits_for_free) +
        "/" + std::to_string(tt.passes) + " should evaluate to Free (0)");
      for (auto * s : scans_f) {
        delete s;
      }
    }

    // Test Occupied boundary
    {
      karto::LocalizedRangeScanVector scans_o;
      for (int h = 0; h < tt.hits_for_occupied; ++h) {
        scans_o.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 2.0));
      }
      for (int p = 0; p < (tt.passes - tt.hits_for_occupied); ++p) {
        scans_o.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 5.0));
      }
      nav_msgs::msg::OccupancyGrid map_o;
      generator.generateMap(scans_o, resolution, map_o);
      int cx = 0, cy = 0;
      generator.getSparseGrid()->world_to_cell(2.0, 0.0, cx, cy);
      STRESS_ASSERT_MSG(generator.getSparseGrid()->get_cost(cx, cy) == 254,
        "Threshold " + std::to_string(tt.threshold) + " with " +
        std::to_string(tt.hits_for_occupied) +
        "/" + std::to_string(tt.passes) + " should evaluate to Occupied (254)");
      for (auto * s : scans_o) {
        delete s;
      }
    }

    std::cout << "  [PASS] occupancy_threshold=" << tt.threshold << " cleanly transitions from "
              << tt.hits_for_free << "/" << tt.passes << " (Free) to "
              << tt.hits_for_occupied << "/" << tt.passes << " (Occupied)\n";
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// Test Suite 3: Simultaneous Dynamic Obstacle Clearance & Multi-Beam Penetration
// =================================================================================================
void testDynamicClearanceMultiObstacles()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 3: Simultaneous Dynamic Obstacle Clearance & Penetration\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("dynamic_lidar");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;
  generator.setMinPassThrough(2);
  generator.setOccupancyThreshold(0.10);

  // Phase 1: Establish 8 distinct obstacles along different angles
  // (0, 45, 90, 135, 180, 225, 270, 315 deg)
  const std::vector<int> beams = {0, 45, 90, 135, 180, 225, 270, 315};
  const double obstacle_dist = 2.5;

  karto::LocalizedRangeScanVector scans;
  for (int iter = 0; iter < 4; ++iter) {
    karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 20.0));
    scan->SetSensorPose(karto::Pose2(0.0, 0.0, 0.0));
    std::vector<kt_double> ranges(360, 20.0);
    karto::PointVectorDouble points(360);

    for (int b = 0; b < 360; ++b) {
      ranges[b] = 20.0;
    }
    for (int b : beams) {
      ranges[b] = obstacle_dist;
    }

    for (int b = 0; b < 360; ++b) {
      const double angle = -M_PI + b * (M_PI / 180.0);
      const double r = ranges[b];
      points[b] = karto::Vector2<kt_double>(r * std::cos(angle), r * std::sin(angle));
    }
    scan->SetRangeReadings(ranges);
    scan->SetPointReadings(points);
    scans.push_back(scan);
  }

  nav_msgs::msg::OccupancyGrid map_init;
  generator.generateMap(scans, resolution, map_init);

  const auto * grid = generator.getSparseGrid();
  for (int b : beams) {
    const double angle = -M_PI + b * (M_PI / 180.0);
    int cx = 0, cy = 0;
    grid->world_to_cell(obstacle_dist * std::cos(angle), obstacle_dist * std::sin(angle), cx, cy);
    STRESS_ASSERT_MSG(grid->get_cost(cx, cy) == 254,
      "Obstacle at beam " + std::to_string(b) + " must be Occupied");
    STRESS_ASSERT_MSG(grid->get_hits(cx, cy) == 4, "Hits must be 4");
    STRESS_ASSERT_MSG(grid->get_passes(cx, cy) >= 4, "Passes must be >= 4");
  }
  std::cout << "  [PASS] Successfully established 8 simultaneous radial obstacles (cost 254)\n";

  // Phase 2: Cast 40 subsequent scans past these obstacles to 8.0m
  // passes become 4 + 40 = 44, hits remain 4. hit_ratio = 4/44 = 0.0909 <= 0.10 -> ALL CLEAR!
  const double far_dist = 8.0;
  for (int iter = 0; iter < 40; ++iter) {
    karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 20.0));
    scan->SetSensorPose(karto::Pose2(0.0, 0.0, 0.0));
    std::vector<kt_double> ranges(360, 20.0);
    karto::PointVectorDouble points(360);

    for (int b = 0; b < 360; ++b) {
      ranges[b] = 20.0;
    }
    for (int b : beams) {
      ranges[b] = far_dist;
    }

    for (int b = 0; b < 360; ++b) {
      const double angle = -M_PI + b * (M_PI / 180.0);
      const double r = ranges[b];
      points[b] = karto::Vector2<kt_double>(r * std::cos(angle), r * std::sin(angle));
    }
    scan->SetRangeReadings(ranges);
    scan->SetPointReadings(points);
    scans.push_back(scan);
  }

  nav_msgs::msg::OccupancyGrid map_cleared;
  generator.generateMap(scans, resolution, map_cleared);

  for (int b : beams) {
    const double angle = -M_PI + b * (M_PI / 180.0);
    int cx = 0, cy = 0;
    grid->world_to_cell(obstacle_dist * std::cos(angle), obstacle_dist * std::sin(angle), cx, cy);
    STRESS_ASSERT_MSG(grid->get_cost(cx, cy) == 0,
      "Obstacle at beam " + std::to_string(b) + " must clear to Free (0)");
    STRESS_ASSERT_MSG(grid->get_passes(cx, cy) >= 44, "Passes must be >= 44");
    STRESS_ASSERT_MSG(grid->get_hits(cx, cy) == 4, "Hits must remain 4");

    // Also verify wall at far_dist became lethal 254
    int wx = 0, wy = 0;
    grid->world_to_cell(far_dist * std::cos(angle), far_dist * std::sin(angle), wx, wy);
    STRESS_ASSERT_MSG(grid->get_cost(wx, wy) == 254, "Distant wall must be Occupied (254)");
    STRESS_ASSERT_MSG(grid->get_hits(wx, wy) == 40, "Distant wall hits must be 40");
  }

  std::cout <<
    "  [PASS] All 8 radial obstacles successfully cleared to Free (0) via pass dilution\n";
  std::cout << "  [PASS] All 8 distant wall endpoints penetrated and marked Occupied (254)\n";

  for (auto * s : scans) {
    delete s;
  }
  destroyTestLaser(laser);
}

// =================================================================================================
// Test Suite 4: Range Capping & Capped Ray Invalidation
// =================================================================================================
void testRangeCapping()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 4: Range Capping (raytrace_range_cap = 16.0m)\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("capping_lidar");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;
  generator.setMinPassThrough(0);
  generator.setOccupancyThreshold(0.10);

  // Cast ray at beam 180 (x axis) with range 22.0m (exceeds cap of 16.0m)
  karto::LocalizedRangeScanVector scans;
  scans.push_back(createSingleRayScan(laser, 0.0, 0.0, 0.0, 180, 22.0));

  nav_msgs::msg::OccupancyGrid map_out;
  generator.generateMap(scans, resolution, map_out);

  const auto * grid = generator.getSparseGrid();
  STRESS_ASSERT(grid != nullptr);

  // 1. Point at 10.0m should be traversed and marked Free (0)
  int mid_x = 0, mid_y = 0;
  grid->world_to_cell(10.0, 0.0, mid_x, mid_y);
  STRESS_ASSERT_MSG(grid->get_cost(mid_x, mid_y) == 0, "Mid-ray cell at 10.0m should be Free (0)");
  STRESS_ASSERT_MSG(grid->get_passes(mid_x, mid_y) == 1, "Mid-ray cell should have 1 pass");

  // 2. Point at 16.0m (cap boundary): should receive pass count, but NOT hit count!
  int cap_x = 0, cap_y = 0;
  grid->world_to_cell(16.0, 0.0, cap_x, cap_y);
  STRESS_ASSERT_MSG(grid->get_hits(cap_x, cap_y) == 0,
    "Capped endpoint at 16.0m must have 0 hits");
  STRESS_ASSERT_MSG(grid->get_passes(cap_x, cap_y) == 1,
    "Capped endpoint at 16.0m must have 1 pass");
  STRESS_ASSERT_MSG(grid->get_cost(cap_x, cap_y) == 0,
    "Capped endpoint must be Free (0), not lethal");

  // 3. Point at 22.0m (uncapped sensor range): should NOT be touched (passes=0, hits=0, cost=255)
  int beyond_x = 0, beyond_y = 0;
  grid->world_to_cell(22.0, 0.0, beyond_x, beyond_y);
  STRESS_ASSERT_MSG(grid->get_passes(beyond_x, beyond_y) == 0,
    "Point beyond cap (22m) must have 0 passes");
  STRESS_ASSERT_MSG(grid->get_hits(beyond_x, beyond_y) == 0,
    "Point beyond cap (22m) must have 0 hits");
  STRESS_ASSERT_MSG(grid->get_cost(beyond_x, beyond_y) == 255,
    "Point beyond cap (22m) must remain unknown (255)");

  std::cout <<
    "  [PASS] Range cap at 16.0m enforced: 10m is Free, 16m capped without hits, 22m unallocated\n";

  for (auto * s : scans) {
    delete s;
  }
  destroyTestLaser(laser);
}

// =================================================================================================
// Test Suite 5: ROS 2 Parameter Configuration via LifecycleNode
// =================================================================================================
void testROS2ParameterWiring(int argc, char ** argv)
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 5: ROS 2 Parameter Wiring via LifecycleNode\n";
  std::cout << "==================================================================\n";

  if (!rclcpp::ok()) {
    rclcpp::init(argc, argv);
  }

  // 1. Test top-level parameters fallback
  {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test_node_top_level");
    node->declare_parameter("min_pass_through", 7);
    node->declare_parameter("occupancy_threshold", 0.35);

    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.configure(node);

    STRESS_ASSERT_MSG(generator.getMinPassThrough() == 7,
      "Top-level parameter min_pass_through should be read as 7");
    STRESS_ASSERT_MSG(std::abs(generator.getOccupancyThreshold() - 0.35) < 1e-5,
      "Top-level parameter occupancy_threshold should be read as 0.35");

    std::cout <<
      "  [PASS] Top-level parameter configuration verified (min_pass: 7, threshold: 0.35)\n";
  }

  // 2. Test scoped overrides (map_generator.*)
  {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test_node_scoped");
    node->declare_parameter("min_pass_through", 3);
    node->declare_parameter("occupancy_threshold", 0.15);
    node->declare_parameter("map_generator.min_pass_through", 12);
    node->declare_parameter("map_generator.occupancy_threshold", 0.42);

    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.configure(node);

    STRESS_ASSERT_MSG(generator.getMinPassThrough() == 12,
      "Scoped parameter map_generator.min_pass_through should override top-level to 12");
    STRESS_ASSERT_MSG(std::abs(generator.getOccupancyThreshold() - 0.42) < 1e-5,
      "Scoped parameter map_generator.occupancy_threshold should override top-level to 0.42");

    std::cout << "  [PASS] Scoped override configuration verified (overridden to 12, 0.42)\n";
  }

  // 3. Test parameter clamping via ROS 2 configure
  {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test_node_clamping");
    node->declare_parameter("map_generator.min_pass_through", -8);
    node->declare_parameter("map_generator.occupancy_threshold", 2.5);

    map_generator_plugins::AdaptiveSparseMapGenerator generator;
    generator.configure(node);

    STRESS_ASSERT_MSG(generator.getMinPassThrough() == 0,
      "Negative min_pass_through in ROS 2 parameter must clamp to 0");
    STRESS_ASSERT_MSG(std::abs(generator.getOccupancyThreshold() - 0.99) < 1e-5,
      "Out-of-bound threshold 2.5 in ROS 2 parameter must clamp to 0.99");

    std::cout << "  [PASS] Parameter clamping via ROS 2 parameter declaration verified\n";
  }

  // 4. Test re-configuration idempotence (zero ParameterAlreadyDeclaredException)
  {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test_node_reconfigure");
    map_generator_plugins::AdaptiveSparseMapGenerator generator;

    try {
      generator.configure(node);
      generator.configure(node);  // Second call must NOT throw!
      STRESS_ASSERT(true);
      std::cout <<
        "  [PASS] Idempotent configure() call verified without ParameterAlreadyDeclaredException\n";
    } catch (const std::exception & ex) {
      STRESS_ASSERT_MSG(false, std::string("configure() threw exception on re-call: ") + ex.what());
    }
  }
}

// =================================================================================================
// Test Suite 6: Multi-Scale Sparse Memory Scaling
// =================================================================================================
void testMultiScaleSparseMemoryScaling()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 6: Multi-Scale Memory Scaling Verification\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("scale_lidar");

  struct ScaleDefinition
  {
    std::string name;
    double size_x;
    double size_y;
    size_t scan_count;
    double max_expected_sparse_mb;
    double min_expected_savings_pct;
  };

  std::vector<ScaleDefinition> scales = {
    {"Small Warehouse (30x20m)", 30.0, 20.0, 40, 5.0, 15.0},
    {"Mega Distribution Hub (120x80m)", 120.0, 80.0, 120, 50.0, 15.0},
    {"Large Logistics Park (250x200m)", 250.0, 200.0, 250, 130.0, 45.0},
    {"1km Megafacility (500x500m)", 500.0, 500.0, 400, 300.0, 70.0}
  };

  const double resolution = 0.05;

  for (const auto & sc : scales) {
    map_generator_plugins::DenseKartoMapGenerator dense_gen;
    map_generator_plugins::AdaptiveSparseMapGenerator sparse_gen;

    karto::LocalizedRangeScanVector scans;
    const double half_x = sc.size_x / 2.0;
    const double half_y = sc.size_y / 2.0;

    for (size_t pose_idx = 0; pose_idx < sc.scan_count; ++pose_idx) {
      const double transit_ratio =
        static_cast<double>(pose_idx) / static_cast<double>(sc.scan_count);

      const double robot_x = -half_x * 0.8 + transit_ratio * (sc.size_x * 1.6);
      const double robot_y = std::sin(transit_ratio * 8.0 * M_PI) * (half_y * 0.7);
      const double robot_yaw = std::atan2(std::cos(transit_ratio * 8.0 * M_PI), 1.0);

      karto::Pose2 robot_pose(robot_x, robot_y, robot_yaw);
      karto::LocalizedRangeScan * localized_scan = new karto::LocalizedRangeScan(
        laser->GetName(), std::vector<kt_double>(360, 10.0));
      localized_scan->SetSensorPose(robot_pose);
      localized_scan->SetOdometricPose(robot_pose);
      localized_scan->SetCorrectedPose(robot_pose);

      std::vector<kt_double> ranges(360);
      karto::PointVectorDouble points;

      for (int beam_idx = 0; beam_idx < 360; ++beam_idx) {
        const double beam_angle = robot_yaw - M_PI + beam_idx * (M_PI / 180.0);
        const double cos_angle = std::cos(beam_angle);
        const double sin_angle = std::sin(beam_angle);

        double hit_distance = 16.0;
        if (cos_angle > 1e-4) {
          hit_distance = std::min(hit_distance, (half_x - robot_x) / cos_angle);
        } else if (cos_angle < -1e-4) {
          hit_distance = std::min(hit_distance, (-half_x - robot_x) / cos_angle);
        }
        if (sin_angle > 1e-4) {
          hit_distance = std::min(hit_distance, (half_y - robot_y) / sin_angle);
        } else if (sin_angle < -1e-4) {
          hit_distance = std::min(hit_distance, (-half_y - robot_y) / sin_angle);
        }

        hit_distance = std::max(0.5, std::min(hit_distance, 20.0));
        ranges[beam_idx] = hit_distance;

        karto::Vector2<kt_double> hit_point(
          robot_x + hit_distance * cos_angle,
          robot_y + hit_distance * sin_angle);
        points.push_back(hit_point);
      }

      localized_scan->SetRangeReadings(ranges);
      localized_scan->SetPointReadings(points);
      scans.push_back(localized_scan);
    }

    nav_msgs::msg::OccupancyGrid dense_map;
    dense_gen.generateMap(scans, resolution, dense_map);
    const size_t dense_bytes = dense_gen.getMemoryBytes();
    const double dense_mb = static_cast<double>(dense_bytes) / (1024.0 * 1024.0);

    nav_msgs::msg::OccupancyGrid sparse_map;
    sparse_gen.generateMap(scans, resolution, sparse_map);
    const size_t sparse_bytes = sparse_gen.getMemoryBytes();
    const double sparse_mb = static_cast<double>(sparse_bytes) / (1024.0 * 1024.0);

    const double savings_pct = (dense_bytes > 0 && dense_bytes > sparse_bytes) ?
      100.0 * (1.0 - static_cast<double>(sparse_bytes) / static_cast<double>(dense_bytes)) :
      0.0;

    STRESS_ASSERT_MSG(sparse_mb <= sc.max_expected_sparse_mb,
      sc.name + " sparse memory (" + std::to_string(sparse_mb) + " MB) exceeded budget (" +
      std::to_string(sc.max_expected_sparse_mb) + " MB)");

    STRESS_ASSERT_MSG(savings_pct >= sc.min_expected_savings_pct,
      sc.name + " savings (" + std::to_string(savings_pct) + "%) below expected minimum (" +
      std::to_string(sc.min_expected_savings_pct) + "%)");

    std::cout << "  [PASS] " << std::left << std::setw(34) << sc.name
              << " Dense: " << std::fixed << std::setprecision(2) << std::setw(9) << dense_mb <<
      " MB | "
              << " Sparse: " << std::setw(8) << sparse_mb << " MB | "
              << " Savings: " << std::setw(6) << savings_pct << "%"
              << "\n";

    // Test generator reset()
    sparse_gen.reset();
    STRESS_ASSERT_MSG(sparse_gen.getSparseGrid()->allocated_blocks_count() == 0,
      "reset() must clear all allocated blocks");
    STRESS_ASSERT_MSG(sparse_gen.getMemoryBytes() < 5000000,
      "reset() must reclaim all tile payload memory (<5MB bucket overhead remaining)");

    for (auto * s : scans) {
      delete s;
    }
  }

  destroyTestLaser(laser);
}

// =================================================================================================
// Test Suite 7: Differential Data Dumper for Oracle Verification
// =================================================================================================
void dumpDifferentialOracleData()
{
  std::cout << "\n==================================================================\n";
  std::cout << "  [TEST] Suite 7: Differential Parity Data Generation\n";
  std::cout << "==================================================================\n";

  karto::LaserRangeFinder * laser = createTestLaser("oracle_lidar");
  const double resolution = 0.05;

  map_generator_plugins::AdaptiveSparseMapGenerator generator;
  generator.setMinPassThrough(2);
  generator.setOccupancyThreshold(0.10);

  // Generate a realistic randomized trajectory with 30 scan poses
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_step(-0.2, 0.2);
  std::uniform_real_distribution<double> dist_range(0.5, 12.0);

  karto::LocalizedRangeScanVector scans;
  double rx = 0.0, ry = 0.0, ryaw = 0.0;

  std::ofstream scan_dump("/tmp/oracle_scans_m2.txt");
  scan_dump << std::fixed << std::setprecision(6);

  for (int p = 0; p < 25; ++p) {
    rx += 0.3 * std::cos(ryaw) + dist_step(rng);
    ry += 0.3 * std::sin(ryaw) + dist_step(rng);
    ryaw += 0.05 + dist_step(rng);

    karto::LocalizedRangeScan * scan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 15.0));
    karto::Pose2 pose(rx, ry, ryaw);
    scan->SetSensorPose(pose);
    scan->SetOdometricPose(pose);
    scan->SetCorrectedPose(pose);

    scan_dump << "POSE " << rx << " " << ry << " " << ryaw << "\n";

    std::vector<kt_double> ranges(360);
    karto::PointVectorDouble points;
    for (int b = 0; b < 360; ++b) {
      const double angle = ryaw - M_PI + b * (M_PI / 180.0);
      const double r = dist_range(rng);
      ranges[b] = r;
      points.push_back(karto::Vector2<kt_double>(rx + r * std::cos(angle),
        ry + r * std::sin(angle)));
      scan_dump << r << (b == 359 ? "" : " ");
    }
    scan_dump << "\n";

    scan->SetRangeReadings(ranges);
    scan->SetPointReadings(points);
    scans.push_back(scan);
  }
  scan_dump.close();

  nav_msgs::msg::OccupancyGrid map_out;
  bool ok = generator.generateMap(scans, resolution, map_out);
  STRESS_ASSERT(ok);

  const auto * grid = generator.getSparseGrid();
  STRESS_ASSERT(grid != nullptr);

  // Export sparse cells: cell_x cell_y passes hits cost
  std::ofstream cell_dump("/tmp/cpp_cells_m2.txt");
  size_t cell_count = 0;

  // Iterate over world bounds
  const int min_cx = grid->min_cell_x();
  const int max_cx = grid->max_cell_x();
  const int min_cy = grid->min_cell_y();
  const int max_cy = grid->max_cell_y();

  for (int cy = min_cy; cy <= max_cy; ++cy) {
    for (int cx = min_cx; cx <= max_cx; ++cx) {
      const uint32_t passes = grid->get_passes(cx, cy);
      if (passes > 0) {
        const uint32_t hits = grid->get_hits(cx, cy);
        const uint8_t cost = grid->get_cost(cx, cy);
        cell_dump << cx << " " << cy << " " << passes << " " << hits << " " <<
          static_cast<int>(cost) << "\n";
        ++cell_count;
      }
    }
  }
  cell_dump.close();

  std::cout << "  [PASS] Dumped " << cell_count <<
    " observed cells to /tmp/cpp_cells_m2.txt for Python oracle verification\n";

  for (auto * s : scans) {
    delete s;
  }
  destroyTestLaser(laser);
}

// =================================================================================================
// Main Runner
// =================================================================================================
int main(int argc, char ** argv)
{
  std::cout << "\n==================================================================\n";
  std::cout << "  EMPIRICAL STRESS TEST HARNESS: MILESTONE 2 (AdaptiveSparseMapGenerator)\n";
  std::cout << "  Parameter Sensitivity, Numerical Boundaries & Memory Scaling\n";
  std::cout << "==================================================================\n";

  const auto start_time = std::chrono::high_resolution_clock::now();

  testMultiScaleSparseMemoryScaling();
  testNumericalBoundaries();
  testExtremeParameters();
  testDynamicClearanceMultiObstacles();
  testRangeCapping();
  testROS2ParameterWiring(argc, argv);
  dumpDifferentialOracleData();

  const auto end_time = std::chrono::high_resolution_clock::now();
  const double total_duration_ms =
    std::chrono::duration<double, std::milli>(end_time - start_time).count();

  std::cout << "\n==================================================================\n";
  std::cout << "  STRESS TEST EXECUTION SUMMARY\n";
  std::cout << "==================================================================\n";
  std::cout << "  Total Assertions Run:    " << g_tests_run << "\n";
  std::cout << "  Assertions Passed:       " << g_tests_passed << "\n";
  std::cout << "  Assertions Failed:       " << g_tests_failed << "\n";
  std::cout << "  Execution Wall Time:     " << std::fixed << std::setprecision(2) <<
    total_duration_ms << " ms\n";
  std::cout << "==================================================================\n\n";

  if (g_tests_failed > 0) {
    std::cerr << "[-] STRESS TEST FAILED with " << g_tests_failed << " errors!\n";
    return 1;
  }

  std::cout << "[+] ALL EMPIRICAL STRESS TESTS PASSED CLEANLY (100% SUCCESS)\n";
  return 0;
}
