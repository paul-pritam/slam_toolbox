/*
 * slam_toolbox
 * Copyright Work Modifications (c) 2018, Simbe Robotics, Inc.
 * Copyright Work Modifications (c) 2019, Steve Macenski
 * Copyright Work Modifications (c) 2026, Pritam Paul
 *
 * THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
 * COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
 * COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
 * AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 *
 * BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
 * BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
 * CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
 * CONDITIONS.
 *
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "karto_sdk/Karto.h"
#include "slam_toolbox/dense_karto_map_generator.hpp"
#include "slam_toolbox/adaptive_sparse_map_generator.hpp"

/**
 * @brief Instantiate and register a synthetic 360-degree LiDAR scanner for benchmarking.
 */
karto::LaserRangeFinder * createBenchmarkLaser()
{
  karto::LaserRangeFinder * laser = karto::LaserRangeFinder::CreateLaserRangeFinder(
    karto::LaserRangeFinder_Custom, karto::Name("benchmark_lidar"));
  laser->SetMinimumAngle(-M_PI);
  laser->SetMaximumAngle(M_PI);
  laser->SetAngularResolution(M_PI / 180.0);  // 1 degree per ray (360 beams)
  laser->SetMinimumRange(0.1);
  laser->SetMaximumRange(25.0);
  laser->SetRangeThreshold(20.0);
  karto::SensorManager::GetInstance()->RegisterSensor(laser, true);
  return laser;
}

/**
 * @brief Generate a synthetic serpentine robot path and ray readings across a facility.
 * @param laser Laser sensor handle.
 * @param facility_length_x Facility dimension in X (meters).
 * @param facility_width_y Facility dimension in Y (meters).
 * @param num_poses Number of trajectory waypoints to simulate.
 */
karto::LocalizedRangeScanVector generateFacilityScans(
  karto::LaserRangeFinder * laser,
  double facility_length_x,
  double facility_width_y,
  size_t num_poses)
{
  karto::LocalizedRangeScanVector scans;
  const double half_x = facility_length_x / 2.0;
  const double half_y = facility_width_y / 2.0;

  for (size_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
    const double transit_ratio =
      static_cast<double>(pose_idx) / static_cast<double>(num_poses);

    // Robot traverses aisles along X while oscillating in Y
    const double robot_x = -half_x * 0.8 + transit_ratio * (facility_length_x * 1.6);
    const double robot_y = std::sin(transit_ratio * 8.0 * M_PI) * (half_y * 0.7);
    const double robot_yaw = std::atan2(std::cos(transit_ratio * 8.0 * M_PI), 1.0);

    karto::Pose2 robot_pose(robot_x, robot_y, robot_yaw);
    karto::LocalizedRangeScan * localized_scan = new karto::LocalizedRangeScan(
      laser->GetName(), std::vector<kt_double>(360, 10.0));
    localized_scan->SetSensorPose(robot_pose);
    localized_scan->SetOdometricPose(robot_pose);
    localized_scan->SetCorrectedPose(robot_pose);

    // Simulate 360 rays intersecting perimeter walls
    std::vector<kt_double> ranges(360);
    karto::PointVectorDouble points;

    for (int beam_idx = 0; beam_idx < 360; ++beam_idx) {
      const double beam_angle = robot_yaw - M_PI + beam_idx * (M_PI / 180.0);
      const double cos_angle = std::cos(beam_angle);
      const double sin_angle = std::sin(beam_angle);

      // Raycast against rectangular facility boundaries
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

  return scans;
}

/**
 * @brief Structure defining facility benchmarking dimensions.
 */
struct BenchmarkConfig
{
  std::string name;
  double length_x;
  double width_y;
  size_t scans_count;
};

void runBenchmarkSuite()
{
  const std::string banner_line(100, '=');
  const std::string table_line(100, '-');

  std::cout << "\n" << banner_line << "\n";
  std::cout << "          SLAM TOOLBOX MODULAR MAP GENERATOR BENCHMARK REPORT\n";
  std::cout << "          Comparing DenseKartoMapGenerator vs AdaptiveSparseMapGenerator\n";
  std::cout << banner_line << "\n\n";

  std::cout << std::left
            << std::setw(24) << "Facility Scale"
            << std::setw(16) << "Dimensions (m)"
            << std::setw(16) << "Dense RAM"
            << std::setw(16) << "Sparse RAM"
            << std::setw(14) << "Savings (%)"
            << std::setw(14) << "Latency"
            << "\n";
  std::cout << table_line << "\n";

  std::vector<BenchmarkConfig> configs = {
    {"Small Warehouse", 30.0, 20.0, 40},
    {"Mega Distribution Hub", 120.0, 80.0, 120},
    {"Large Logistics Park", 250.0, 200.0, 250},
    {"1km Megafacility", 500.0, 500.0, 400}
  };

  karto::LaserRangeFinder * laser = createBenchmarkLaser();
  map_generator_plugins::DenseKartoMapGenerator dense_generator;
  map_generator_plugins::AdaptiveSparseMapGenerator sparse_generator;

  for (const auto & config : configs) {
    try {
      karto::LocalizedRangeScanVector scans = generateFacilityScans(
        laser, config.length_x, config.width_y, config.scans_count);
      const double map_resolution = 0.05;

      // Evaluate Dense Karto Generator
      nav_msgs::msg::OccupancyGrid dense_grid_msg;
      dense_generator.generateMap(scans, map_resolution, dense_grid_msg);
      const size_t dense_memory_bytes = dense_generator.getMemoryBytes();
      const double dense_mb =
        static_cast<double>(dense_memory_bytes) / (1024.0 * 1024.0);

      // Evaluate Adaptive Sparse Generator
      nav_msgs::msg::OccupancyGrid sparse_grid_msg;
      const auto start_time = std::chrono::high_resolution_clock::now();
      sparse_generator.generateMap(scans, map_resolution, sparse_grid_msg);
      const auto end_time = std::chrono::high_resolution_clock::now();
      const double sparse_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

      const size_t sparse_memory_bytes = sparse_generator.getMemoryBytes();
      const double sparse_mb =
        static_cast<double>(sparse_memory_bytes) / (1024.0 * 1024.0);

      const double savings_pct =
        (dense_memory_bytes > 0 && dense_memory_bytes > sparse_memory_bytes) ?
        100.0 * (1.0 - static_cast<double>(sparse_memory_bytes) /
        static_cast<double>(dense_memory_bytes)) : 0.0;

      std::ostringstream dim_ss;
      dim_ss << static_cast<int>(config.length_x) << "x" << static_cast<int>(config.width_y);

      std::ostringstream dense_ss;
      dense_ss << std::fixed << std::setprecision(2) << dense_mb << " MB";

      std::ostringstream sparse_ss;
      sparse_ss << std::fixed << std::setprecision(2) << sparse_mb << " MB";

      std::ostringstream time_ss;
      time_ss << std::fixed << std::setprecision(1) << sparse_time_ms << " ms";

      std::cout << std::left
                << std::setw(24) << config.name
                << std::setw(16) << dim_ss.str()
                << std::setw(16) << dense_ss.str()
                << std::setw(16) << sparse_ss.str()
                << std::setw(14) << (std::to_string(savings_pct).substr(0, 4) + "%")
                << std::setw(14) << time_ss.str()
                << "\n";

      // Deallocate synthetic scan objects
      for (auto * scan : scans) {
        delete scan;
      }
    } catch (const karto::Exception & ex) {
      std::cerr << "Karto Exception in " << config.name << ": " << ex.GetErrorMessage() << "\n";
    } catch (const std::exception & ex) {
      std::cerr << "Std Exception in " << config.name << ": " << ex.what() << "\n";
    }
  }

  delete laser;
  std::cout << banner_line << "\n\n";
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;
  runBenchmarkSuite();
  return 0;
}
