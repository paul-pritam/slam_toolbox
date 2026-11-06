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

#include <algorithm>
#include <cmath>
#include <memory>

#include "slam_toolbox/adaptive_sparse_map_generator.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace map_generator_plugins
{

AdaptiveSparseMapGenerator::AdaptiveSparseMapGenerator()
: raytrace_range_cap_(5.0),
  occlusion_stop_(true),
  min_reinforcement_hits_(2),
  decay_expiry_sec_(8.0),
  resolution_(0.05)
{
}

void AdaptiveSparseMapGenerator::configure(rclcpp_lifecycle::LifecycleNode::SharedPtr node)
{
  node_ = node;
  if (!node_) {
    return;
  }

  if (!node_->has_parameter("map_generator.raytrace_range_cap")) {
    node_->declare_parameter("map_generator.raytrace_range_cap", raytrace_range_cap_);
  }
  node_->get_parameter("map_generator.raytrace_range_cap", raytrace_range_cap_);

  if (!node_->has_parameter("map_generator.occlusion_stop")) {
    node_->declare_parameter("map_generator.occlusion_stop", occlusion_stop_);
  }
  node_->get_parameter("map_generator.occlusion_stop", occlusion_stop_);

  if (!node_->has_parameter("map_generator.min_reinforcement_hits")) {
    node_->declare_parameter(
      "map_generator.min_reinforcement_hits",
      static_cast<int>(min_reinforcement_hits_));
  }
  int hits = min_reinforcement_hits_;
  node_->get_parameter("map_generator.min_reinforcement_hits", hits);
  min_reinforcement_hits_ = static_cast<uint32_t>(std::max(1, hits));

  if (!node_->has_parameter("map_generator.decay_expiry_sec")) {
    node_->declare_parameter("map_generator.decay_expiry_sec", decay_expiry_sec_);
  }
  node_->get_parameter("map_generator.decay_expiry_sec", decay_expiry_sec_);

  RCLCPP_INFO(
    node_->get_logger(),
    "Configured AdaptiveSparseMapGenerator (cap: %.1fm, occlusion_stop: %s, hits: %u)",
    raytrace_range_cap_, occlusion_stop_ ? "true" : "false", min_reinforcement_hits_);
}

bool AdaptiveSparseMapGenerator::generateMap(
  const karto::LocalizedRangeScanVector & scans,
  double resolution,
  nav_msgs::msg::OccupancyGrid & out_map)
{
  if (scans.empty()) {
    return false;
  }

  resolution_ = resolution;

  // Initialize or re-create sparse grid if resolution changed
  if (!sparse_grid_ || std::abs(sparse_grid_->resolution() - resolution) > 1e-6) {
    sparse_grid_ = std::make_unique<slam_toolbox::adaptive::SparseGrid2D<16>>(
      resolution, 0.0, 0.0, 255);
  } else {
    sparse_grid_->clear();
  }

  for (karto::LocalizedRangeScan * localized_scan : scans) {
    if (!localized_scan) {
      continue;
    }

    karto::LaserRangeFinder * laser = localized_scan->GetLaserRangeFinder();
    if (!laser) {
      continue;
    }

    const double range_threshold = laser->GetRangeThreshold();
    const double max_sensor_range = laser->GetMaximumRange();
    const double min_sensor_range = laser->GetMinimumRange();
    const karto::Vector2<kt_double> sensor_position =
      localized_scan->GetSensorPose().GetPosition();
    const karto::PointVectorDouble & scan_points = localized_scan->GetPointReadings(false);
    const kt_double * scan_ranges = localized_scan->GetRangeReadings();
    const kt_int32u num_readings = localized_scan->GetNumberOfRangeReadings();

    const size_t num_points = std::min(scan_points.size(), static_cast<size_t>(num_readings));
    for (size_t i = 0; i < num_points; ++i) {
      const double range_meters = scan_ranges[i];
      if (range_meters <= min_sensor_range || range_meters >= max_sensor_range ||
        std::isnan(range_meters))
      {
        continue;
      }

      karto::Vector2<kt_double> ray_endpoint = scan_points[i];
      bool is_endpoint_valid = (range_meters < (range_threshold - karto::KT_TOLERANCE));

      // Limit ray clearance range to prevent void space bleeding across large open areas
      if (range_meters > raytrace_range_cap_) {
        const double scale_ratio = raytrace_range_cap_ / range_meters;
        const double delta_x = ray_endpoint.GetX() - sensor_position.GetX();
        const double delta_y = ray_endpoint.GetY() - sensor_position.GetY();
        ray_endpoint.SetX(sensor_position.GetX() + scale_ratio * delta_x);
        ray_endpoint.SetY(sensor_position.GetY() + scale_ratio * delta_y);
        is_endpoint_valid = false;
      }

      int sensor_cell_x = 0;
      int sensor_cell_y = 0;
      int endpoint_cell_x = 0;
      int endpoint_cell_y = 0;
      sparse_grid_->world_to_cell(
        sensor_position.GetX(), sensor_position.GetY(),
        sensor_cell_x, sensor_cell_y);
      sparse_grid_->world_to_cell(
        ray_endpoint.GetX(), ray_endpoint.GetY(),
        endpoint_cell_x, endpoint_cell_y);

      // Raytrace free cells along the line with wall occlusion termination:
      // If the ray encounters an already-confirmed lethal obstacle cell (cost 254),
      // we terminate ray traversal to prevent punch-through and wall erosion.
      sparse_grid_->raytrace(
        sensor_cell_x, sensor_cell_y,
        endpoint_cell_x, endpoint_cell_y,
        [this, endpoint_cell_x, endpoint_cell_y](
          int traversed_cell_x, int traversed_cell_y) -> bool
        {
          if (traversed_cell_x == endpoint_cell_x && traversed_cell_y == endpoint_cell_y) {
            return false;
          }
          if (this->occlusion_stop_ &&
          this->sparse_grid_->get_cost(traversed_cell_x, traversed_cell_y) == 254)
          {
            return false;  // Occlusion: Stop ray at confirmed wall
          }
          this->sparse_grid_->set_cost(traversed_cell_x, traversed_cell_y, 0, 0.0);
          return true;
        });

      if (is_endpoint_valid) {
        sparse_grid_->set_cost(endpoint_cell_x, endpoint_cell_y, 254, 0.0);
      }
    }
  }

  // Populate output OccupancyGrid message
  sparse_grid_->to_occupancy_grid(out_map, "map", builtin_interfaces::msg::Time());
  return true;
}

void AdaptiveSparseMapGenerator::reset()
{
  if (sparse_grid_) {
    sparse_grid_->clear();
  }
}

size_t AdaptiveSparseMapGenerator::getMemoryBytes() const
{
  return sparse_grid_ ? sparse_grid_->memory_bytes_used() : 0;
}

}  // namespace map_generator_plugins

PLUGINLIB_EXPORT_CLASS(
  map_generator_plugins::AdaptiveSparseMapGenerator,
  slam_toolbox::MapGenerator)
