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

#include "slam_toolbox/dense_karto_map_generator.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace map_generator_plugins
{

DenseKartoMapGenerator::DenseKartoMapGenerator()
: min_pass_through_(2),
  occupancy_threshold_(0.1),
  last_memory_bytes_(0)
{
}

void DenseKartoMapGenerator::configure(rclcpp_lifecycle::LifecycleNode::SharedPtr node)
{
  node_ = node;
  if (!node_) {
    return;
  }

  if (!node_->has_parameter("map_generator.min_pass_through")) {
    node_->declare_parameter("map_generator.min_pass_through", min_pass_through_);
  }
  node_->get_parameter("map_generator.min_pass_through", min_pass_through_);

  if (!node_->has_parameter("map_generator.occupancy_threshold")) {
    node_->declare_parameter("map_generator.occupancy_threshold", occupancy_threshold_);
  }
  node_->get_parameter("map_generator.occupancy_threshold", occupancy_threshold_);

  RCLCPP_INFO(
    node_->get_logger(),
    "Configured DenseKartoMapGenerator (min_pass_through: %d, occupancy_threshold: %.2f)",
    min_pass_through_, occupancy_threshold_);
}

bool DenseKartoMapGenerator::generateMap(
  const karto::LocalizedRangeScanVector & scans,
  double resolution,
  nav_msgs::msg::OccupancyGrid & out_map)
{
  if (scans.empty()) {
    return false;
  }

  // Create Karto SDK dense occupancy grid (allocates 9 bytes per cell across the bounding box)
  karto::OccupancyGrid * occ_grid = karto::OccupancyGrid::CreateFromScans(
    scans,
    resolution,
    static_cast<kt_int32u>(min_pass_through_),
    static_cast<kt_double>(occupancy_threshold_));

  if (!occ_grid) {
    return false;
  }

  // Calculate allocated footprint (1 byte cell state + 4 bytes pass count + 4 bytes hit count)
  const size_t grid_width = static_cast<size_t>(occ_grid->GetWidth());
  const size_t grid_height = static_cast<size_t>(occ_grid->GetHeight());
  last_memory_bytes_ = grid_width * grid_height * 9;

  // Convert Karto grid cells to ROS OccupancyGrid message format
  vis_utils::toNavMap(occ_grid, out_map);

  // Deallocate temporary dense grid
  delete occ_grid;
  occ_grid = nullptr;

  return true;
}

void DenseKartoMapGenerator::reset()
{
  last_memory_bytes_ = 0;
}

size_t DenseKartoMapGenerator::getMemoryBytes() const
{
  return last_memory_bytes_;
}

}  // namespace map_generator_plugins

PLUGINLIB_EXPORT_CLASS(
  map_generator_plugins::DenseKartoMapGenerator,
  slam_toolbox::MapGenerator)
