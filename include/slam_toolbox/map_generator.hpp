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

#ifndef SLAM_TOOLBOX__MAP_GENERATOR_HPP_
#define SLAM_TOOLBOX__MAP_GENERATOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "karto_sdk/Karto.h"

namespace slam_toolbox
{

/**
 * @class MapGenerator
 * @brief Abstract pluginlib interface for modular occupancy grid generation in SLAM Toolbox.
 *
 * Decouples pose graph optimization (Ceres solver, correlative scan matching) from
 * 2D occupancy grid rasterization.
 *
 * In legacy SLAM Toolbox, map updates required allocating a monolithic dense grid
 * (karto::OccupancyGrid, 9 bytes per cell) over the entire bounding box of all poses.
 * This interface allows runtime selection of map generation strategies via ROS 2 parameters:
 *  - "map_generator_plugins::DenseKartoMapGenerator": Default 100% backward-compatible plugin.
 *  - "map_generator_plugins::AdaptiveSparseMapGenerator": High-performance block-sparse engine.
 */
class MapGenerator
{
public:
  using Ptr = std::shared_ptr<MapGenerator>;

  MapGenerator() = default;
  virtual ~MapGenerator() = default;

  /**
   * @brief Configure the generator with node parameters and logging handles.
   * @param node Shared pointer to parent LifecycleNode for declaring/reading parameters.
   */
  virtual void configure(rclcpp_lifecycle::LifecycleNode::SharedPtr node) = 0;

  /**
   * @brief Generate a ROS OccupancyGrid from drift-corrected localized range scans.
   * @param scans Vector of localized range scans from the Ceres-optimized pose graph.
   * @param resolution Map resolution in meters per cell (e.g. 0.05 m/cell).
   * @param out_map Reference to ROS 2 nav_msgs::msg::OccupancyGrid to populate.
   * @return true if map generation succeeded, false otherwise.
   */
  virtual bool generateMap(
    const karto::LocalizedRangeScanVector & scans,
    double resolution,
    nav_msgs::msg::OccupancyGrid & out_map) = 0;

  /**
   * @brief Reset or clear internal map tiles, spatial caches, and hit counters.
   */
  virtual void reset() = 0;

  /**
   * @brief Get active heap memory currently consumed by the internal map structure.
   * @return Memory footprint in bytes.
   */
  virtual size_t getMemoryBytes() const = 0;
};

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX__MAP_GENERATOR_HPP_
