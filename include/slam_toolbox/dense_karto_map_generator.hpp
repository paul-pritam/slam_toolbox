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

#ifndef SLAM_TOOLBOX__DENSE_KARTO_MAP_GENERATOR_HPP_
#define SLAM_TOOLBOX__DENSE_KARTO_MAP_GENERATOR_HPP_

#include <memory>

#include "slam_toolbox/toolbox_types.hpp"
#include "slam_toolbox/map_generator.hpp"
#include "slam_toolbox/visualization_utils.hpp"

namespace map_generator_plugins
{

/**
 * @class DenseKartoMapGenerator
 * @brief Default map generator plugin wrapping Karto SDK's monolithic 9-byte/cell grid.
 *
 * Preserves 100% exact legacy behavior of SLAM Toolbox. Operates by instantiating
 * a single dense karto::OccupancyGrid covering the entire bounding box of all poses,
 * executing Karto's ray clearing and hit accumulation, and converting to nav_msgs::msg::OccupancyGrid.
 */
class DenseKartoMapGenerator : public slam_toolbox::MapGenerator
{
public:
  DenseKartoMapGenerator();
  virtual ~DenseKartoMapGenerator() = default;

  /**
   * @brief Configure the dense generator parameters from ROS 2 node parameters.
   * @param node LifecycleNode shared pointer.
   */
  void configure(rclcpp_lifecycle::LifecycleNode::SharedPtr node) override;

  /**
   * @brief Rasterize localized scans into a standard dense ROS 2 OccupancyGrid.
   * @param scans Localized scans from mapper.
   * @param resolution Resolution in meters per cell.
   * @param out_map Reference to output OccupancyGrid message.
   * @return true on success, false if scans are empty or grid creation fails.
   */
  bool generateMap(
    const karto::LocalizedRangeScanVector & scans,
    double resolution,
    nav_msgs::msg::OccupancyGrid & out_map) override;

  /**
   * @brief Reset internal memory tracking counters.
   */
  void reset() override;

  /**
   * @brief Get heap memory allocated during the most recent map generation in bytes.
   * @return Size in bytes (width * height * 9 bytes per cell).
   */
  size_t getMemoryBytes() const override;

private:
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_{nullptr};
  int min_pass_through_{2};
  double occupancy_threshold_{0.1};
  size_t last_memory_bytes_{0};
};

}  // namespace map_generator_plugins

#endif  // SLAM_TOOLBOX__DENSE_KARTO_MAP_GENERATOR_HPP_
