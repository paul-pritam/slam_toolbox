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

#ifndef SLAM_TOOLBOX__ADAPTIVE__SPARSE_GRID_2D_HPP_
#define SLAM_TOOLBOX__ADAPTIVE__SPARSE_GRID_2D_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "slam_toolbox/adaptive/sparse_block.hpp"

namespace slam_toolbox
{
namespace adaptive
{

/**
 * @struct BlockCoordHash
 * @brief High-speed 64-bit MurmurHash3 finalizer for packed 2D block coordinates (bx, by).
 *
 * Provides near-zero collision rates and single-cycle avalanche distribution across hash buckets.
 */
struct BlockCoordHash
{
  [[nodiscard]] std::size_t operator()(uint64_t key) const noexcept
  {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key);
  }
};

/**
 * @class SparseGrid2D
 * @brief High-performance block-sparse 2D spatial grid for AMR navigation and SLAM.
 *
 * Implements a two-level hierarchical memory architecture:
 *  - Level 1: Hashed 2D spatial directory of block pointers indexed by 64-bit coordinates.
 *  - Level 2: Flat contiguous SparseBlock tiles (16x16 cells = 256 bytes, 64-byte aligned).
 *
 * Empty warehouse regions and unexplored open spaces require 0 bytes of heap memory.
 * Features coarse empty-block leaping raycasting and submap extraction.
 */
template<int TileSize = 16>
class SparseGrid2D
{
public:
  using BlockType = SparseBlock<TileSize>;
  static constexpr int TILE_SIZE = TileSize;
  static constexpr int TILE_SHIFT = BlockType::SHIFT;
  static constexpr int TILE_MASK = BlockType::MASK;

  /**
   * @brief Construct a sparse grid.
   * @param resolution Grid resolution in meters per cell (default: 0.05 m).
   * @param origin_x Global world X coordinate of grid origin.
   * @param origin_y Global world Y coordinate of grid origin.
   * @param default_value Sentinel value returned for unallocated regions (0 for free, 255 unknown).
   */
  explicit SparseGrid2D(
    double resolution = 0.05,
    double origin_x = 0.0,
    double origin_y = 0.0,
    uint8_t default_value = 0,
    uint32_t min_pass_through = 2,
    double occupancy_threshold = 0.1)
  : resolution_(resolution),
    origin_x_(origin_x),
    origin_y_(origin_y),
    default_value_(default_value),
    min_pass_through_(min_pass_through),
    occupancy_threshold_(occupancy_threshold),
    min_cell_x_(std::numeric_limits<int>::max()),
    min_cell_y_(std::numeric_limits<int>::max()),
    max_cell_x_(std::numeric_limits<int>::lowest()),
    max_cell_y_(std::numeric_limits<int>::lowest())
  {
  }

  /**
   * @brief Pack two 32-bit signed block coordinates into a unique 64-bit hash table key.
   * @param block_x Spatial block coordinate in X direction.
   * @param block_y Spatial block coordinate in Y direction.
   * @return 64-bit packed key.
   */
  [[nodiscard]] static constexpr uint64_t pack_key(int32_t block_x, int32_t block_y) noexcept
  {
    return (static_cast<uint64_t>(static_cast<uint32_t>(block_x)) << 32) |
           static_cast<uint32_t>(block_y);
  }

  /**
   * @brief Unpack a 64-bit key back into two 32-bit signed block coordinates.
   * @param key Packed 64-bit key.
   * @param block_x Output block X coordinate.
   * @param block_y Output block Y coordinate.
   */
  static void unpack_key(uint64_t key, int32_t & block_x, int32_t & block_y) noexcept
  {
    block_x = static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
    block_y = static_cast<int32_t>(static_cast<uint32_t>(key & 0xFFFFFFFFULL));
  }

  /**
   * @brief Convert world coordinates (meters) to global discrete cell coordinates.
   */
  inline void world_to_cell(
    double world_x, double world_y,
    int & cell_x, int & cell_y) const noexcept
  {
    cell_x = static_cast<int>(std::floor((world_x - origin_x_) / resolution_));
    cell_y = static_cast<int>(std::floor((world_y - origin_y_) / resolution_));
  }

  /**
   * @brief Convert global cell coordinates to world center coordinates (meters).
   */
  inline void cell_to_world(
    int cell_x, int cell_y,
    double & world_x, double & world_y) const noexcept
  {
    world_x = origin_x_ + (static_cast<double>(cell_x) + 0.5) * resolution_;
    world_y = origin_y_ + (static_cast<double>(cell_y) + 0.5) * resolution_;
  }

  /**
   * @brief Decompose global cell coordinates into block index and intra-tile local offsets.
   */
  static inline void cell_to_block(
    int cell_x, int cell_y,
    int & block_x, int & block_y,
    int & local_x, int & local_y) noexcept
  {
    block_x = cell_x >> TILE_SHIFT;
    block_y = cell_y >> TILE_SHIFT;
    local_x = cell_x & TILE_MASK;
    local_y = cell_y & TILE_MASK;
  }

  /**
   * @brief Query cost value at discrete cell coordinates. Returns default_value if unallocated.
   */
  [[nodiscard]] inline uint8_t get_cost(int cell_x, int cell_y) const noexcept
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);

    const auto it = blocks_.find(pack_key(block_x, block_y));
    if (it == blocks_.end()) {
      return default_value_;
    }
    return it->second->get(local_x, local_y);
  }

  /**
   * @brief Query cost value at continuous world coordinates (meters).
   */
  [[nodiscard]] inline uint8_t get_cost_world(double world_x, double world_y) const noexcept
  {
    int cell_x, cell_y;
    world_to_cell(world_x, world_y, cell_x, cell_y);
    return get_cost(cell_x, cell_y);
  }

  /**
   * @brief Set cell cost at discrete cell coordinates with opportunistic lazy block allocation.
   */
  inline void set_cost(int cell_x, int cell_y, uint8_t cost, double timestamp = 0.0)
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);
    const uint64_t key = pack_key(block_x, block_y);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      if (cost == default_value_) {
        return;  // Zero allocation for uniform default regions
      }
      auto new_block = std::make_unique<BlockType>(default_value_);
      new_block->set(local_x, local_y, cost);
      new_block->touch(timestamp, cost == 254);
      blocks_.emplace(key, std::move(new_block));
    } else {
      it->second->set(local_x, local_y, cost);
      it->second->touch(timestamp, cost == 254);
    }

    update_bounds(cell_x, cell_y);
  }

  /**
   * @brief Set cell cost at world coordinates (meters).
   */
  inline void set_cost_world(
    double world_x, double world_y,
    uint8_t cost, double timestamp = 0.0)
  {
    int cell_x, cell_y;
    world_to_cell(world_x, world_y, cell_x, cell_y);
    set_cost(cell_x, cell_y, cost, timestamp);
  }

  // =========================================================================
  // Probabilistic Occupancy Configuration & Methods (Karto SDK logic parity)
  // =========================================================================

  /**
   * @brief Set the minimum pass count required before classifying a cell.
   * @param val Minimum pass count (default: 2).
   */
  void set_min_pass_through(uint32_t val) noexcept
  {
    min_pass_through_ = val;
  }

  /**
   * @brief Get the minimum pass count threshold.
   */
  [[nodiscard]] uint32_t get_min_pass_through() const noexcept
  {
    return min_pass_through_;
  }

  /**
   * @brief Set the occupancy threshold ratio (hits / passes).
   * @param val Threshold ratio in range (0.0, 1.0) (default: 0.1).
   */
  void set_occupancy_threshold(double val) noexcept
  {
    occupancy_threshold_ = val;
  }

  /**
   * @brief Get the occupancy threshold ratio.
   */
  [[nodiscard]] double get_occupancy_threshold() const noexcept
  {
    return occupancy_threshold_;
  }

  /**
   * @brief Get total ray pass count for discrete cell coordinates.
   */
  [[nodiscard]] inline uint32_t get_passes(int cell_x, int cell_y) const noexcept
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);
    const auto it = blocks_.find(pack_key(block_x, block_y));
    if (it == blocks_.end()) {
      return 0;
    }
    return it->second->get_passes(local_x, local_y);
  }

  /**
   * @brief Get total obstacle hit count for discrete cell coordinates.
   */
  [[nodiscard]] inline uint32_t get_hits(int cell_x, int cell_y) const noexcept
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);
    const auto it = blocks_.find(pack_key(block_x, block_y));
    if (it == blocks_.end()) {
      return 0;
    }
    return it->second->get_hits(local_x, local_y);
  }

  /**
   * @brief Record a laser ray passing through discrete cell coordinates.
   */
  inline void record_pass(int cell_x, int cell_y)
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);
    const uint64_t key = pack_key(block_x, block_y);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      auto new_block = std::make_unique<BlockType>(default_value_);
      new_block->record_pass(local_x, local_y, min_pass_through_, occupancy_threshold_);
      blocks_.emplace(key, std::move(new_block));
    } else {
      it->second->record_pass(local_x, local_y, min_pass_through_, occupancy_threshold_);
    }

    update_bounds(cell_x, cell_y);
  }

  /**
   * @brief Record a laser obstacle hit at discrete cell coordinates.
   */
  inline void record_hit(int cell_x, int cell_y)
  {
    int block_x, block_y, local_x, local_y;
    cell_to_block(cell_x, cell_y, block_x, block_y, local_x, local_y);
    const uint64_t key = pack_key(block_x, block_y);

    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      auto new_block = std::make_unique<BlockType>(default_value_);
      new_block->record_hit(local_x, local_y, min_pass_through_, occupancy_threshold_);
      blocks_.emplace(key, std::move(new_block));
    } else {
      it->second->record_hit(local_x, local_y, min_pass_through_, occupancy_threshold_);
    }

    update_bounds(cell_x, cell_y);
  }

  /**
   * @brief Update occupancy states across all active blocks and cells.
   */
  void update_all_occupancy()
  {
    for (auto & [key, block] : blocks_) {
      block->update_all_occupancy(min_pass_through_, occupancy_threshold_);
    }
  }

  /**
   * @brief Check whether the block containing (block_x, block_y) is physically allocated.
   */
  [[nodiscard]] bool is_block_allocated(int block_x, int block_y) const noexcept
  {
    return blocks_.find(pack_key(block_x, block_y)) != blocks_.end();
  }

  /**
   * @brief Check whether a block is uniform and retrieve its uniform value.
   */
  [[nodiscard]] bool is_block_uniform(
    int block_x, int block_y,
    uint8_t & value) const noexcept
  {
    const auto it = blocks_.find(pack_key(block_x, block_y));
    if (it == blocks_.end()) {
      value = default_value_;
      return true;
    }
    if (it->second->is_uniform()) {
      value = it->second->default_value;
      return true;
    }
    return false;
  }

  /**
   * @brief Check whether a block consists entirely of free space (cost 0).
   */
  [[nodiscard]] bool is_block_free(int block_x, int block_y) const noexcept
  {
    uint8_t value = 255;
    if (is_block_uniform(block_x, block_y, value)) {
      return value == 0;
    }
    return false;
  }

  /**
   * @brief Clear all allocated blocks and reset spatial bounds.
   */
  void clear() noexcept
  {
    blocks_.clear();
    min_cell_x_ = std::numeric_limits<int>::max();
    min_cell_y_ = std::numeric_limits<int>::max();
    max_cell_x_ = std::numeric_limits<int>::lowest();
    max_cell_y_ = std::numeric_limits<int>::lowest();
  }

  /**
   * @brief Deallocate uniform blocks whose contents match their default values.
   */
  void prune()
  {
    for (auto it = blocks_.begin(); it != blocks_.end(); ) {
      if (it->second->is_uniform()) {
        it = blocks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  /**
   * @brief Garbage collection: deallocate blocks that can be pruned and return count reclaimed.
   */
  size_t prune_unmodified_blocks()
  {
    size_t pruned = 0;
    for (auto it = blocks_.begin(); it != blocks_.end(); ) {
      if (it->second->can_prune()) {
        it = blocks_.erase(it);
        ++pruned;
      } else {
        ++it;
      }
    }
    return pruned;
  }

  /**
   * @brief Fast hierarchical 2D DDA raycaster with coarse empty block leaping.
   *
   * Traverses cells from (start_x, start_y) to (end_x, end_y). If an traversed block is uniform
   * and does not contain the target endpoint, the inner loop advances quickly across the block
   * boundaries without redundant per-cell hash lookups.
   *
   * @tparam RayAction Callback invocable as bool(int cell_x, int cell_y).
   *                   Returning false terminates ray traversal immediately.
   */
  template<typename RayAction>
  void raytrace(
    int start_x, int start_y,
    int end_x, int end_y,
    RayAction && action)
  {
    int delta_x = std::abs(end_x - start_x);
    int delta_y = std::abs(end_y - start_y);
    int step_x = (start_x < end_x) ? 1 : -1;
    int step_y = (start_y < end_y) ? 1 : -1;
    int error = delta_x - delta_y;

    int current_x = start_x;
    int current_y = start_y;

    while (true) {
      if (!action(current_x, current_y)) {
        break;
      }

      if (current_x == end_x && current_y == end_y) {
        break;
      }

      int block_x = current_x >> TILE_SHIFT;
      int block_y = current_y >> TILE_SHIFT;

      int target_block_x = end_x >> TILE_SHIFT;
      int target_block_y = end_y >> TILE_SHIFT;

      uint8_t uniform_val = 0;
      if ((block_x != target_block_x || block_y != target_block_y) &&
        is_block_uniform(block_x, block_y, uniform_val))
      {
        int next_boundary_x = (step_x > 0) ? ((block_x + 1) << TILE_SHIFT) - 1 :
          (block_x << TILE_SHIFT);
        int next_boundary_y = (step_y > 0) ? ((block_y + 1) << TILE_SHIFT) - 1 :
          (block_y << TILE_SHIFT);

        while ((current_x != next_boundary_x || current_y != next_boundary_y) &&
          (current_x != end_x || current_y != end_y))
        {
          int error2 = 2 * error;
          if (error2 > -delta_y) {
            error -= delta_y;
            current_x += step_x;
          }
          if (error2 < delta_x) {
            error += delta_x;
            current_y += step_y;
          }
          if (!action(current_x, current_y)) {
            return;
          }
          if ((current_x >> TILE_SHIFT) != block_x || (current_y >> TILE_SHIFT) != block_y) {
            break;
          }
        }
        continue;
      }

      int error2 = 2 * error;
      if (error2 > -delta_y) {
        error -= delta_y;
        current_x += step_x;
      }
      if (error2 < delta_x) {
        error += delta_x;
        current_y += step_y;
      }
    }
  }

  /**
   * @brief Get total number of actively allocated 16x16 blocks.
   */
  [[nodiscard]] size_t allocated_blocks_count() const noexcept
  {
    return blocks_.size();
  }

  /**
   * @brief Calculate total active heap and structural memory consumed in bytes.
   */
  [[nodiscard]] size_t memory_bytes_used() const noexcept
  {
    const size_t block_bytes = blocks_.size() * sizeof(BlockType);
    const size_t hash_overhead = blocks_.bucket_count() * sizeof(void *) +
      blocks_.size() * (sizeof(uint64_t) + sizeof(void *));
    return sizeof(*this) + block_bytes + hash_overhead;
  }

  /**
   * @brief Compute the equivalent flat dense grid memory consumption in bytes.
   */
  [[nodiscard]] size_t equivalent_dense_bytes(
    int min_x, int min_y,
    int max_x, int max_y) const noexcept
  {
    if (min_x > max_x || min_y > max_y) {
      return 0;
    }
    const size_t width = static_cast<size_t>(max_x - min_x + 1);
    const size_t height = static_cast<size_t>(max_y - min_y + 1);
    return width * height;
  }

  /**
   * @brief Compute memory compression ratio relative to a dense bounding box.
   */
  [[nodiscard]] double compression_ratio(
    int min_x, int min_y,
    int max_x, int max_y) const noexcept
  {
    const size_t dense = equivalent_dense_bytes(min_x, min_y, max_x, max_y);
    if (dense == 0) {
      return 0.0;
    }
    const size_t sparse = memory_bytes_used();
    if (sparse >= dense) {
      return 0.0;
    }
    return 1.0 - (static_cast<double>(sparse) / static_cast<double>(dense));
  }

  [[nodiscard]] double resolution() const noexcept {return resolution_;}
  [[nodiscard]] double origin_x() const noexcept {return origin_x_;}
  [[nodiscard]] double origin_y() const noexcept {return origin_y_;}
  [[nodiscard]] uint8_t default_value() const noexcept {return default_value_;}

  [[nodiscard]] int min_cell_x() const noexcept {return min_cell_x_;}
  [[nodiscard]] int min_cell_y() const noexcept {return min_cell_y_;}
  [[nodiscard]] int max_cell_x() const noexcept {return max_cell_x_;}
  [[nodiscard]] int max_cell_y() const noexcept {return max_cell_y_;}

  /**
   * @brief Retrieve global metric bounding box enclosing all modified cells.
   */
  void get_world_bounds(
    double & min_x, double & min_y,
    double & max_x, double & max_y) const noexcept
  {
    if (blocks_.empty()) {
      min_x = origin_x_;
      min_y = origin_y_;
      max_x = origin_x_;
      max_y = origin_y_;
      return;
    }
    cell_to_world(min_cell_x_, min_cell_y_, min_x, min_y);
    cell_to_world(max_cell_x_, max_cell_y_, max_x, max_y);
  }

  /**
   * @brief Read-only accessor to the underlying spatial hash map.
   */
  const std::unordered_map<uint64_t, std::unique_ptr<BlockType>, BlockCoordHash> &
  blocks() const noexcept
  {
    return blocks_;
  }

  // =========================================================================
  // Standard Dense Grid API Parity (Costmap2D & SLAM Toolbox compatibility)
  // =========================================================================

  [[nodiscard]] unsigned int getSizeInCellsX() const noexcept
  {
    return (min_cell_x_ <= max_cell_x_) ?
           static_cast<unsigned int>(max_cell_x_ - min_cell_x_ + 1) : 0U;
  }

  [[nodiscard]] unsigned int getSizeInCellsY() const noexcept
  {
    return (min_cell_y_ <= max_cell_y_) ?
           static_cast<unsigned int>(max_cell_y_ - min_cell_y_ + 1) : 0U;
  }

  [[nodiscard]] double getSizeInMetersX() const noexcept
  {
    return static_cast<double>(getSizeInCellsX()) * resolution_;
  }

  [[nodiscard]] double getSizeInMetersY() const noexcept
  {
    return static_cast<double>(getSizeInCellsY()) * resolution_;
  }

  [[nodiscard]] double getOriginX() const noexcept {return origin_x_;}
  [[nodiscard]] double getOriginY() const noexcept {return origin_y_;}
  [[nodiscard]] double getResolution() const noexcept {return resolution_;}

  [[nodiscard]] uint8_t getCost(unsigned int cell_mx, unsigned int cell_my) const noexcept
  {
    const int cell_x = min_cell_x_ + static_cast<int>(cell_mx);
    const int cell_y = min_cell_y_ + static_cast<int>(cell_my);
    return get_cost(cell_x, cell_y);
  }

  void setCost(
    unsigned int cell_mx, unsigned int cell_my,
    uint8_t cost, double timestamp = 0.0)
  {
    const int cell_x = min_cell_x_ + static_cast<int>(cell_mx);
    const int cell_y = min_cell_y_ + static_cast<int>(cell_my);
    set_cost(cell_x, cell_y, cost, timestamp);
  }

  bool worldToMap(
    double world_x, double world_y,
    unsigned int & cell_mx, unsigned int & cell_my) const noexcept
  {
    int cell_x, cell_y;
    world_to_cell(world_x, world_y, cell_x, cell_y);
    if (cell_x < min_cell_x_ || cell_y < min_cell_y_ ||
      cell_x > max_cell_x_ || cell_y > max_cell_y_)
    {
      return false;
    }
    cell_mx = static_cast<unsigned int>(cell_x - min_cell_x_);
    cell_my = static_cast<unsigned int>(cell_y - min_cell_y_);
    return true;
  }

  void mapToWorld(
    unsigned int cell_mx, unsigned int cell_my,
    double & world_x, double & world_y) const noexcept
  {
    const int cell_x = min_cell_x_ + static_cast<int>(cell_mx);
    const int cell_y = min_cell_y_ + static_cast<int>(cell_my);
    cell_to_world(cell_x, cell_y, world_x, world_y);
  }

  /**
   * @brief Extracts a rectangular submap patch for fast scan matching or local planning.
   */
  void extract_submap(
    int min_cell_x, int min_cell_y,
    int max_cell_x, int max_cell_y,
    std::vector<uint8_t> & out_data,
    unsigned int & out_width,
    unsigned int & out_height) const
  {
    if (min_cell_x > max_cell_x || min_cell_y > max_cell_y) {
      out_width = 0;
      out_height = 0;
      out_data.clear();
      return;
    }
    out_width = static_cast<unsigned int>(max_cell_x - min_cell_x + 1);
    out_height = static_cast<unsigned int>(max_cell_y - min_cell_y + 1);
    out_data.resize(out_width * out_height);

    for (unsigned int j = 0; j < out_height; ++j) {
      const int cell_y = min_cell_y + static_cast<int>(j);
      for (unsigned int i = 0; i < out_width; ++i) {
        const int cell_x = min_cell_x + static_cast<int>(i);
        out_data[j * out_width + i] = get_cost(cell_x, cell_y);
      }
    }
  }

  /**
   * @brief Decay ephemeral obstacles across all allocated blocks.
   */
  size_t decay_ephemeral_obstacles(
    double current_time,
    double expiry_sec = 8.0,
    uint32_t min_hits = 4)
  {
    size_t decayed_blocks = 0;
    for (auto & [key, block] : blocks_) {
      if (block->decay_ephemeral(current_time, expiry_sec, min_hits)) {
        ++decayed_blocks;
      }
    }
    return decayed_blocks;
  }

  /**
   * @brief Serialize sparse grid into a standard ROS 2 nav_msgs::msg::OccupancyGrid.
   */
  void to_occupancy_grid(
    nav_msgs::msg::OccupancyGrid & msg,
    const std::string & frame_id = "map",
    const builtin_interfaces::msg::Time & stamp = builtin_interfaces::msg::Time()) const
  {
    const uint32_t width = getSizeInCellsX();
    const uint32_t height = getSizeInCellsY();
    msg.header.frame_id = frame_id;
    msg.header.stamp = stamp;
    msg.info.resolution = static_cast<float>(resolution_);
    msg.info.width = width;
    msg.info.height = height;

    if (width == 0 || height == 0) {
      msg.info.origin.position.x = origin_x_;
      msg.info.origin.position.y = origin_y_;
      msg.info.origin.position.z = 0.0;
      msg.info.origin.orientation.w = 1.0;
      msg.data.clear();
      return;
    }

    double origin_world_x, origin_world_y;
    cell_to_world(min_cell_x_, min_cell_y_, origin_world_x, origin_world_y);
    msg.info.origin.position.x = origin_world_x - 0.5 * resolution_;
    msg.info.origin.position.y = origin_world_y - 0.5 * resolution_;
    msg.info.origin.position.z = 0.0;
    msg.info.origin.orientation.w = 1.0;

    msg.data.resize(width * height, -1);

    for (uint32_t j = 0; j < height; ++j) {
      const int cell_y = min_cell_y_ + static_cast<int>(j);
      for (uint32_t i = 0; i < width; ++i) {
        const int cell_x = min_cell_x_ + static_cast<int>(i);
        const uint8_t cost = get_cost(cell_x, cell_y);

        int8_t occupancy_val = -1;
        if (cost == 0) {
          occupancy_val = 0;
        } else if (cost == 254) {
          occupancy_val = 100;
        } else if (cost == 255) {
          occupancy_val = -1;
        } else {
          occupancy_val = static_cast<int8_t>((static_cast<int>(cost) * 100) / 254);
        }
        msg.data[j * width + i] = occupancy_val;
      }
    }
  }

  /**
   * @brief Ingest a standard ROS 2 nav_msgs::msg::OccupancyGrid into sparse blocks.
   */
  void from_occupancy_grid(const nav_msgs::msg::OccupancyGrid & msg)
  {
    clear();
    resolution_ = static_cast<double>(msg.info.resolution);
    origin_x_ = msg.info.origin.position.x;
    origin_y_ = msg.info.origin.position.y;

    const uint32_t width = msg.info.width;
    const uint32_t height = msg.info.height;

    for (uint32_t j = 0; j < height; ++j) {
      for (uint32_t i = 0; i < width; ++i) {
        const int8_t occ = msg.data[j * width + i];
        if (occ < 0) {
          continue;  // Unexplored void remains an unallocated sentinel (0 bytes)
        }
        uint8_t cost = 0;
        if (occ == 0) {
          cost = 0;
        } else if (occ >= 100) {
          cost = 254;
        } else {
          cost = static_cast<uint8_t>((static_cast<int>(occ) * 254) / 100);
        }
        set_cost(static_cast<int>(i), static_cast<int>(j), cost);
      }
    }
  }

private:
  inline void update_bounds(int cell_x, int cell_y) noexcept
  {
    min_cell_x_ = std::min(min_cell_x_, cell_x);
    min_cell_y_ = std::min(min_cell_y_, cell_y);
    max_cell_x_ = std::max(max_cell_x_, cell_x);
    max_cell_y_ = std::max(max_cell_y_, cell_y);
  }

  double resolution_{0.05};
  double origin_x_{0.0};
  double origin_y_{0.0};
  uint8_t default_value_{0};
  uint32_t min_pass_through_{2};
  double occupancy_threshold_{0.1};

  int min_cell_x_{std::numeric_limits<int>::max()};
  int min_cell_y_{std::numeric_limits<int>::max()};
  int max_cell_x_{std::numeric_limits<int>::lowest()};
  int max_cell_y_{std::numeric_limits<int>::lowest()};

  std::unordered_map<uint64_t, std::unique_ptr<BlockType>, BlockCoordHash> blocks_;
};

}  // namespace adaptive
}  // namespace slam_toolbox

// Backwards-compatibility alias for adaptive_costmap_2d consumers
namespace adaptive_costmap_2d
{
template<int TileSize = 16>
using SparseGrid2D = slam_toolbox::adaptive::SparseGrid2D<TileSize>;
}  // namespace adaptive_costmap_2d

#endif  // SLAM_TOOLBOX__ADAPTIVE__SPARSE_GRID_2D_HPP_
