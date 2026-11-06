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

#ifndef SLAM_TOOLBOX__ADAPTIVE__SPARSE_BLOCK_HPP_
#define SLAM_TOOLBOX__ADAPTIVE__SPARSE_BLOCK_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace slam_toolbox
{
namespace adaptive
{

/**
 * @brief Fixed-size cache-aligned tile storing a contiguous local 2D grid of occupancy cells.
 *
 * Designed to fit cleanly across CPU L1 cache lines (64-byte aligned).
 * Default TileSize = 16 corresponds to a 256-cell tile (exactly 256 bytes = 4 CPU cache lines).
 * Bitwise shifts (>> 4) and bitmasks (& 0x0F) provide sub-nanosecond cell indexing
 * without division or modulo operations.
 */
template<int TileSize = 16>
struct alignas(64) SparseBlock
{
  static constexpr int SIZE = TileSize;
  static constexpr int NUM_CELLS = TileSize * TileSize;
  static constexpr int SHIFT = (TileSize == 16) ? 4 : (TileSize == 32 ? 5 : 4);
  static constexpr int MASK = TileSize - 1;

  std::array<uint8_t, NUM_CELLS> cells;
  uint32_t non_default_count{0};
  uint32_t lethal_count{0};
  uint8_t default_value{0};
  double last_update_time{0.0};
  uint32_t reinforcement_hits{0};

  /**
   * @brief Construct a new sparse block initialized with default value.
   * @param initial_value Sentinel value to initialize all cells (e.g. 255 for unknown, 0 for free).
   */
  explicit SparseBlock(uint8_t initial_value = 0)
  : default_value(initial_value)
  {
    cells.fill(initial_value);
  }

  /**
   * @brief Record timestamp and increment reinforcement hit counter when an obstacle is observed.
   * @param stamp Sensor reading timestamp in seconds.
   * @param hit_obstacle True if an obstacle (cost 254) was observed in this block.
   */
  inline void touch(double stamp, bool hit_obstacle = false) noexcept
  {
    last_update_time = stamp;
    if (hit_obstacle) {
      ++reinforcement_hits;
    }
  }

  /**
   * @brief Decays unreinforced ephemeral obstacles after an elapsed timeout.
   *
   * If an obstacle was observed with fewer than min_reinforcement_hits and has not
   * been touched for expiry_sec, it is cleared to free space (cost 0).
   * Permanent structures (walls, shelves) with >= min_reinforcement_hits persist indefinitely.
   *
   * @param current_time Current clock time in seconds.
   * @param expiry_sec Expiration duration in seconds.
   * @param min_reinforcement_hits Minimum hits required for permanent persistence.
   * @return true if ephemeral obstacles were cleared, false otherwise.
   */
  inline bool decay_ephemeral(
    double current_time,
    double expiry_sec,
    uint32_t min_reinforcement_hits = 4) noexcept
  {
    if (has_lethal() && reinforcement_hits < min_reinforcement_hits) {
      if ((current_time - last_update_time) > expiry_sec) {
        fill(0);
        reinforcement_hits = 0;
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Read cell value at local tile coordinates.
   * @param local_x Local cell X coordinate within tile [0, TileSize - 1].
   * @param local_y Local cell Y coordinate within tile [0, TileSize - 1].
   * @return Cell occupancy cost.
   */
  [[nodiscard]] inline uint8_t get(int local_x, int local_y) const noexcept
  {
    return cells[(local_y & MASK) * SIZE + (local_x & MASK)];
  }

  /**
   * @brief Write cell value at local tile coordinates and update tracking counters.
   * @param local_x Local cell X coordinate within tile.
   * @param local_y Local cell Y coordinate within tile.
   * @param cost New cost value.
   */
  inline void set(int local_x, int local_y, uint8_t cost) noexcept
  {
    const int idx = (local_y & MASK) * SIZE + (local_x & MASK);
    const uint8_t old_cost = cells[idx];
    if (old_cost == cost) {
      return;
    }

    if (old_cost == default_value && cost != default_value) {
      ++non_default_count;
    } else if (old_cost != default_value && cost == default_value) {
      --non_default_count;
    }

    if (old_cost == 254 && cost != 254) {
      --lethal_count;
    } else if (old_cost != 254 && cost == 254) {
      ++lethal_count;
    }

    cells[idx] = cost;
  }

  /**
   * @brief Fill entire block with a uniform cost value.
   * @param fill_value Cost value to populate across all cells.
   */
  inline void fill(uint8_t fill_value) noexcept
  {
    cells.fill(fill_value);
    non_default_count = (fill_value == default_value) ? 0 : NUM_CELLS;
    lethal_count = (fill_value == 254) ? NUM_CELLS : 0;
  }

  /**
   * @brief Check whether all cells in this block have the default sentinel value.
   */
  [[nodiscard]] inline bool is_uniform() const noexcept
  {
    return non_default_count == 0;
  }

  /**
   * @brief Check whether this block can be reclaimed by the spatial garbage collector.
   */
  [[nodiscard]] inline bool can_prune() const noexcept
  {
    return non_default_count == 0;
  }

  /**
   * @brief Check whether any lethal obstacle cells (254) are present in this tile.
   */
  [[nodiscard]] inline bool has_lethal() const noexcept
  {
    return lethal_count > 0;
  }

  /**
   * @brief Static byte footprint of a single tile instance.
   */
  [[nodiscard]] static constexpr size_t byte_size() noexcept
  {
    return sizeof(SparseBlock<TileSize>);
  }
};

}  // namespace adaptive
}  // namespace slam_toolbox

// Backwards-compatibility alias for adaptive_costmap_2d consumers
namespace adaptive_costmap_2d
{
template<int TileSize = 16>
using SparseBlock = slam_toolbox::adaptive::SparseBlock<TileSize>;
}  // namespace adaptive_costmap_2d

#endif  // SLAM_TOOLBOX__ADAPTIVE__SPARSE_BLOCK_HPP_
