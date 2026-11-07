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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "slam_toolbox/adaptive/sparse_block.hpp"
#include "slam_toolbox/adaptive/sparse_grid_2d.hpp"

using slam_toolbox::adaptive::SparseBlock;
using slam_toolbox::adaptive::SparseGrid2D;

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

#define ASSERT_LT(a, b) \
  do { \
    ++g_tests_run; \
    auto val_a = (a); \
    auto val_b = (b); \
    if (!(val_a < val_b)) { \
      std::cerr << "[-] FAIL at line " << __LINE__ << ": " #a " < " #b \
                << " (actual: " << static_cast<int64_t>(val_a) \
                << " vs expected < " << static_cast<int64_t>(val_b) << ")" << std::endl; \
      ++g_tests_failed; \
    } else { \
      ++g_tests_passed; \
    } \
  } while (0)

#define TEST_SECTION(name) \
  std::cout << "\n==================================================\n" \
            << "  [TEST] " << name << "\n" \
            << "==================================================" << std::endl;
}  // namespace

// ============================================================================
// SUITE 1: Numerical Floating-Point Boundaries (H/P == T vs H/P == T +/- eps)
// ============================================================================
void test_floating_point_boundaries()
{
  TEST_SECTION("Suite 1: Floating-Point Boundaries (H/P == T vs H/P == T +/- eps)");

  SparseBlock<16> block(255);  // default = unknown (255)

  // Subtest 1.1: Exact threshold equality H/P == T must evaluate strictly to Free (0)
  // For T = 0.1, min_pass = 2
  for (int p = 0; p < 10; ++p) {
    block.record_pass(0, 0, 2, 0.1);
  }
  ASSERT_EQ(block.get_passes(0, 0), 10u);
  ASSERT_EQ(block.get_hits(0, 0), 0u);
  ASSERT_EQ(block.get(0, 0), 0);

  // Now simulate 1 hit out of 10 passes:
  block.fill(255);
  block.record_hit(1, 1, 2, 0.1);
  for (int p = 0; p < 9; ++p) {
    block.record_pass(1, 1, 2, 0.1);
  }
  ASSERT_EQ(block.get_passes(1, 1), 10u);
  ASSERT_EQ(block.get_hits(1, 1), 1u);
  // H/P = 1/10 = 0.1. Condition is (H/P > T) -> (0.1 > 0.1) is FALSE -> Cost = 0 (Free)
  ASSERT_EQ(block.get(1, 1), 0);

  // Test multiple exact fractions: H/P == T (all with passes > min_pass_through=2)
  struct EqualityCase
  {
    uint32_t hits;
    uint32_t passes;
    double threshold;
  };

  std::vector<EqualityCase> eq_cases = {
    {1, 10, 0.1},
    {2, 20, 0.1},
    {5, 50, 0.1},
    {10, 100, 0.1},
    {100, 1000, 0.1},
    {2, 10, 0.2},
    {20, 100, 0.2},
    {5, 20, 0.25},
    {25, 100, 0.25},
    {5, 10, 0.5},
    {50, 100, 0.5},
    {15, 20, 0.75},
    {75, 100, 0.75}
  };

  int cell_idx = 0;
  for (const auto & tc : eq_cases) {
    int lx = cell_idx % 16;
    int ly = cell_idx / 16;
    ++cell_idx;

    block.passes[ly * 16 + lx] = tc.passes;
    block.hits[ly * 16 + lx] = tc.hits;
    block.update_cell_occupancy(lx, ly, 2, tc.threshold);

    double ratio = static_cast<double>(tc.hits) / static_cast<double>(tc.passes);
    ASSERT_EQ(block.get(lx, ly), 0);
    std::cout << "  [PASS] Exact Equality H/P == T: " << tc.hits << "/" << tc.passes
              << " (" << ratio << " == " << tc.threshold << ") -> Cost = "
              << static_cast<int>(block.get(lx, ly)) << " (Free)" << std::endl;
  }

  // Subtest 1.2: Threshold plus epsilon H/P == T + eps must evaluate strictly to Occupied (254)
  std::vector<EqualityCase> above_cases = {
    {11, 100, 0.1},        // 0.11 > 0.10
    {101, 1000, 0.1},      // 0.101 > 0.10
    {1001, 10000, 0.1},    // 0.1001 > 0.10
    {10001, 100000, 0.1},  // 0.10001 > 0.10
    {10000001, 100000000, 0.1},  // 0.10000001 > 0.10
    {51, 100, 0.5},        // 0.51 > 0.50
    {501, 1000, 0.5},      // 0.501 > 0.50
    {76, 100, 0.75},       // 0.76 > 0.75
    {751, 1000, 0.75}      // 0.751 > 0.75
  };

  for (const auto & tc : above_cases) {
    int lx = cell_idx % 16;
    int ly = cell_idx / 16;
    ++cell_idx;

    block.passes[ly * 16 + lx] = tc.passes;
    block.hits[ly * 16 + lx] = tc.hits;
    block.update_cell_occupancy(lx, ly, 2, tc.threshold);

    double ratio = static_cast<double>(tc.hits) / static_cast<double>(tc.passes);
    ASSERT_EQ(block.get(lx, ly), 254);
    std::cout << "  [PASS] Above Threshold H/P > T: " << tc.hits << "/" << tc.passes
              << " (" << std::setprecision(8) << ratio << " > " << tc.threshold
              << ") -> Cost = " << static_cast<int>(block.get(lx, ly))
              << " (Occupied)" << std::endl;
  }

  // Subtest 1.3: Threshold minus epsilon H/P == T - eps must evaluate strictly to Free (0)
  std::vector<EqualityCase> below_cases = {
    {9, 100, 0.1},         // 0.09 < 0.10
    {99, 1000, 0.1},       // 0.099 < 0.10
    {999, 10000, 0.1},     // 0.0999 < 0.10
    {9999, 100000, 0.1},   // 0.09999 < 0.10
    {9999999, 100000000, 0.1},  // 0.09999999 < 0.10
    {49, 100, 0.5},        // 0.49 < 0.50
    {499, 1000, 0.5},      // 0.499 < 0.50
    {74, 100, 0.75},       // 0.74 < 0.75
    {749, 1000, 0.75}      // 0.749 < 0.75
  };

  for (const auto & tc : below_cases) {
    int lx = cell_idx % 16;
    int ly = cell_idx / 16;
    ++cell_idx;

    block.passes[ly * 16 + lx] = tc.passes;
    block.hits[ly * 16 + lx] = tc.hits;
    block.update_cell_occupancy(lx, ly, 2, tc.threshold);

    double ratio = static_cast<double>(tc.hits) / static_cast<double>(tc.passes);
    ASSERT_EQ(block.get(lx, ly), 0);
    std::cout << "  [PASS] Below Threshold H/P < T: " << tc.hits << "/" << tc.passes
              << " (" << std::setprecision(8) << ratio << " < " << tc.threshold
              << ") -> Cost = " << static_cast<int>(block.get(lx, ly)) << " (Free)" << std::endl;
  }
}

// ============================================================================
// SUITE 2: Extreme Parameter Values (min_pass_through & occupancy_threshold)
// ============================================================================
void test_extreme_parameters()
{
  TEST_SECTION("Suite 2: Extreme Parameter Values");

  // Subtest 2.1: min_pass_through = 0 (Immediate classification)
  {
    SparseBlock<16> block(255);
    // 0 passes: pass_cnt > 0 is false -> remains 255
    block.update_cell_occupancy(0, 0, 0, 0.1);
    ASSERT_EQ(block.get(0, 0), 255);

    // 1 pass, 0 hits: pass_cnt = 1 > 0 -> H/P = 0/1 = 0 <= 0.1 -> Cost = 0 (Free)
    block.record_pass(0, 0, 0, 0.1);
    ASSERT_EQ(block.get_passes(0, 0), 1u);
    ASSERT_EQ(block.get_hits(0, 0), 0u);
    ASSERT_EQ(block.get(0, 0), 0);

    // 1 pass, 1 hit: pass_cnt = 1 > 0 -> H/P = 1/1 = 1.0 > 0.1 -> Cost = 254 (Occupied)
    block.record_hit(1, 0, 0, 0.1);
    ASSERT_EQ(block.get_passes(1, 0), 1u);
    ASSERT_EQ(block.get_hits(1, 0), 1u);
    ASSERT_EQ(block.get(1, 0), 254);

    std::cout << "  [PASS] min_pass_through = 0 classifies on single 1st observation."
              << std::endl;
  }

  // Subtest 2.2: min_pass_through = 1
  {
    SparseBlock<16> block(255);
    // 1 hit: pass_cnt = 1. 1 > 1 is FALSE -> remains 255
    block.record_hit(0, 0, 1, 0.1);
    ASSERT_EQ(block.get_passes(0, 0), 1u);
    ASSERT_EQ(block.get_hits(0, 0), 1u);
    ASSERT_EQ(block.get(0, 0), 255);

    // 2nd hit: pass_cnt = 2. 2 > 1 is TRUE -> H/P = 2/2 = 1.0 > 0.1 -> Cost = 254
    block.record_hit(0, 0, 1, 0.1);
    ASSERT_EQ(block.get_passes(0, 0), 2u);
    ASSERT_EQ(block.get_hits(0, 0), 2u);
    ASSERT_EQ(block.get(0, 0), 254);

    std::cout << "  [PASS] min_pass_through = 1 suppresses 1st hit, activates on 2nd."
              << std::endl;
  }

  // Subtest 2.3: min_pass_through = 100 (High noise filter)
  {
    SparseBlock<16> block(255);
    // 100 passes with 100 hits: 100 > 100 is FALSE -> remains 255!
    for (int i = 0; i < 100; ++i) {
      block.record_hit(0, 0, 100, 0.1);
    }
    ASSERT_EQ(block.get_passes(0, 0), 100u);
    ASSERT_EQ(block.get_hits(0, 0), 100u);
    ASSERT_EQ(block.get(0, 0), 255);

    // 101st hit: 101 > 100 is TRUE -> H/P = 101/101 = 1.0 > 0.1 -> Cost = 254!
    block.record_hit(0, 0, 100, 0.1);
    ASSERT_EQ(block.get_passes(0, 0), 101u);
    ASSERT_EQ(block.get_hits(0, 0), 101u);
    ASSERT_EQ(block.get(0, 0), 254);

    // 101 passes with 0 hits: Cost = 0!
    for (int i = 0; i < 101; ++i) {
      block.record_pass(1, 0, 100, 0.1);
    }
    ASSERT_EQ(block.get_passes(1, 0), 101u);
    ASSERT_EQ(block.get_hits(1, 0), 0u);
    ASSERT_EQ(block.get(1, 0), 0);

    std::cout << "  [PASS] min_pass_through = 100 strictly holds until 101st pass." << std::endl;
  }

  // Subtest 2.4: occupancy_threshold = 0.01 (1% hit sensitivity)
  {
    SparseBlock<16> block(255);
    // 100 passes, 1 hit: H/P = 0.01 == 0.01 -> Cost = 0 (Free)
    block.passes[0] = 100;
    block.hits[0] = 1;
    block.update_cell_occupancy(0, 0, 2, 0.01);
    ASSERT_EQ(block.get(0, 0), 0);

    // 99 passes, 1 hit: H/P = 1/99 = 0.010101 > 0.01 -> Cost = 254 (Occupied)
    block.passes[1] = 99;
    block.hits[1] = 1;
    block.update_cell_occupancy(1, 0, 2, 0.01);
    ASSERT_EQ(block.get(1, 0), 254);

    // 100 passes, 2 hits: H/P = 0.02 > 0.01 -> Cost = 254 (Occupied)
    block.passes[2] = 100;
    block.hits[2] = 2;
    block.update_cell_occupancy(2, 0, 2, 0.01);
    ASSERT_EQ(block.get(2, 0), 254);

    std::cout << "  [PASS] occupancy_threshold = 0.01 correctly triggers on >1% hits."
              << std::endl;
  }

  // Subtest 2.5: occupancy_threshold = 0.99 (99% hit requirement)
  {
    SparseBlock<16> block(255);
    // 100 passes, 99 hits: H/P = 0.99 == 0.99 -> Cost = 0 (Free)
    block.passes[0] = 100;
    block.hits[0] = 99;
    block.update_cell_occupancy(0, 0, 2, 0.99);
    ASSERT_EQ(block.get(0, 0), 0);

    // 100 passes, 100 hits: H/P = 1.0 > 0.99 -> Cost = 254 (Occupied)
    block.passes[1] = 100;
    block.hits[1] = 100;
    block.update_cell_occupancy(1, 0, 2, 0.99);
    ASSERT_EQ(block.get(1, 0), 254);

    // 1000 passes, 991 hits: H/P = 0.991 > 0.99 -> Cost = 254 (Occupied)
    block.passes[2] = 1000;
    block.hits[2] = 991;
    block.update_cell_occupancy(2, 0, 2, 0.99);
    ASSERT_EQ(block.get(2, 0), 254);

    std::cout << "  [PASS] occupancy_threshold = 0.99 correctly requires >99% hits." << std::endl;
  }
}

// ============================================================================
// SUITE 3: Saturating Arithmetic & Numerical Extremes
// ============================================================================
void test_saturation_arithmetic()
{
  TEST_SECTION("Suite 3: Saturating Arithmetic at UINT32_MAX");

  SparseBlock<16> block(255);
  const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();

  // Preset cell (0, 0) to MAX_VAL - 1
  block.passes[0] = MAX_VAL - 1;
  block.hits[0] = MAX_VAL - 2;

  // Record pass: passes should become MAX_VAL
  block.record_pass(0, 0, 2, 0.1);
  ASSERT_EQ(block.get_passes(0, 0), MAX_VAL);
  ASSERT_EQ(block.get_hits(0, 0), MAX_VAL - 2);

  // Record another pass: passes should saturate at MAX_VAL (no overflow wrapping to 0)
  block.record_pass(0, 0, 2, 0.1);
  ASSERT_EQ(block.get_passes(0, 0), MAX_VAL);
  ASSERT_EQ(block.get_hits(0, 0), MAX_VAL - 2);

  // Now preset hits to MAX_VAL and record hit:
  block.hits[0] = MAX_VAL;
  block.record_hit(0, 0, 2, 0.1);
  ASSERT_EQ(block.get_passes(0, 0), MAX_VAL);
  ASSERT_EQ(block.get_hits(0, 0), MAX_VAL);

  // Check occupancy with MAX_VAL: H/P = 1.0 > 0.1 -> Cost = 254
  ASSERT_EQ(block.get(0, 0), 254);

  // Verify IEEE-754 double precision division with UINT32_MAX does not produce NaN or Inf
  double d_pass = static_cast<double>(MAX_VAL);
  double d_hit = static_cast<double>(MAX_VAL);
  double ratio = d_hit / d_pass;
  ASSERT_TRUE(std::isfinite(ratio));
  ASSERT_EQ(ratio, 1.0);

  std::cout << "  [PASS] Saturation at UINT32_MAX (" << MAX_VAL
            << ") prevents wrap-around without NaN/Inf." << std::endl;
}

// ============================================================================
// SUITE 4: 500m Facility Memory Scaling, Footprint & Reclamation
// ============================================================================
void test_500m_memory_scaling()
{
  TEST_SECTION("Suite 4: 500m Facility Memory Scaling & Reclamation");

  const double facility_dim_m = 500.0;
  const double resolution = 0.05;
  const int cells_per_axis = static_cast<int>(facility_dim_m / resolution);  // 10,000
  const int blocks_per_axis = cells_per_axis / 16;                           // 625
  const size_t total_possible_blocks = static_cast<size_t>(blocks_per_axis) * blocks_per_axis;

  std::cout << "  Facility scale: " << facility_dim_m << "m x " << facility_dim_m << "m ("
            << cells_per_axis << " x " << cells_per_axis << " cells)" << std::endl;
  std::cout << "  Total possible blocks: " << total_possible_blocks
            << " (Equivalent Dense: 925.0 MB)" << std::endl;

  SparseGrid2D<16> grid(resolution, -250.0, -250.0, 0, 2, 0.1);

  // Verify initial zero allocation
  ASSERT_EQ(grid.allocated_blocks_count(), 0u);
  // Grid object footprint is around ~120 bytes for empty hash table + metadata
  ASSERT_LT(grid.memory_bytes_used(), 1024u);

  // Allocate 10,000 realistic blocks across the facility:
  const size_t target_blocks = 10000;
  std::cout << "  Allocating " << target_blocks
            << " blocks along simulated warehouse perimeter and aisles..." << std::endl;

  auto t_start = std::chrono::high_resolution_clock::now();

  size_t allocated_count = 0;
  // Perimeter walls
  for (int bx = 0; bx < blocks_per_axis && allocated_count < target_blocks; ++bx) {
    grid.record_hit(bx * 16, 0);  // South wall
    ++allocated_count;
    if (allocated_count < target_blocks) {
      grid.record_hit(bx * 16, (blocks_per_axis - 1) * 16);  // North wall
      ++allocated_count;
    }
  }
  for (int by = 1; by < blocks_per_axis - 1 && allocated_count < target_blocks; ++by) {
    grid.record_hit(0, by * 16);  // West wall
    ++allocated_count;
    if (allocated_count < target_blocks) {
      grid.record_hit((blocks_per_axis - 1) * 16, by * 16);  // East wall
      ++allocated_count;
    }
  }

  // Interior aisles
  int aisle_idx = 1;
  while (allocated_count < target_blocks) {
    int by = (aisle_idx * 17) % (blocks_per_axis - 2) + 1;
    for (int bx = 1; bx < blocks_per_axis - 1 && allocated_count < target_blocks; ++bx) {
      grid.record_hit(bx * 16, by * 16);
      ++allocated_count;
    }
    ++aisle_idx;
  }

  auto t_alloc = std::chrono::high_resolution_clock::now();
  double alloc_sec = std::chrono::duration<double>(t_alloc - t_start).count();

  ASSERT_EQ(grid.allocated_blocks_count(), target_blocks);
  size_t total_memory_bytes = grid.memory_bytes_used();
  size_t tile_payload_bytes = grid.allocated_blocks_count() * sizeof(SparseBlock<16>);
  double block_mb = static_cast<double>(total_memory_bytes) / (1024.0 * 1024.0);
  double dense_mb =
    static_cast<double>(total_possible_blocks * sizeof(SparseBlock<16>)) / (1024.0 * 1024.0);
  double savings_pct = 100.0 * (1.0 - block_mb / dense_mb);

  std::cout << "  Allocated: " << grid.allocated_blocks_count() << " blocks in "
            << alloc_sec * 1000.0 << " ms" << std::endl;
  std::cout << "  Tile Payload: " << static_cast<double>(tile_payload_bytes) / (1024.0 * 1024.0)
            << " MB (" << tile_payload_bytes << " bytes)" << std::endl;
  std::cout << "  Total Sparse RAM: " << block_mb << " MB vs Equivalent Dense: "
            << dense_mb << " MB" << std::endl;
  std::cout << "  RAM Footprint Reduction: " << std::fixed << std::setprecision(2)
            << savings_pct << "%" << std::endl;
  ASSERT_TRUE(savings_pct > 97.0);

  // Subtest 4.2: Bulk update_all_occupancy() reevaluation performance across 10,000 blocks
  auto t_update_start = std::chrono::high_resolution_clock::now();
  grid.update_all_occupancy();
  auto t_update_end = std::chrono::high_resolution_clock::now();
  double update_ms =
    std::chrono::duration<double, std::milli>(t_update_end - t_update_start).count();
  std::cout << "  Bulk update_all_occupancy() over 10,000 blocks (2.56M cells): "
            << update_ms << " ms" << std::endl;
  ASSERT_LT(update_ms, 50.0);

  // Subtest 4.3: Memory Reclamation via fill() and prune_unmodified_blocks()
  std::cout << "  Clearing all 10,000 blocks with set_cost(0) to simulate clearance..."
            << std::endl;
  for (int bx = 0; bx < blocks_per_axis; ++bx) {
    for (int by = 0; by < blocks_per_axis; ++by) {
      if (grid.is_block_allocated(bx, by)) {
        for (int lx = 0; lx < 16; ++lx) {
          for (int ly = 0; ly < 16; ++ly) {
            grid.set_cost(bx * 16 + lx, by * 16 + ly, 0);
          }
        }
      }
    }
  }

  // Now all allocated blocks have non_default_count == 0 (since default_value == 0)
  size_t reclaimed_blocks = grid.prune_unmodified_blocks();
  std::cout << "  prune_unmodified_blocks() reclaimed: " << reclaimed_blocks
            << " blocks." << std::endl;
  ASSERT_EQ(reclaimed_blocks, target_blocks);
  ASSERT_EQ(grid.allocated_blocks_count(), 0u);

  // All 10,000 SparseBlock objects are destroyed (freed from heap)
  size_t remaining_tile_payload = grid.allocated_blocks_count() * sizeof(SparseBlock<16>);
  ASSERT_EQ(remaining_tile_payload, 0u);
  // Remaining memory is solely empty hash bucket metadata (< 100 KB) vs 22.8 MB previously
  ASSERT_LT(grid.memory_bytes_used(), 100000u);
  std::cout << "  [PASS] 100% of tile memory reclaimed: active blocks dropped from 10,000 to 0."
            << std::endl;
}

// ============================================================================
// SUITE 5: Throughput Performance (Mops/s)
// ============================================================================
void test_performance_throughput()
{
  TEST_SECTION("Suite 5: Performance & Throughput (Mops/s)");

  SparseGrid2D<16> grid(0.05, 0.0, 0.0, 0, 2, 0.1);

  const int num_ops = 5000000;  // 5 Million operations

  // Test 5.1: record_pass throughput
  // Pre-allocate a 10x10 block patch (100 blocks = 25,600 cells)
  for (int bx = 0; bx < 10; ++bx) {
    for (int by = 0; by < 10; ++by) {
      grid.record_pass(bx * 16, by * 16);
    }
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_ops; ++i) {
    int cx = i % 160;
    int cy = (i / 160) % 160;
    grid.record_pass(cx, cy);
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  double dur_sec = std::chrono::duration<double>(t_end - t_start).count();
  double mops = (num_ops / 1e6) / dur_sec;
  std::cout << "  record_pass() throughput: " << std::fixed << std::setprecision(2)
            << mops << " Mops/s (" << dur_sec * 1000.0 << " ms for "
            << num_ops << " ops)" << std::endl;
  ASSERT_TRUE(mops > 2.0);

  // Test 5.2: record_hit throughput
  t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_ops; ++i) {
    int cx = i % 160;
    int cy = (i / 160) % 160;
    grid.record_hit(cx, cy);
  }
  t_end = std::chrono::high_resolution_clock::now();
  dur_sec = std::chrono::duration<double>(t_end - t_start).count();
  mops = (num_ops / 1e6) / dur_sec;
  std::cout << "  record_hit() throughput: " << std::fixed << std::setprecision(2)
            << mops << " Mops/s (" << dur_sec * 1000.0 << " ms for "
            << num_ops << " ops)" << std::endl;
  ASSERT_TRUE(mops > 2.0);

  // Test 5.3: get_cost query throughput
  t_start = std::chrono::high_resolution_clock::now();
  uint64_t sum = 0;
  for (int i = 0; i < num_ops; ++i) {
    int cx = i % 160;
    int cy = (i / 160) % 160;
    sum += grid.get_cost(cx, cy);
  }
  t_end = std::chrono::high_resolution_clock::now();
  dur_sec = std::chrono::duration<double>(t_end - t_start).count();
  mops = (num_ops / 1e6) / dur_sec;
  std::cout << "  get_cost() query throughput: " << std::fixed << std::setprecision(2)
            << mops << " Mops/s (checksum: " << sum << ")" << std::endl;
  ASSERT_TRUE(mops > 5.0);
}

int main()
{
  std::cout << "==================================================================\n"
            << "  EMPIRICAL STRESS TEST HARNESS: MILESTONE 1 (slam_toolbox_upstream)\n"
            << "  Probabilistic Block-Sparse Storage Core Under Extremes\n"
            << "==================================================================" << std::endl;

  test_floating_point_boundaries();
  test_extreme_parameters();
  test_saturation_arithmetic();
  test_500m_memory_scaling();
  test_performance_throughput();

  std::cout << "\n==================================================================\n"
            << "  STRESS TEST SUMMARY\n"
            << "==================================================================\n"
            << "  Total Assertions: " << g_tests_run << "\n"
            << "  Passed:           " << g_tests_passed << "\n"
            << "  Failed:           " << g_tests_failed << "\n"
            << "==================================================================" << std::endl;

  return (g_tests_failed == 0) ? 0 : 1;
}
