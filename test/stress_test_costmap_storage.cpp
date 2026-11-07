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

#include "adaptive_costmap_2d/sparse_block.hpp"
#include "adaptive_costmap_2d/sparse_grid_2d.hpp"
#include "adaptive_costmap_2d/karto_sparse_grid.hpp"

using adaptive_costmap_2d::SparseBlock;
using adaptive_costmap_2d::SparseGrid2D;
using adaptive_costmap_2d::SparseKartoGrid;

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

#define TEST_SECTION(name) \
  std::cout << "\n==================================================\n" \
            << "  [TEST] " << name << "\n" \
            << "==================================================" << std::endl;
}  // namespace

// ============================================================================
// SUITE 1: Numerical Floating-Point Boundaries on adaptive_costmap_2d
// ============================================================================
void test_floating_point_boundaries()
{
  TEST_SECTION("Suite 1: Floating-Point Boundaries on adaptive_costmap_2d::SparseBlock");

  SparseBlock<16> block(255);

  // Exact threshold equality cases (all with passes > min_pass_through=2)
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

  // Above threshold H/P > T
  std::vector<EqualityCase> above_cases = {
    {11, 100, 0.1},        // 0.11 > 0.10
    {101, 1000, 0.1},      // 0.101 > 0.10
    {1001, 10000, 0.1},    // 0.1001 > 0.10
    {51, 100, 0.5},        // 0.51 > 0.50
    {76, 100, 0.75}        // 0.76 > 0.75
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

  // Below threshold H/P < T
  std::vector<EqualityCase> below_cases = {
    {9, 100, 0.1},         // 0.09 < 0.10
    {99, 1000, 0.1},       // 0.099 < 0.10
    {999, 10000, 0.1},     // 0.0999 < 0.10
    {49, 100, 0.5},        // 0.49 < 0.50
    {74, 100, 0.75}        // 0.74 < 0.75
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
// SUITE 2: Extreme Parameter Values on adaptive_costmap_2d
// ============================================================================
void test_extreme_parameters()
{
  TEST_SECTION("Suite 2: Extreme Parameter Values on adaptive_costmap_2d");

  // Subtest 2.1: min_pass_through = 0
  {
    SparseGrid2D<16> grid(0.05, 0.0, 0.0, 255, 0, 0.1);
    ASSERT_EQ(grid.get_min_pass_through(), 0u);
    ASSERT_EQ(grid.get_occupancy_threshold(), 0.1);

    // 1 pass: immediately classifies to 0
    grid.record_pass(5, 5);
    ASSERT_EQ(grid.get_passes(5, 5), 1u);
    ASSERT_EQ(grid.get_hits(5, 5), 0u);
    ASSERT_EQ(grid.get_cost(5, 5), 0);

    // 1 hit: immediately classifies to 254
    grid.record_hit(6, 6);
    ASSERT_EQ(grid.get_passes(6, 6), 1u);
    ASSERT_EQ(grid.get_hits(6, 6), 1u);
    ASSERT_EQ(grid.get_cost(6, 6), 254);
    std::cout << "  [PASS] Grid with min_pass_through = 0 classifies on single 1st observation."
              << std::endl;
  }

  // Subtest 2.2: min_pass_through = 100
  {
    SparseGrid2D<16> grid(0.05, 0.0, 0.0, 255, 100, 0.1);
    ASSERT_EQ(grid.get_min_pass_through(), 100u);

    for (int i = 0; i < 100; ++i) {
      grid.record_hit(10, 10);
    }
    ASSERT_EQ(grid.get_passes(10, 10), 100u);
    ASSERT_EQ(grid.get_hits(10, 10), 100u);
    ASSERT_EQ(grid.get_cost(10, 10), 255);  // strictly untouched / sentinel

    grid.record_hit(10, 10);  // 101st pass
    ASSERT_EQ(grid.get_passes(10, 10), 101u);
    ASSERT_EQ(grid.get_hits(10, 10), 101u);
    ASSERT_EQ(grid.get_cost(10, 10), 254);  // confirmed lethal obstacle
    std::cout << "  [PASS] Grid with min_pass_through = 100 filters 100 hits, classifies on 101st."
              << std::endl;
  }

  // Subtest 2.3: occupancy_threshold = 0.01 & 0.99
  {
    SparseGrid2D<16> grid_low(0.05, 0.0, 0.0, 255, 2, 0.01);
    SparseGrid2D<16> grid_high(0.05, 0.0, 0.0, 255, 2, 0.99);

    // Low threshold: 1 hit in 99 passes (1/99 = 0.0101 > 0.01) -> 254
    grid_low.record_hit(0, 0);
    for (int i = 0; i < 98; ++i) {
      grid_low.record_pass(0, 0);
    }
    ASSERT_EQ(grid_low.get_passes(0, 0), 99u);
    ASSERT_EQ(grid_low.get_hits(0, 0), 1u);
    ASSERT_EQ(grid_low.get_cost(0, 0), 254);

    // High threshold: 99 hits in 100 passes (99/100 = 0.99 <= 0.99) -> 0
    for (int i = 0; i < 99; ++i) {
      grid_high.record_hit(0, 0);
    }
    grid_high.record_pass(0, 0);  // 1 pass without hit -> passes=100, hits=99
    ASSERT_EQ(grid_high.get_passes(0, 0), 100u);
    ASSERT_EQ(grid_high.get_hits(0, 0), 99u);
    ASSERT_EQ(grid_high.get_cost(0, 0), 0);

    // 100th hit -> 100/100 = 1.0 > 0.99 -> 254
    grid_high.record_hit(0, 0);
    ASSERT_EQ(grid_high.get_passes(0, 0), 101u);
    ASSERT_EQ(grid_high.get_hits(0, 0), 100u);  // 100/101 = 0.990099 > 0.99
    ASSERT_EQ(grid_high.get_cost(0, 0), 254);

    std::cout << "  [PASS] Low/High thresholds (0.01 and 0.99) function with exact sensitivity."
              << std::endl;
  }
}

// ============================================================================
// SUITE 3: Memory Reclamation via prune_unmodified_blocks()
// ============================================================================
void test_memory_reclamation()
{
  TEST_SECTION("Suite 3: Memory Reclamation via prune_unmodified_blocks()");

  SparseGrid2D<16> grid(0.05, 0.0, 0.0, 0, 2, 0.1);

  const size_t num_blocks = 2000;
  std::cout << "  Allocating " << num_blocks << " blocks..." << std::endl;
  for (size_t b = 0; b < num_blocks; ++b) {
    grid.record_hit(b * 16, b * 16);
  }
  ASSERT_EQ(grid.allocated_blocks_count(), num_blocks);

  // Dilute back to free space (cost 0 == default_value)
  for (size_t b = 0; b < num_blocks; ++b) {
    grid.set_cost(b * 16, b * 16, 0);
  }

  // Prune unmodified blocks
  size_t reclaimed = grid.prune_unmodified_blocks();
  ASSERT_EQ(reclaimed, num_blocks);
  ASSERT_EQ(grid.allocated_blocks_count(), 0u);
  std::cout << "  [PASS] prune_unmodified_blocks() cleanly deallocated all "
            << reclaimed << " blocks." << std::endl;
}

// ============================================================================
// SUITE 4: SparseKartoGrid Adapter Verification
// ============================================================================
void test_karto_sparse_grid_adapter()
{
  TEST_SECTION("Suite 4: SparseKartoGrid Adapter Integration");

  SparseKartoGrid karto_grid(200, 200, 0.05, 2, 0.1);

  ASSERT_EQ(karto_grid.get_min_pass_through(), 2u);
  ASSERT_EQ(karto_grid.get_occupancy_threshold(), 0.1);

  // Test parameter setters/getters
  karto_grid.set_min_pass_through(5);
  ASSERT_EQ(karto_grid.get_min_pass_through(), 5u);
  karto_grid.set_occupancy_threshold(0.2);
  ASSERT_EQ(karto_grid.get_occupancy_threshold(), 0.2);

  // Reset to default
  karto_grid.set_min_pass_through(2);
  karto_grid.set_occupancy_threshold(0.1);

  // Noise filtering: 2 hits -> passes=2, hits=2 <= min_pass_through -> StateUnknown (0), cost 255
  karto_grid.record_hit(50, 50);
  karto_grid.record_hit(50, 50);
  ASSERT_EQ(karto_grid.get_passes(50, 50), 2u);
  ASSERT_EQ(karto_grid.get_hits(50, 50), 2u);
  ASSERT_EQ(karto_grid.get_state(50, 50), SparseKartoGrid::StateUnknown);
  ASSERT_EQ(karto_grid.get_underlying_sparse_grid().get_cost(50, 50), 255);

  // 3rd hit -> passes=3, hits=3 > 2 -> H/P = 1.0 > 0.1 -> StateOccupied (1), cost 254
  karto_grid.record_hit(50, 50);
  ASSERT_EQ(karto_grid.get_passes(50, 50), 3u);
  ASSERT_EQ(karto_grid.get_hits(50, 50), 3u);
  ASSERT_EQ(karto_grid.get_state(50, 50), SparseKartoGrid::StateOccupied);
  ASSERT_EQ(karto_grid.get_underlying_sparse_grid().get_cost(50, 50), 254);

  // Dynamic obstacle dilution: 30 passes -> passes=33, hits=3 -> Free (StateFree=2, cost 0)
  for (int i = 0; i < 30; ++i) {
    karto_grid.record_pass(50, 50);
  }
  ASSERT_EQ(karto_grid.get_passes(50, 50), 33u);
  ASSERT_EQ(karto_grid.get_hits(50, 50), 3u);
  ASSERT_EQ(karto_grid.get_state(50, 50), SparseKartoGrid::StateFree);
  ASSERT_EQ(karto_grid.get_underlying_sparse_grid().get_cost(50, 50), 0);

  // Bulk update_all_occupancy forwarding test
  karto_grid.set_occupancy_threshold(0.05);
  // Now H/P = 3/33 = 0.0909 > 0.05! Cell should become Occupied upon update_all_occupancy!
  karto_grid.update_all_occupancy();
  ASSERT_EQ(karto_grid.get_state(50, 50), SparseKartoGrid::StateOccupied);
  ASSERT_EQ(karto_grid.get_underlying_sparse_grid().get_cost(50, 50), 254);

  std::cout << "  [PASS] SparseKartoGrid adapter seamlessly forwards all probabilistic methods."
            << std::endl;
}

int main()
{
  std::cout << "==================================================================\n"
            << "  EMPIRICAL STRESS TEST HARNESS: MILESTONE 1 (adaptive_costmap_2d)\n"
            << "  Storage Core, Parameters, Reclamation, and Karto Adapter\n"
            << "==================================================================" << std::endl;

  test_floating_point_boundaries();
  test_extreme_parameters();
  test_memory_reclamation();
  test_karto_sparse_grid_adapter();

  std::cout << "\n==================================================================\n"
            << "  STRESS TEST SUMMARY\n"
            << "==================================================================\n"
            << "  Total Assertions: " << g_tests_run << "\n"
            << "  Passed:           " << g_tests_passed << "\n"
            << "  Failed:           " << g_tests_failed << "\n"
            << "==================================================================" << std::endl;

  return (g_tests_failed == 0) ? 0 : 1;
}
