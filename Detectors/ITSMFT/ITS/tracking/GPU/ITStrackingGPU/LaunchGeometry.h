// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file LaunchGeometry.h
/// \brief Compile-time launch geometry of the ITS tracking kernels, per GPU family.
/// Poor man's RTC
/// to be removed/reworked entirely once we can use Gabriele's tuner
///

#ifndef ITSTRACKINGGPU_LAUNCHGEOMETRY_H_
#define ITSTRACKINGGPU_LAUNCHGEOMETRY_H_

namespace o2::its::gpu
{

#if defined(GPUCA_GPUTYPE_VEGA) // gfx906: MI50, Radeon VII
constexpr int ComputeUnits = 60;
constexpr int WarpSize = 64;
#elif defined(GPUCA_GPUTYPE_MI100)     // gfx908
constexpr int ComputeUnits = 120;
constexpr int WarpSize = 64;
#elif defined(GPUCA_GPUTYPE_MI210)     // gfx90a
constexpr int ComputeUnits = 104;
constexpr int WarpSize = 64;
#elif defined(GPUCA_GPUTYPE_MI300)     // gfx942: MI300X (MI300A has 228)
constexpr int ComputeUnits = 304;
constexpr int WarpSize = 64;
#elif defined(GPUCA_GPUTYPE_RDNA)      // gfx10xx/11xx consumer parts, wave32
constexpr int ComputeUnits = 60;
constexpr int WarpSize = 32;
#elif defined(GPUCA_GPUTYPE_BLACKWELL) // sm_120: RTX 5080
constexpr int ComputeUnits = 84;
constexpr int WarpSize = 32;
#elif defined(GPUCA_GPUTYPE_HOPPER)    // sm_90: H100
constexpr int ComputeUnits = 132;
constexpr int WarpSize = 32;
#elif defined(GPUCA_GPUTYPE_ADA)       // sm_89: RTX 4090
constexpr int ComputeUnits = 128;
constexpr int WarpSize = 32;
#elif defined(GPUCA_GPUTYPE_AMPERE)    // sm_80/86: A100 has 108, RTX 3090 has 82
constexpr int ComputeUnits = 108;
constexpr int WarpSize = 32;
#elif defined(GPUCA_GPUTYPE_TURING)    // sm_75: RTX 2080 Ti
constexpr int ComputeUnits = 68;
constexpr int WarpSize = 32;
#else
// this is the fallback as we had it before
constexpr int ComputeUnits = 60;
constexpr int WarpSize = 64;
#endif

constexpr int GPUThreads = 256;
constexpr int DefaultBlocksPerComputeUnit = 4;
constexpr int MaxBlocksPerComputeUnit = 10;

/// Minimum resident blocks per compute unit to request when no per-kernel measurement exists.
struct KernelOccupancy {
  int computeLayerTracklets{1};
  int computeLayerCells{1};
  int computeLayerCellNeighbours{1};
  int processNeighboursCellSeed{1};
  int processNeighboursTrackSeed{1};
  int fitTrackSeeds{1};
  int fitTrackSeedsExtended{1};
  int compileLookupTable{1};

  /// Return the smallest occupancy value in the table.
  constexpr int min() const
  {
    const int a{computeLayerTracklets < computeLayerCells ? computeLayerTracklets : computeLayerCells};
    const int b{computeLayerCellNeighbours < processNeighboursCellSeed ? computeLayerCellNeighbours : processNeighboursCellSeed};
    const int c{processNeighboursTrackSeed < fitTrackSeeds ? processNeighboursTrackSeed : fitTrackSeeds};
    const int d{fitTrackSeedsExtended < compileLookupTable ? fitTrackSeedsExtended : compileLookupTable};
    const int ab{a < b ? a : b};
    const int cd{c < d ? c : d};
    return ab < cd ? ab : cd;
  }

  /// Return the largest occupancy value in the table.
  constexpr int max() const
  {
    const int a{computeLayerTracklets > computeLayerCells ? computeLayerTracklets : computeLayerCells};
    const int b{computeLayerCellNeighbours > processNeighboursCellSeed ? computeLayerCellNeighbours : processNeighboursCellSeed};
    const int c{processNeighboursTrackSeed > fitTrackSeeds ? processNeighboursTrackSeed : fitTrackSeeds};
    const int d{fitTrackSeedsExtended > compileLookupTable ? fitTrackSeedsExtended : compileLookupTable};
    const int ab{a > b ? a : b};
    const int cd{c > d ? c : d};
    return ab > cd ? ab : cd;
  }
};

/// Use the same occupancy floor for every kernel when no per-kernel measurements are available.
constexpr KernelOccupancy uniformOccupancy(int minBlocks)
{
  return {.computeLayerTracklets = minBlocks,
          .computeLayerCells = minBlocks,
          .computeLayerCellNeighbours = minBlocks,
          .processNeighboursCellSeed = minBlocks,
          .processNeighboursTrackSeed = minBlocks,
          .fitTrackSeeds = minBlocks,
          .fitTrackSeedsExtended = minBlocks,
          .compileLookupTable = minBlocks};
}

#if defined(GPUCA_GPUTYPE_VEGA) // gfx906: MI50, Radeon VII

/// Per-kernel minimum occupancy floors measured on gfx906.
constexpr KernelOccupancy MinBlocks{
  .computeLayerTracklets = 2,
  .computeLayerCells = 3,
  .computeLayerCellNeighbours = 3,
  .processNeighboursCellSeed = 3,
  .processNeighboursTrackSeed = 3,
  .fitTrackSeeds = 4,
  .fitTrackSeedsExtended = 3, // untested: the follower is compiled out of every default iteration
  .compileLookupTable = 1,
};

/// Number of blocks per CU used to size the grid for the measured gfx906 kernels.
constexpr KernelOccupancy ResidentBlocks{
  .computeLayerTracklets = 4,      //  56 VGPR
  .computeLayerCells = 3,          //  84 VGPR
  .computeLayerCellNeighbours = 3, //  84 VGPR
  .processNeighboursCellSeed = 3,  //  84 VGPR
  .processNeighboursTrackSeed = 3, //  84 VGPR
  .fitTrackSeeds = 4,              //  64 VGPR
  .fitTrackSeedsExtended = 3,      //  84 VGPR
  .compileLookupTable = 4,         //   8 VGPR,
};

#elif defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
/// Other AMD parts: unmeasured.
constexpr KernelOccupancy MinBlocks = uniformOccupancy(3);
constexpr KernelOccupancy ResidentBlocks = uniformOccupancy(DefaultBlocksPerComputeUnit);
#else
/// NVIDIA: unmeasured.
constexpr KernelOccupancy MinBlocks = uniformOccupancy(1);
constexpr KernelOccupancy ResidentBlocks = uniformOccupancy(DefaultBlocksPerComputeUnit);
#endif

/// Number of blocks in a grid whose depth is residentBlocksPerComputeUnit blocks per CU.
constexpr int gridBlocks(int residentBlocksPerComputeUnit)
{
  return ComputeUnits * residentBlocksPerComputeUnit;
}

/// Number of threads covered by a grid whose depth is residentBlocksPerComputeUnit blocks per CU.
constexpr int gridThreads(int residentBlocksPerComputeUnit)
{
  return gridBlocks(residentBlocksPerComputeUnit) * GPUThreads;
}

static_assert(MinBlocks.min() >= 1,
              "an occupancy floor below one resident block is meaningless");

static_assert(MinBlocks.max() <= MaxBlocksPerComputeUnit,
              "the occupancy floor cannot exceed the blocks a CU can hold");

static_assert(ResidentBlocks.min() >= 1,
              "every kernel must have at least one resident block per CU");

static_assert(ResidentBlocks.max() <= MaxBlocksPerComputeUnit,
              "resident blocks per CU cannot exceed what a CU can hold");

/// The grid must provide at least as many blocks per CU as the corresponding occupancy floor.
constexpr bool residentCoversFloor()
{
  return ResidentBlocks.computeLayerTracklets >= MinBlocks.computeLayerTracklets &&
         ResidentBlocks.computeLayerCells >= MinBlocks.computeLayerCells &&
         ResidentBlocks.computeLayerCellNeighbours >= MinBlocks.computeLayerCellNeighbours &&
         ResidentBlocks.processNeighboursCellSeed >= MinBlocks.processNeighboursCellSeed &&
         ResidentBlocks.processNeighboursTrackSeed >= MinBlocks.processNeighboursTrackSeed &&
         ResidentBlocks.fitTrackSeeds >= MinBlocks.fitTrackSeeds &&
         ResidentBlocks.fitTrackSeedsExtended >= MinBlocks.fitTrackSeedsExtended &&
         ResidentBlocks.compileLookupTable >= MinBlocks.compileLookupTable;
}

static_assert(residentCoversFloor(), "a kernel's grid is narrower than the occupancy its __launch_bounds__ floor demands");

static_assert(GPUThreads % WarpSize == 0, "block size must be a whole number of warps/waves");

static_assert(ComputeUnits > 0 && GPUThreads > 0 && DefaultBlocksPerComputeUnit > 0, "degenerate launch geometry");

} // namespace o2::its::gpu

#endif // ITSTRACKINGGPU_LAUNCHGEOMETRY_H_
