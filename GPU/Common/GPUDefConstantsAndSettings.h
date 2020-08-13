// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefConstantsAndSettings.h
/// \author David Rohr

// This files contains compile-time constants affecting the GPU algorithms / reconstruction results.
// Architecture-dependant compile-time constants affecting the performance without changing the results are stored in GPUDefGPUParameters.h

#ifndef GPUDEFCONSTANTSANDSETTINGS_H
#define GPUDEFCONSTANTSANDSETTINGS_H

// clang-format off

#include "GPUCommonDef.h"

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)
  #error You are using the CA GPU tracking without defining the build type (O2/AliRoot/Standalone). If you are running an O2 ROOT macro, please include GPUO2Interface.h first!
#endif

#if (defined(GPUCA_ALIROOT_LIB) && defined(GPUCA_O2_LIB)) || (defined(GPUCA_ALIROOT_LIB) && defined(GPUCA_STANDALONE)) || (defined(GPUCA_O2_LIB) && defined(GPUCA_STANDALONE))
  #error Invalid Compile Definitions, need to build for either AliRoot or O2 or Standalone
#endif

#define GPUCA_TRACKLET_SELECTOR_MIN_HITS(QPT) (CAMath::Abs(QPT) > 10 ? 10 : (CAMath::Abs(QPT) > 5 ? 15 : 29)) // Minimum hits should depend on Pt, low Pt tracks can have few hits. 29 Hits default, 15 for < 200 mev, 10 for < 100 mev

#define GPUCA_GLOBAL_TRACKING_RANGE 45                // Number of rows from the upped/lower limit to search for global track candidates in for
#define GPUCA_GLOBAL_TRACKING_Y_RANGE_UPPER 0.85      // Inner portion of y-range in slice that is not used in searching for global track candidates
#define GPUCA_GLOBAL_TRACKING_Y_RANGE_LOWER 0.85
#define GPUCA_GLOBAL_TRACKING_MIN_ROWS 10             // Min num of rows an additional global track must span over
#define GPUCA_GLOBAL_TRACKING_MIN_HITS 8              // Min num of hits for an additional global track

#define GPUCA_MERGER_CE_ROWLIMIT 5                    //Distance from first / last row in order to attempt merging accross CE
#define GPUCA_MERGER_LOOPER_QPT_LIMIT 4               // Min Q/Pt to run special looper merging procedure
#define GPUCA_MERGER_HORIZONTAL_DOUBLE_QPT_LIMIT 2    // Min Q/Pt to attempt second horizontal merge between slices after a vertical merge was found
#define GPUCA_MERGER_MAX_TRACK_CLUSTERS 1000          // Maximum number of clusters a track may have after merging

#define GPUCA_Y_FACTOR 4                              // Weight of y residual vs z residual in tracklet constructor
#define GPUCA_MAXN 40                                 // Maximum number of neighbor hits to consider in one row in neightbors finder
#define GPUCA_TRACKLET_CONSTRUCTOR_MAX_ROW_GAP 4      // Maximum number of consecutive rows without hit in track following
#define GPUCA_TRACKLET_CONSTRUCTOR_MAX_ROW_GAP_SEED 2 // Same, but during fit of seed
#define GPUCA_MERGER_MAXN_MISSED_HARD 10              // Hard limit for number of missed rows in fit / propagation
#define GPUCA_MERGER_COV_LIMIT 1000                   // Abort fit when y/z cov exceed the limit
#define GPUCA_MIN_TRACK_PT_DEFAULT 0.010              // Default setting for minimum track Pt at some places

#define GPUCA_MAX_SIN_PHI_LOW 0.99f                   // Limits for maximum sin phi during fit
#define GPUCA_MAX_SIN_PHI 0.999f                      // Must be preprocessor define because c++ pre 11 cannot use static constexpr for initializes

#define GPUCA_MIN_BIN_SIZE 2.f                        // Minimum bin size in TPC fast access grid
#define GPUCA_MAX_BIN_SIZE 1000.f                     // Maximum bin size in TPC fast access grid

#define GPUCA_TPC_COMP_CHUNK_SIZE 1024                // Chunk size of sorted unattached TPC cluster in compression

#if defined(HAVE_O2HEADERS) && (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !(defined(ROOT_VERSION_CODE) && ROOT_VERSION_CODE < 393216)
  //Use definitions from the O2 headers if available for nicer code and type safety
  #include "DataFormatsTPC/Constants.h"
  #define GPUCA_NSLICES o2::tpc::constants::MAXSECTOR
  #define GPUCA_ROW_COUNT o2::tpc::constants::MAXGLOBALPADROW
#else
  //Define it manually, if O2 headers not available, ROOT5, and OpenCL 1.2, which do not know C++11.
  #define GPUCA_NSLICES 36
  #ifdef GPUCA_TPC_GEOMETRY_O2
    #define GPUCA_ROW_COUNT 152
  #else
    #define GPUCA_ROW_COUNT 159
  #endif
#endif

//#define GPUCA_MERGER_BY_MC_LABEL                    // Use MC labels for TPC track merging - for performance studies
//#define GPUCA_FULL_CLUSTERDATA                      // Store all cluster information in the cluster data, also those not needed for tracking.
//#define GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME        // Propagate Pad, Row, Time cluster information to GM
//#define GPUCA_GM_USE_FULL_FIELD                     // Use offline magnetic field during GMPropagator prolongation
//#define GPUCA_TPC_USE_STAT_ERROR                    // Use statistical errors from offline in track fit

// clang-format on

#endif
