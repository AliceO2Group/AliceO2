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

/// \file GPUDefConstantsAndSettings.h
/// \author David Rohr

// This files contains compile-time constants affecting the GPU algorithms / reconstruction results.
// Architecture-dependant compile-time constants affecting the performance without changing the results are stored in GPUDefParameters*.h

#ifndef GPUDEFCONSTANTSANDSETTINGS_H
#define GPUDEFCONSTANTSANDSETTINGS_H

// clang-format off

#include "GPUCommonDef.h"

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)
  #error You are using the CA GPU tracking without defining the build type (O2/Standalone). If you are running an O2 ROOT macro, please include GPUO2ExternalUser.h first!
#endif

#if (defined(GPUCA_O2_LIB) && defined(GPUCA_STANDALONE))
  #error Invalid Compile Definitions, need to build for either O2 or Standalone!
#endif

#define GPUCA_TPC_MIN_HITS_B5(QPTB5) (CAMath::Abs(QPTB5) > 10 ? 10 : (CAMath::Abs(QPTB5) > 5 ? 15 : 29)) // Minimum hits should depend on Pt, low Pt tracks can have few hits. 29 Hits default, 15 for < 200 mev, 10 for < 100 mev

#define GPUCA_MERGER_MAX_TRACK_CLUSTERS 1024          // Maximum number of clusters a track may have after merging

#define GPUCA_MAXN 40                                 // Maximum number of neighbor hits to consider in one row in neightbors finder

#define GPUCA_MAX_SIN_PHI_LOW 0.99f                   // Limits for maximum sin phi during fit
#define GPUCA_MAX_SIN_PHI 0.999f                      // Must be preprocessor define because c++ pre 11 cannot use static constexpr for initializes

#define GPUCA_MIN_BIN_SIZE 2.f                        // Minimum bin size in TPC fast access grid
#define GPUCA_MAX_BIN_SIZE 1000.f                     // Maximum bin size in TPC fast access grid

#define GPUCA_TPC_COMP_CHUNK_SIZE 1024                // Chunk size of sorted unattached TPC cluster in compression

#define TPC_MAX_TIME_BIN_TRIGGERED 600

#if defined(GPUCA_NSECTORS) || defined(GPUCA_NROWS)
  #error GPUCA_NSECTORS or GPUCA_NROWS already defined, do not include GPUTPCGeometry.h before!
#endif
#if !defined(GPUCA_RUN2) && !(defined(ROOT_VERSION_CODE) && ROOT_VERSION_CODE < 393216)
  //Use definitions from the O2 headers if available for nicer code and type safety
  #include "DataFormatsTPC/Constants.h"
  #define GPUCA_NSECTORS o2::tpc::constants::MAXSECTOR
  #define GPUCA_NROWS o2::tpc::constants::MAXGLOBALPADROW
#else
  //Define it manually, if O2 headers not available, ROOT5, and OpenCL 1.2, which do not know C++11.
  #define GPUCA_NSECTORS 36
  #ifndef GPUCA_RUN2
    #define GPUCA_NROWS 152
  #else
    #define GPUCA_NROWS 159
  #endif
#endif

//#define GPUCA_MERGER_BY_MC_LABEL                    // Use MC labels for TPC track merging - for performance studies // TODO: Cleanup unneeded options

// clang-format on

#endif
