// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCDef.h
/// \author David Rohr, Sergey Gorbunov

// clang-format off
#ifndef GPUTPCDEF_H
#define GPUTPCDEF_H

#include "GPUDef.h"

#define CALINK_INVAL ((calink) -1)
namespace GPUCA_NAMESPACE
{
namespace gpu
{
#if defined(GPUCA_O2_LIB) || defined(GPUCA_O2_INTERFACE)
typedef unsigned int calink;
typedef unsigned int cahit;
#else
typedef unsigned int calink;
typedef unsigned int cahit;
#endif

struct cahit2{cahit x, y;};
}
} // GPUCA_NAMESPACE::GPU

#ifdef GPUCA_TPC_USE_STAT_ERROR
  #define GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
#endif

#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME // Needs full clusterdata
  #define GPUCA_FULL_CLUSTERDATA
#endif

#if defined(GPUCA_STANDALONE) || defined(GPUCA_GPUCODE) // No support for Full Field Propagator or Statistical errors
  #ifdef GPUCA_GM_USE_FULL_FIELD
    #undef GPUCA_GM_USE_FULL_FIELD
  #endif
  #ifdef GPUCA_TPC_USE_STAT_ERROR
    #undef GPUCA_TPC_USE_STAT_ERROR
  #endif
#endif

#ifdef GPUCA_EXTERN_ROW_HITS
  #define CA_GET_ROW_HIT(iRow) tracker.TrackletRowHits()[iRow * s.mNTracklets + r.mItr]
  #define CA_SET_ROW_HIT(iRow, val) tracker.TrackletRowHits()[iRow * s.mNTracklets + r.mItr] = val
#else
  #define CA_GET_ROW_HIT(iRow) tracklet.RowHit(iRow)
  #define CA_SET_ROW_HIT(iRow, val) tracklet.SetRowHit(iRow, val)
#endif

#endif //GPUDTPCEF_H
// clang format on
