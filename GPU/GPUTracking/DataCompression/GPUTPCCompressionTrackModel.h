// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompressionTrackModel.h
/// \author David Rohr

#ifndef GPUTPCCOMPRESSIONTRACKMODEL_H
#define GPUTPCCOMPRESSIONTRACKMODEL_H

// For debugging purposes, we provide means to use other track models
#define GPUCA_COMPRESSION_TRACK_MODEL_MERGER
//#define GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER

#include "GPUDef.h"

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMTrackParam.h"
#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
#error Not yet implemented

#else // Default internal track model for compression
#error Not yet implemented

#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
// ATTENTION! This track model is used for the data compression.
// Changes to the propagation and fit will prevent the decompression of data
// encoded with the old version!!!

struct GPUParam;

class GPUTPCCompressionTrackModel
{
 public:
  void Init(float x, float y, float z, float alpha, unsigned char qPt, const GPUParam& proc);
  int Propagate(float x, float alpha);
  int Filter(float y, float z, int iRow);
  int Mirror();

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
  float Y() const
  {
    return mTrk.GetY();
  }
  float Z() const { return mTrk.GetZ(); }

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)

#else // Default internal track model for compression

#endif

 protected:
  const GPUParam* mParam;

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
  GPUTPCGMPropagator mProp;
  GPUTPCGMTrackParam mTrk;

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)

#else // Default internal track model for compression

#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
