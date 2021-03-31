// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDSpacePointInternal.h
/// \brief This data structure stores the TRD space point used internally by the tracking code

/// \author Ole Schmidt

#ifndef GPUTRDSPACEPOINTINTERNAL_H
#define GPUTRDSPACEPOINTINTERNAL_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// struct to hold the information on the space points
struct GPUTRDSpacePointInternal {
  float mR;                 // x position (3.5 mm above anode wires) - radial offset due to t0 mis-calibration, measured -1 mm for run 245353
  float mX[2];              // y and z position (sector coordinates)
  float mDy;                // deflection over drift length
  unsigned short mVolumeId; // basically derived from TRD chamber number
  GPUd() GPUTRDSpacePointInternal(float x, float y, float z, float dy) : mR(x), mDy(dy), mVolumeId(0)
  {
    mX[0] = y;
    mX[1] = z;
  }
  GPUd() GPUTRDSpacePointInternal() : mR(0), mDy(0), mVolumeId(0)
  {
    mX[0] = 0;
    mX[1] = 0;
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDSPACEPOINTINTERNAL_H
