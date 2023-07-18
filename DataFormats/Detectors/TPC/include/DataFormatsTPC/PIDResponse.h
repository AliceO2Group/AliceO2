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
/// @file   PIDResponse.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
///

#ifndef AliceO2_TPC_PIDResponse_H
#define AliceO2_TPC_PIDResponse_H

// o2 includes
#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::tpc
{
class TrackTPC;

/// \brief PID response class
///
/// This class is used to handle the TPC PID response.
///

class PIDResponse
{
 public:
  /// default constructor
  PIDResponse() CON_DEFAULT;

  /// default destructor
  ~PIDResponse() CON_DEFAULT;

  /// setters
  GPUd() void setBetheBlochParams(const float betheBlochParams[5]);
  GPUd() void setMIP(float mip) { mMIP = mip; }
  GPUd() void setChargeFactor(float chargeFactor) { mChargeFactor = chargeFactor; }

  /// getters
  GPUd() const float* getBetheBlochParams() const { return mBetheBlochParams; }
  GPUd() float getMIP() const { return mMIP; }
  GPUd() float getChargeFactor() const { return mChargeFactor; }

  /// get expected signal of the track
  GPUd() float getExpectedSignal(const TrackTPC& track, const o2::track::PID::ID id) const;

  /// get most probable PID of the track
  GPUd() o2::track::PID::ID getMostProbablePID(const TrackTPC& track) const;

 private:
  float mBetheBlochParams[5] = {0.19310481, 4.26696118, 0.00522579, 2.38124907, 0.98055396}; // BBAleph average fit parameters
  float mMIP = 50.f;
  float mChargeFactor = 2.299999952316284f;

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(PIDResponse, 1);
#endif
};
} // namespace o2::tpc

#endif
