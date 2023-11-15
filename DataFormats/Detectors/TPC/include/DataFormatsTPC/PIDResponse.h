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
#include "GPUCommonMath.h"
#include "ReconstructionDataFormats/PID.h"
#include "DataFormatsTPC/PIDResponse.h"
#include "DataFormatsTPC/BetheBlochAleph.h"
#include "DataFormatsTPC/TrackTPC.h"

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
  GPUd() o2::track::PID::ID getMostProbablePID(const TrackTPC& track, float PID_EKrangeMin, float PID_EKrangeMax, float PID_EPrangeMin, float PID_EPrangeMax, float PID_EDrangeMin, float PID_EDrangeMax, float PID_ETrangeMin, float PID_ETrangeMax, char PID_useNsigma, float PID_sigma) const;

 private:
  float mBetheBlochParams[5] = {0.19310481, 4.26696118, 0.00522579, 2.38124907, 0.98055396}; // BBAleph average fit parameters
  float mMIP = 50.f;
  float mChargeFactor = 2.299999952316284f;

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(PIDResponse, 1);
#endif
};

GPUd() void PIDResponse::setBetheBlochParams(const float betheBlochParams[5])
{
  for (int i = 0; i < 5; i++) {
    mBetheBlochParams[i] = betheBlochParams[i];
  }
}

GPUd() float PIDResponse::getExpectedSignal(const TrackTPC& track, const o2::track::PID::ID id) const
{
  const float bg = static_cast<float>(track.getP() / o2::track::pid_constants::sMasses[id]);
  if (bg < 0.05) {
    return -999.;
  }
  const float bethe = mMIP * o2::tpc::BetheBlochAleph(bg, mBetheBlochParams[0], mBetheBlochParams[1], mBetheBlochParams[2], mBetheBlochParams[3], mBetheBlochParams[4]) * o2::gpu::GPUCommonMath::Pow(static_cast<float>(o2::track::pid_constants::sCharges[id]), mChargeFactor);
  return bethe >= 0. ? bethe : -999.;
}

// get most probable PID
GPUd() o2::track::PID::ID PIDResponse::getMostProbablePID(const TrackTPC& track, float PID_EKrangeMin, float PID_EKrangeMax, float PID_EPrangeMin, float PID_EPrangeMax, float PID_EDrangeMin, float PID_EDrangeMax, float PID_ETrangeMin, float PID_ETrangeMax, char PID_useNsigma, float PID_sigma) const
{
  const float dEdx = track.getdEdx().dEdxTotTPC;

  if (dEdx < 10.) {
    return o2::track::PID::Pion;
  }

  auto id = o2::track::PID::Electron;
  float distanceMin = 0.;
  float dEdxExpected = getExpectedSignal(track, id);
  if (PID_useNsigma) {
    // using nSigma
    distanceMin = o2::gpu::GPUCommonMath::Abs((dEdx - dEdxExpected) / (PID_sigma * dEdxExpected));
  } else {
    // using absolute distance
    distanceMin = o2::gpu::GPUCommonMath::Abs(dEdx - dEdxExpected);
  }

  // calculate the distance to the expected dEdx signals
  // start from Pion to exlude Muons
  for (o2::track::PID::ID i = o2::track::PID::Pion; i < o2::track::PID::NIDs; i++) {
    float distance = 0.;
    dEdxExpected = getExpectedSignal(track, i);
    if (PID_useNsigma) {
      // using nSigma
      distance = o2::gpu::GPUCommonMath::Abs((dEdx - dEdxExpected) / (PID_sigma * dEdxExpected));
    } else {
      // using absolute distance
      distance = o2::gpu::GPUCommonMath::Abs(dEdx - dEdxExpected);
    }
    if (distance < distanceMin) {
      id = i;
      distanceMin = distance;
    }
  }

  // override the electrons to the closest alternative PID in the bands crossing regions
  if (id == o2::track::PID::Electron) {
    const float p = track.getP();
    if ((p > PID_EKrangeMin) && (p < PID_EKrangeMax)) {
      id = o2::track::PID::Kaon;
    } else if ((p > PID_EPrangeMin) && (p < PID_EPrangeMax)) {
      id = o2::track::PID::Proton;
    } else if ((p > PID_EDrangeMin) && (p < PID_EDrangeMax)) {
      id = o2::track::PID::Deuteron;
    } else if ((p > PID_ETrangeMin) && (p < PID_ETrangeMax)) {
      id = o2::track::PID::Triton;
    }
  }

  return id;
}

} // namespace o2::tpc

#endif
