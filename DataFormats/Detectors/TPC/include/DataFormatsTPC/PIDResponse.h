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
#include "ReconstructionDataFormats/PID.h"
#include "DataFormatsTPC/TrackTPC.h"

namespace o2::tpc
{

/// \brief PID response class
///
/// This class is used to handle the TPC PID response.
///

class PIDResponse
{
 public:
  /// default constructor
  PIDResponse() = default;

  /// default destructor
  ~PIDResponse() = default;

  /// setters
  void setBetheBlochParams(const std::array<double, 5>& betheBlochParams) { mBetheBlochParams = betheBlochParams; }
  void setMIP(double mip) { mMIP = mip; }
  void setChargeFactor(double chargeFactor) { mChargeFactor = chargeFactor; }

  /// getters
  std::array<double, 5> getBetheBlochParams() const { return mBetheBlochParams; }
  double getMIP() const { return mMIP; }
  double getChargeFactor() const { return mChargeFactor; }

  /// get expected signal of the track
  double getExpectedSignal(const TrackTPC& track, const o2::track::PID::ID id) const;

  /// get most probable PID of the track
  o2::track::PID::ID getMostProbablePID(const TrackTPC& track) const;

 private:
  std::array<double, 5> mBetheBlochParams = {0.19310481, 4.26696118, 0.00522579, 2.38124907, 0.98055396}; // BBAleph average fit parameters
  double mMIP = 50.f;
  double mChargeFactor = 2.299999952316284f;
};
} // namespace o2::tpc

#endif
