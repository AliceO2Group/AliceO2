// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   PIDTPC.h
/// \author Nicolo' Jacazio
/// \since  2020-07-24
/// \brief  Implementation of the TPC detector response for PID
///

#ifndef O2_FRAMEWORK_PIDTPC_H_
#define O2_FRAMEWORK_PIDTPC_H_

// ROOT includes
#include "Rtypes.h"
#include "TMath.h"

// O2 includes
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/PID.h"
#include "PIDBase/DetectorResponse.h"

namespace o2::pid::tpc
{

/// \brief Class to handle the the TPC detector response
class Response : public DetectorResponse
{
 public:
  Response() = default;
  ~Response() override = default;

  /// Updater for the TPC response to setup the track parameters
  /// i.e. sets the track of interest
  void UpdateTrack(float mom, float tpcsignal, float tpcpoints);

  // Expected resolution
  /// Gets the expected resolution of the measurement
  float GetExpectedSigma(o2::track::PID::ID id) const override;

  // Expected signal
  /// Gets the expected signal of the measurement
  float GetExpectedSignal(o2::track::PID::ID id) const override;

  // Nsigma
  float GetSeparation(o2::track::PID::ID id) const override { return (mTPCSignal - GetExpectedSignal(id)) / GetExpectedSigma(id); }

 private:
  // Event of interest information
  // Track of interest information
  float mMomentum;  /// Momentum
  float mTPCSignal; /// TPC signal
  float mTPCPoints; /// Number of TPC points for TPC signal
};

} // namespace o2::pid::tpc

#endif // O2_FRAMEWORK_PIDTPC_H_
