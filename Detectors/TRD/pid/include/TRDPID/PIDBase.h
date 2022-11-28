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

/// \file PIDBase.h
/// \brief This file provides the base interface for pid policies.
/// \author Felix Schlepper

#ifndef O2_TRD_PIDBASE_H
#define O2_TRD_PIDBASE_H

#include "Rtypes.h"
#include "DataFormatsTRD/PID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "Framework/ProcessingContext.h"
#include "TRDPID/PIDParameters.h"

#include <gsl/span>
#include <memory>

namespace o2
{
namespace trd
{

/// This is the PID Base class which defines the interface all other models
/// must provide.
///
/// In the most basic sense all that is needed to get a reference to the
/// (ITS-)TPC-TRD tracks and output a corresponding vector containing the
/// electron likelihood (float; < 0.f meaning that no PID is available). This
/// can than be subscribed by the aod-producer.
class PIDBase
{
 public:
  PIDBase(PIDPolicy policy) : mPolicy(policy) {}

  /// Calculate pid values.
  void process(o2::framework::ProcessingContext& pc)
  {
    init(pc);
    run();
  }

  /// Set the reference to the tracks and tracklets.
  void setInput(const o2::globaltracking::RecoContainer& input);

  /// Return PID value for ITS-TPC-TRD tracks
  auto getPIDITSTPC() const noexcept { return mPIDITSTPC; }

  /// Return PID value for TPC-TRD tracks
  auto getPIDTPC() const noexcept { return mPIDTPC; }

 protected:
  const TRDPIDParams& mParams{TRDPIDParams::Instance()}; ///< parameters
  PIDPolicy mPolicy;                                     ///< policy

  // Output
  std::vector<PIDValue> mPIDITSTPC; ///< PID for ITS-TPC-TRD tracks
  std::vector<PIDValue> mPIDTPC;    ///< PID for TPC-TRD tracks

  // Input
  gsl::span<const TrackTRD> mTracksInITSTPCTRD; ///< TRD tracks reconstructed from TPC or ITS-TPC seeds
  gsl::span<const TrackTRD> mTracksInTPCTRD;    ///< TRD tracks reconstructed from TPC or TPC seeds
  gsl::span<const Tracklet64> mTrackletsRaw;    ///< array of raw tracklets for TRD PID

 private:
  /// Initialize the policy.
  virtual void init(o2::framework::ProcessingContext& pc) = 0;

  /// Calculate for each Track the electron likelihood and place the
  /// results in the output vectors.
  virtual void run() = 0;

  ClassDefNV(PIDBase, 1);
};

/// Factory function to create a PID policy.
std::unique_ptr<PIDBase> getTRDPIDBase(PIDPolicy policy);

} // namespace trd
} // namespace o2

#endif
