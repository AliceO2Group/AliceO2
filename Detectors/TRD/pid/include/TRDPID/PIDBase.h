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
#include "DataFormatsTRD/TrackTRD.h"
#include "TRDPID/PIDParameters.h"
#include "Framework/ProcessingContext.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

#include <gsl/span>
#include <memory>

namespace o2
{
namespace trd
{

/// This is the PID Base class which defines the interface all other models
/// must provide.
///
/// A 'policy' describes how a PID value (PIDValue) should be
/// calculated. For the classical algorithms there is no
/// initialization needed since these work off LUTs. However, for ML
/// models some initialization is needed, e.g. creating the
/// ONNXRuntime Session.
///
/// Afterwards, a PID value can be calculated via the given policy for
/// each TrackTRD.
class PIDBase
{
 public:
  virtual ~PIDBase() = default;
  PIDBase(PIDPolicy policy) : mPolicy(policy) {}

  /// Initialize the policy.
  virtual void init(o2::framework::ProcessingContext& pc) = 0;

  /// Calculate a PID for a given track.
  virtual PIDValue process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, bool isTPC) = 0;

 protected:
  const TRDPIDParams& mParams{TRDPIDParams::Instance()}; ///< parameters
  PIDPolicy mPolicy;                                     ///< policy

 private:
  ClassDefNV(PIDBase, 1);
};

/// Factory function to create a PID policy.
std::unique_ptr<PIDBase> getTRDPIDBase(PIDPolicy policy);

} // namespace trd
} // namespace o2

#endif
