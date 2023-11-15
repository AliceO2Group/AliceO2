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

/// \file PIDBase.cxx
/// \author Felix Schlepper

#include "TRDPID/PIDBase.h"
#include "DataFormatsTRD/PID.h"
#include "Framework/Logger.h"

#ifdef TRDPID_WITH_ONNX
#include "TRDPID/ML.h"
#endif
#include "TRDPID/LQND.h"
#include "TRDPID/Dummy.h"

namespace o2
{
namespace trd
{

std::array<float, constants::NCHARGES> PIDBase::getCharges(const Tracklet64& tracklet, const int layer, const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, float snp, float tgl) const noexcept
{
  // Check z-row merging needs to be performed to recover full charge information
  if (trk.getIsCrossingNeighbor(layer) && trk.getHasNeighbor()) { // tracklet needs correction
    for (const auto& trklt : input.getTRDTracklets()) {           // search for nearby tracklet
      if (std::abs(tracklet.getPadCol() - trklt.getPadCol()) <= 1 && std::abs(tracklet.getPadRow() - trklt.getPadRow()) == 1) {
        if (tracklet.getTrackletWord() == trklt.getTrackletWord()) { // skip original tracklet
          continue;
        }

        // Add charge information
        const auto [aQ0, aQ1, aQ2] = correctCharges(tracklet, snp, tgl);
        const auto [bQ0, bQ1, bQ2] = correctCharges(tracklet, snp, tgl);
        return {aQ0 + bQ0, aQ1 + bQ1, aQ2 + bQ2};
      }
    }
  }

  return correctCharges(tracklet, snp, tgl);
}

std::array<float, constants::NCHARGES> PIDBase::correctCharges(const Tracklet64& trklt, float snp, float tgl) const noexcept
{
  auto tphi = snp / std::sqrt((1.f - snp) + (1.f + snp));
  auto trackletLength = std::sqrt(1.f + tphi * tphi + tgl * tgl);
  const float correction = mLocalGain->getValue(trklt.getHCID() / 2, trklt.getPadCol(), trklt.getPadRow()) * trackletLength;
  return {
    trklt.getQ0() / correction,
    trklt.getQ1() / correction,
    trklt.getQ2() / correction,
  };
}

std::unique_ptr<PIDBase> getTRDPIDPolicy(PIDPolicy policy)
{
  LOG(info) << "Creating PID policy. Loading model " << policy;
  switch (policy) {
    case PIDPolicy::LQ1D:
      return std::make_unique<LQ1D>(PIDPolicy::LQ1D);
    case PIDPolicy::LQ2D:
      return std::make_unique<LQ2D>(PIDPolicy::LQ2D);
    case PIDPolicy::LQ3D:
      return std::make_unique<LQ3D>(PIDPolicy::LQ3D);
#ifdef TRDPID_WITH_ONNX // Add all policies that use ONNX in this ifdef
    case PIDPolicy::XGB:
      return std::make_unique<XGB>(PIDPolicy::XGB);
    case PIDPolicy::PY:
      return std::make_unique<PY>(PIDPolicy::PY);
#endif
    case PIDPolicy::Dummy:
      return std::make_unique<Dummy>(PIDPolicy::Dummy);
    default:
      return nullptr;
  }
  return nullptr; // cannot be reached
}

} // namespace trd
} // namespace o2
