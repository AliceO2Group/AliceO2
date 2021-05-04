// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   RecoInputContainer.h
/// \author ole.schmidt@cern.ch
/// \brief Struct for input data required by TRD tracking workflow

#ifndef O2_TRD_RECOINPUTCONTAINER_H
#define O2_TRD_RECOINPUTCONTAINER_H

#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonConstants/LHCConstants.h"

#include "GPUDataTypes.h"

#include <gsl/span>
#include <memory>

namespace o2
{
namespace trd
{

struct RecoInputContainer {

  gsl::span<const o2::dataformats::TrackTPCITS> mTracksTPCITS;
  gsl::span<const o2::tpc::TrackTPC> mTracksTPC;
  gsl::span<const o2::trd::Tracklet64> mTracklets;
  gsl::span<const o2::trd::CalibratedTracklet> mSpacePoints;
  gsl::span<const o2::trd::TriggerRecord> mTriggerRecords;
  unsigned int mNTracksTPCITS;
  unsigned int mNTracksTPC;
  unsigned int mNTracklets;
  unsigned int mNSpacePoints;
  unsigned int mNTriggerRecords;
  std::vector<float> trdTriggerTimes;
  std::vector<int> trdTriggerIndices;
};

inline auto getRecoInputContainer(o2::framework::ProcessingContext& pc, o2::gpu::GPUTrackingInOutPointers* ptrs, const o2::globaltracking::RecoContainer* inputTracks)
{
  auto retVal = std::make_unique<RecoInputContainer>();
  retVal->mTracksTPCITS = inputTracks->getTPCITSTracks<o2::dataformats::TrackTPCITS>();
  retVal->mTracksTPC = inputTracks->getTPCTracks<o2::tpc::TrackTPC>();
  retVal->mTracklets = pc.inputs().get<gsl::span<o2::trd::Tracklet64>>("trdtracklets");
  retVal->mSpacePoints = pc.inputs().get<gsl::span<CalibratedTracklet>>("trdctracklets");
  retVal->mTriggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("trdtriggerrec");

  retVal->mNTracksTPCITS = retVal->mTracksTPCITS.size();
  retVal->mNTracksTPC = retVal->mTracksTPC.size();
  retVal->mNTracklets = retVal->mTracklets.size();
  retVal->mNSpacePoints = retVal->mSpacePoints.size();
  retVal->mNTriggerRecords = retVal->mTriggerRecords.size();

  for (unsigned int iEv = 0; iEv < retVal->mNTriggerRecords; ++iEv) {
    const auto& trg = retVal->mTriggerRecords[iEv];
    retVal->trdTriggerIndices.push_back(trg.getFirstTracklet());
    int64_t evTime = trg.getBCData().toLong() * o2::constants::lhc::LHCBunchSpacingNS; // event time in ns
    retVal->trdTriggerTimes.push_back(evTime / 1000.);                                 // event time in us
  }

  if (ptrs) {
    if (ptrs->nOutputTracksTPCO2 == 0 && retVal->mNTracksTPC) {
      ptrs->nOutputTracksTPCO2 = retVal->mNTracksTPC;
      ptrs->outputTracksTPCO2 = retVal->mTracksTPC.data();
    }
    if (ptrs->nTracksTPCITSO2 == 0 && retVal->mNTracksTPCITS) {
      ptrs->nTracksTPCITSO2 = retVal->mNTracksTPCITS;
      ptrs->tracksTPCITSO2 = retVal->mTracksTPCITS.data();
    }
    ptrs->nTRDTriggerRecords = retVal->mNTriggerRecords;
    ptrs->trdTriggerTimes = &(retVal->trdTriggerTimes[0]);
    ptrs->trdTrackletIdxFirst = &(retVal->trdTriggerIndices[0]);
    ptrs->nTRDTracklets = retVal->mNTracklets;
    ptrs->trdTracklets = reinterpret_cast<const o2::gpu::GPUTRDTrackletWord*>(retVal->mTracklets.data());
    ptrs->trdSpacePoints = reinterpret_cast<const o2::gpu::GPUTRDSpacePoint*>(retVal->mSpacePoints.data());
  }

  return std::move(retVal);
}

} // namespace trd
} // namespace o2

#endif
