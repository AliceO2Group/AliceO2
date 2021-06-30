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
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUDataTypes.h"

#include <gsl/span>
#include <memory>

namespace o2
{
namespace trd
{

struct RecoInputContainer {
  gsl::span<const o2::trd::Tracklet64> mTracklets;
  gsl::span<const o2::trd::CalibratedTracklet> mSpacePoints;
  gsl::span<const o2::trd::TriggerRecord> mTriggerRecords;
  unsigned int mNTracklets;
  unsigned int mNSpacePoints;
  unsigned int mNTriggerRecords;
  std::vector<float> trdTriggerTimes;
  std::vector<int> trdTriggerIndices;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mTrackletLabels;

  void fillGPUIOPtr(o2::gpu::GPUTrackingInOutPointers* ptrs);
};

inline auto getRecoInputContainer(o2::framework::ProcessingContext& pc, o2::gpu::GPUTrackingInOutPointers* ptrs, const o2::globaltracking::RecoContainer* inputTracks, bool mc = false)
{
  auto retVal = std::make_unique<RecoInputContainer>();

  retVal->mTracklets = pc.inputs().get<gsl::span<o2::trd::Tracklet64>>("trdtracklets");
  retVal->mSpacePoints = pc.inputs().get<gsl::span<CalibratedTracklet>>("trdctracklets");
  retVal->mTriggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("trdtriggerrec");

  retVal->mNTracklets = retVal->mTracklets.size();
  retVal->mNSpacePoints = retVal->mSpacePoints.size();
  retVal->mNTriggerRecords = retVal->mTriggerRecords.size();

  if (mc) {
    retVal->mTrackletLabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("trdtrackletlabels");
  }

  for (unsigned int iEv = 0; iEv < retVal->mNTriggerRecords; ++iEv) {
    const auto& trg = retVal->mTriggerRecords[iEv];
    retVal->trdTriggerIndices.push_back(trg.getFirstTracklet());
    auto evTime = trg.getBCData().differenceInBC(inputTracks->startIR) * o2::constants::lhc::LHCBunchSpacingNS; // event time in ns
    retVal->trdTriggerTimes.push_back(evTime * 1e-3);                                                           // event time in us
  }

  if (ptrs) {
    retVal->fillGPUIOPtr(ptrs);
  }

  return std::move(retVal);
}

inline void RecoInputContainer::fillGPUIOPtr(o2::gpu::GPUTrackingInOutPointers* ptrs)
{
  ptrs->nTRDTriggerRecords = mNTriggerRecords;
  ptrs->trdTriggerTimes = &(trdTriggerTimes[0]);
  ptrs->trdTrackletIdxFirst = &(trdTriggerIndices[0]);
  ptrs->nTRDTracklets = mNTracklets;
  ptrs->trdTracklets = reinterpret_cast<const o2::gpu::GPUTRDTrackletWord*>(mTracklets.data());
  ptrs->trdSpacePoints = reinterpret_cast<const o2::gpu::GPUTRDSpacePoint*>(mSpacePoints.data());
}

} // namespace trd
} // namespace o2

#endif
