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

/// @file   FilteringSpec.h

#ifndef O2_DATA_FILTERING_SPEC
#define O2_DATA_FILTERING_SPEC

#include "DataFormatsGlobalTracking/FilteredRecoTF.h"

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsFV0/RecPoints.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsEMCAL/EventHandler.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ConcreteDataMatcher.h"
#include "DataFormatsGlobalTracking/FilteredRecoTF.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Steer/MCKinematicsReader.h"
#include "TMap.h"
#include "TStopwatch.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2::filtering
{

class FilteringSpec : public Task
{
 public:
  FilteringSpec(GID::mask_t src, std::shared_ptr<DataRequest> dataRequest, bool enableSV, bool useMC = true)
    : mInputSources(src), mDataRequest(dataRequest), mEnableSV(enableSV), mUseMC(useMC) {}
  ~FilteringSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;

 private:
  void fillData(const o2::globaltracking::RecoContainer& recoData);
  void processTracksOfVertex(const o2::dataformats::VtxTrackRef& vtxref, const o2::globaltracking::RecoContainer& recoData);
  int processBarrelTrack(GIndex idx, const o2::globaltracking::RecoContainer& recoData);
  bool selectTrack(GIndex id, const o2::globaltracking::RecoContainer& recoData);
  void updateTimeDependentParams(ProcessingContext& pc);
  void clear();

  o2::dataformats::FilteredRecoTF mFTF{};

  bool mUseMC = true;
  bool mEnableSV = true; // enable secondary vertices

  o2::InteractionRecord mStartIR{};
  GID::mask_t mInputSources;
  TStopwatch mTimer;

  bool mNeedToSave = false;                // flag that there was something selected to save
  std::map<int, int> mITSTrackIDCache{};   // cache for selected ITS track IDS
  std::map<int, int> mITSClusterIDCache{}; // cache for selected ITS clusters

  // unordered map connects global indices and table indices of barrel tracks
  std::unordered_map<GIndex, int> mGIDToTableID;

  std::shared_ptr<DataRequest> mDataRequest;

  // CCDB conditions
  const o2::itsmft::TopologyDictionary* mDictITS = nullptr;
};

/// create a processor spec
framework::DataProcessorSpec getDataFilteringSpec(GID::mask_t src, bool enableSV, bool useMC);

} // namespace o2::filtering

#endif /* O2_DATA_FILTERING_SPEC */
