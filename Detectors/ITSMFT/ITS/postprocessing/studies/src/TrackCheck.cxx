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

#include <ITSStudies/TrackCheck.h>
#include <Framework/Task.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <DataFormatsGlobalTracking/RecoContainer.h>

#include <DataFormatsITS/TrackITS.h>
#include <SimulationDataFormat/MCTrack.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;

class TrackCheckStudy : public Task
{
  struct ParticleInfo {
    int event;
    int pdg;
    float pt;
    float eta;
    float phi;
    int mother;
    int first;
    unsigned short clusters = 0u;
    unsigned char isReco = 0u;
    unsigned char isFake = 0u;
    bool isPrimary = 0u;
    unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
    o2::its::TrackITS track;
  };

 public:
  TrackCheckStudy(std::shared_ptr<DataRequest> dr, mask_t src) : mDataRequest(dr), mTracksSrc(src){};
  ~TrackCheckStudy() final = default;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);

  // Data
  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;
  gsl::span<const o2::dataformats::MCEventHeader> mHeader;
  gsl::span<const o2::MCTrack> mMCTracks;
};

void TrackCheckStudy::run(ProcessingContext& pc)
{
  LOGP(info, "Called run function from trackcheckstudy");
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  // mHeader = pc.inputs().get<gsl::span<o2::dataformats::MCEventHeader>>("MCHeader"); TODO: header is not messageable, left unmanaged for now
  mMCTracks = pc.inputs().get<gsl::span<o2::MCTrack>>("MCTracks");
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void TrackCheckStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto itsTracksROFRecords = recoData.getITSTracksROFRecords();
  auto itsTracks = recoData.getITSTracks();
  auto itsTracksMCLabels = recoData.getITSTracksMCLabels();

  LOGP(info, "Got {} rofs", itsTracksROFRecords.size());
  LOGP(info, "Got {} tracks", itsTracks.size());
  LOGP(info, "Got {} labels", itsTracksMCLabels.size());
  // LOGP(info, "Got {} MCHeaders", mHeader.size());
  LOGP(info, "Got {} MCTracks", mMCTracks.size());
}

void TrackCheckStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
  }
}

void TrackCheckStudy::endOfStream(EndOfStreamContext& ec)
{
}

void TrackCheckStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getTrackCheckStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);

  std::vector<InputSpec> inputs = dataRequest->inputs;
  // inputs.emplace_back("MCHeader", "MC", "MCHEADER", 0, Lifetime::Timeframe);  TODO: Headers we cannot get them
  inputs.emplace_back("MCTracks", "MC", "MCTRACKS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "its-study-check-tracks",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackCheckStudy>(dataRequest, srcTracksMask)},
    Options{}};
}

} // namespace study
} // namespace its
} // namespace o2