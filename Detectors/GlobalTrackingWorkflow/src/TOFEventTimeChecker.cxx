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

/// @file   TOFEventTimeChecker.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"

// from Tracks
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"

// from TOF
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
#include "TOFReconstruction/EventTimeMaker.h"
//#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFEventTimeChecker.h"

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GID = o2::dataformats::GlobalTrackID;
using MyTrack = o2::tof::eventTimeTrackTest;

bool MyFilter(const MyTrack& tr)
{
  return (tr.mP < 2.0);
} // accept all

namespace o2
{
namespace globaltracking
{

class TOFEventTimeChecker : public Task
{
 public:
  TOFEventTimeChecker(std::shared_ptr<DataRequest> dr, bool useMC) : mDataRequest(dr), mUseMC(useMC) {}
  ~TOFEventTimeChecker() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void fillMatching(GID gid);
  void processEvent(std::vector<MyTrack>& tracks);

 private:
  bool mIsTPC;
  bool mIsTPCTRD;
  bool mIsITSTPCTRD;
  bool mIsITSTPC;
  gsl::span<const o2::tof::Cluster> mTOFClustersArrayInp; ///< input TOF clusters
  std::vector<MyTrack> mMyTracks;

  RecoContainer mRecoData;
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TOFEventTimeChecker::processEvent(std::vector<MyTrack>& tracks)
{
  auto evtime = o2::tof::evTimeMaker<std::vector<MyTrack>, MyTrack, MyFilter>(tracks);
  float et = evtime.eventTime;
  float erret = evtime.eventTimeError;

  printf("Event time = %f +/- %f\n", et, erret);
}

void TOFEventTimeChecker::fillMatching(GID gid)
{
  if (!gid.includesDet(DetID::TOF)) {
    return;
  }
  const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
  const o2::track::TrackLTIntegral& info = match.getLTIntegralOut();

  MyTrack trk;
  int trksource = 5;
  if (gid.getSource() == GID::TPCTOF) {
    const auto& array = mRecoData.getTPCTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getP();
    trksource = 0;
  } else if (gid.getSource() == GID::ITSTPCTOF) {
    const auto& array = mRecoData.getTPCITSTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 1;
  } else if (gid.getSource() == GID::TPCTRDTOF) {
    const auto& array = mRecoData.getTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 2;
  } else if (gid.getSource() == GID::ITSTPCTRDTOF) {
    const auto& array = mRecoData.getITSTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 3;
  }

  const char* sources[5] = {"TPC", "ITS-TPC", "TPC-TRD", "ITS-TPC-TRD", "NONE"};

  trk.expTimes[0] = info.getTOF(2); // pi
  trk.expTimes[1] = info.getTOF(3); // ka
  trk.expTimes[2] = info.getTOF(4); // pr
  trk.expSigma[0] = 120;            // dummy resolution (to be updated)
  trk.expSigma[1] = 120;            // dummy resolution (to be updated)
  trk.expSigma[2] = 120;            // dummy resolution (to be updated)

  //  int tofcl = match.getIdxTOFCl();
  //  trk.mSignal = mTOFClustersArrayInp[tofcl].getTime();
  trk.mSignal = match.getSignal();
  trk.mTOFChi2 = match.getChi2();
  trk.mLength = info.getL();
  //  trk.mHypo = 0;

  mMyTracks.push_back(trk);
}

void TOFEventTimeChecker::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry("", false);
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
}

void TOFEventTimeChecker::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  mMyTracks.clear();
  mRecoData.collectData(pc, *mDataRequest.get());

  mIsTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF));
  mIsITSTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF));
  mIsITSTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF));
  mIsTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF));

  mTOFClustersArrayInp = mRecoData.getTOFClusters();

  LOG(DEBUG) << "isTrackSourceLoaded: TPC -> " << mIsTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: ITSTPC -> " << mIsITSTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: TPCTRD -> " << mIsTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: ITSTPCTRD -> " << mIsITSTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ")";
  LOG(DEBUG) << "TOF cluster size = " << mTOFClustersArrayInp.size();

  if (!mTOFClustersArrayInp.size()) {
    return;
  }

  auto creator = [this](auto& trk, GID gid, float time0, float terr) {
    this->fillMatching(gid);
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  // sorting matching in time
  std::sort(mMyTracks.begin(), mMyTracks.end(),
            [](MyTrack a, MyTrack b) { return a.tofSignal() < b.tofSignal(); });

  for (auto& element : mMyTracks) { // loop print
    LOG(INFO) << "Time cluster = " << element.tofSignal() << " ps - pt = " << element.pt();
  }

  std::vector<MyTrack> tracks;
  for (int i = 0; i < mMyTracks.size(); i++) { // loop looking for interaction candidates
    tracks.clear();
    int ntrk = 1;
    double time = mMyTracks[i].tofSignal();
    tracks.emplace_back(mMyTracks[i]);
    for (; i < mMyTracks.size(); i++) {
      double timeCurrent = mMyTracks[i].tofSignal();
      if (timeCurrent - time > 25E3) {
        i--;
        break;
      }
      tracks.emplace_back(mMyTracks[i]);
      ntrk++;
    }
    if (ntrk > 2) { // good candidate with time
      processEvent(tracks);
    }
  }

  mTimer.Stop();
}

void TOFEventTimeChecker::endOfStream(EndOfStreamContext& ec)
{
  LOGF(DEBUG, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTOFEventTimeCheckerSpec(GID::mask_t src, bool useMC)
{
  auto dataRequest = std::make_shared<DataRequest>();

  // request TOF clusters
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::TOF), useMC);
  dataRequest->requestTOFMatches(src, useMC);

  return DataProcessorSpec{
    "tof-eventime",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<TOFEventTimeChecker>(dataRequest, useMC)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
