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

/// @file   TOFMatchChecker.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/NameConf.h"
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
#include "DetectorsBase/GRPGeomHelper.h"

// from TOF
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
//#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFMatchChecker.h"

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class TOFMatchChecker : public Task
{
 public:
  TOFMatchChecker(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~TOFMatchChecker() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void checkMatching(GID gid);
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  bool mIsTPC;
  bool mIsTPCTRD;
  bool mIsITSTPCTRD;
  bool mIsITSTPC;
  gsl::span<const o2::tof::Cluster> mTOFClustersArrayInp; ///< input TOF clusters

  RecoContainer mRecoData;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TOFMatchChecker::checkMatching(GID gid)
{
  if (!gid.includesDet(DetID::TOF)) {
    return;
  }
  const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);

  int trksource = 5;
  if (gid.getSource() == GID::TPCTOF) {
    trksource = 0;
  } else if (gid.getSource() == GID::ITSTPCTOF) {
    trksource = 1;
  } else if (gid.getSource() == GID::TPCTRDTOF) {
    trksource = 2;
  } else if (gid.getSource() == GID::ITSTPCTRDTOF) {
    trksource = 3;
  }

  const char* sources[5] = {"TPC", "ITS-TPC", "TPC-TRD", "ITS-TPC-TRD", "NONE"};

  int tofcl = match.getIdxTOFCl();
  int trIndex = match.getTrackIndex();
  float chi2 = match.getChi2();
  float ttof = mTOFClustersArrayInp[tofcl].getTime();
  float x = mTOFClustersArrayInp[tofcl].getX();
  float y = mTOFClustersArrayInp[tofcl].getY();
  float z = mTOFClustersArrayInp[tofcl].getZ();
  int sector = mTOFClustersArrayInp[tofcl].getSector();
  LOG(info) << "trkSource=" << sources[trksource] << " -- cl=" << tofcl << " - trk=" << trIndex << " - chi2 =" << chi2 << " - time=" << ttof << " - coordinates (to be rotated, sector=" << sector << ") = (" << x << "," << y << "," << z << ")";

  // digit coordinates
  int mainCh = mTOFClustersArrayInp[tofcl].getMainContributingChannel();
  int addCh;
  int det[5];
  float pos[3];
  o2::tof::Geo::getVolumeIndices(mainCh, det);
  o2::tof::Geo::getPos(det, pos);
  LOG(debug) << "Cluster mult = " << mTOFClustersArrayInp[tofcl].getNumOfContributingChannels() << " - main channel (" << mainCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";

  // top->+48, bottom->-48, left->-1, right->+1
  if (mTOFClustersArrayInp[tofcl].getNumOfContributingChannels() > 1) {
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kUpRight)) {
      addCh = mainCh + 48 + 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "top right (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kUp)) {
      addCh = mainCh + 48;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "top (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kUpLeft)) {
      addCh = mainCh + 48 - 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "top left (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kRight)) {
      addCh = mainCh + 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "right (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kLeft)) {
      addCh = mainCh - 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "left (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kDownRight)) {
      addCh = mainCh - 48 + 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "down right (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kDown)) {
      addCh = mainCh - 48;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "down (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
    if (mTOFClustersArrayInp[tofcl].isAdditionalChannelSet(o2::tof::Cluster::kDownLeft)) {
      addCh = mainCh - 48 - 1;
      o2::tof::Geo::getVolumeIndices(addCh, det);
      o2::tof::Geo::getPos(det, pos);
      LOG(debug) << "down left (" << addCh << ") -> pos(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
    }
  }

  LOG(debug) << "";
}

void TOFMatchChecker::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void TOFMatchChecker::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  mRecoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);

  mIsTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF));
  mIsITSTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF));
  mIsITSTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF));
  mIsTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF));

  mTOFClustersArrayInp = mRecoData.getTOFClusters();

  LOG(debug) << "isTrackSourceLoaded: TPC -> " << mIsTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: ITSTPC -> " << mIsITSTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: TPCTRD -> " << mIsTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTRD -> " << mIsITSTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ")";
  LOG(debug) << "TOF cluster size = " << mTOFClustersArrayInp.size();

  if (!mTOFClustersArrayInp.size()) {
    return;
  }

  auto creator = [this](auto& trk, GID gid, float time0, float terr) {
    this->checkMatching(gid);
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  mTimer.Stop();
}

void TOFMatchChecker::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void TOFMatchChecker::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

void TOFMatchChecker::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // put here one-time inits
  }
  // we may have other params which need to be queried regularly
}

DataProcessorSpec getTOFMatchCheckerSpec(GID::mask_t src, bool useMC)
{
  auto dataRequest = std::make_shared<DataRequest>();

  // request TOF clusters
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::TOF), useMC);
  dataRequest->requestTOFMatches(src, useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "tof-matcher",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<TOFMatchChecker>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
