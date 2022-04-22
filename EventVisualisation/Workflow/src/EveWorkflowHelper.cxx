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

/// \file EveWorkflowHelper.cxx
/// \author julian.myrcha@cern.ch

#include <EveWorkflow/EveWorkflowHelper.h>
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "EveWorkflow/FileProducer.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "ITStracking/IOUtils.h"
#include "MFTTracking/IOUtils.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsBase/Propagator.h"
#include "TPCBase/ParameterElectronics.h"
#include "DataFormatsTPC/Defs.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/TrackExtrap.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

using namespace o2::event_visualisation;

void EveWorkflowHelper::selectTracks(const CalibObjectsConst* calib,
                                     GID::mask_t maskCl, GID::mask_t maskTrk, GID::mask_t maskMatch)
{
  auto creator = [maskTrk, this](auto& trk, GID gid, float time, float) {
    if (!maskTrk[gid.getSource()]) {
      return true;
    }
    if constexpr (isTPCTrack<decltype(trk)>()) { // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      time = trk.getTime0();                     // for TPC we need internal time, not the center of the possible interval
    }
    mTrackSet.trackGID.push_back(gid);
    mTrackSet.trackTime.push_back(time);
    return true;
  };
  this->mRecoCont.createTracksVariadic(creator);
}

void EveWorkflowHelper::draw(int numberOfTracks)
{
  size_t nTracks = mTrackSet.trackGID.size();
  if (numberOfTracks != -1 && numberOfTracks < nTracks) {
    nTracks = numberOfTracks; // less than available
  }
  for (size_t it = 0; it < nTracks; it++) {
    const auto& gid = mTrackSet.trackGID[it];
    auto tim = mTrackSet.trackTime[it];
    // LOG(info) << "EveWorkflowHelper::draw " << gid.asString();
    switch (gid.getSource()) {
      case GID::TPC:
        drawTPC(gid, tim);
        break;
      case GID::MFT:
        drawMFT(gid, tim);
        break;
      case GID::MCH:
        drawMCH(gid, tim);
        break;
      case GID::MID:
        drawMID(gid, tim);
        break;
      case GID::ITS:
        drawITS(gid, tim);
        break;
      case GID::ITSTPC:
        drawITSTPC(gid, tim);
        break;
      case GID::ITSTPCTOF:
        drawITSTPCTOF(gid, tim);
        break;
      case GID::TPCTRD:
        drawTPCTRD(gid, tim);
        break;
      case GID::TPCTOF:
        drawTPCTOF(gid, tim);
        break;
      case GID::TPCTRDTOF:
        drawTPCTRDTOF(gid, tim);
        break;
      case GID::ITSTPCTRD:
        drawITSTPCTRD(gid, tim);
        break;
      case GID::ITSTPCTRDTOF:
        drawITSTPCTRDTOF(gid, tim);
        break;
      default:
        LOG(info) << "Track type " << gid.getSource() << " not handled";
    }
  }
}

void EveWorkflowHelper::save(const std::string& jsonPath, int numberOfFiles,
                             o2::dataformats::GlobalTrackID::mask_t trkMask, o2::dataformats::GlobalTrackID::mask_t clMask,
                             float workflowVersion, o2::header::DataHeader::RunNumberType runNumber, o2::framework::DataProcessingHeader::CreationTime creation)
{
  mEvent.setWorkflowVersion(workflowVersion);
  mEvent.setRunNumber(runNumber);
  std::time_t timeStamp = std::time(nullptr);
  std::string asciiTimeStamp = std::asctime(std::localtime(&timeStamp));
  asciiTimeStamp.pop_back(); // remove trailing \n
  mEvent.setWorkflowParameters(asciiTimeStamp + " t:" + trkMask.to_string() + " c:" + clMask.to_string());

  std::time_t creationTime = creation;
  std::string asciiCreationTime = std::asctime(std::localtime(&creationTime));
  asciiCreationTime.pop_back(); // remove trailing \n
  mEvent.setCollisionTime(asciiCreationTime);

  FileProducer producer(jsonPath, numberOfFiles);
  mEvent.toFile(producer.newFileName());
}

std::vector<PNT> EveWorkflowHelper::getTrackPoints(const o2::track::TrackPar& trc, float minR, float maxR, float maxStep, float minZ, float maxZ)
{
  // adjust minR according to real track start from track starting point
  auto maxR2 = maxR * maxR;
  float rMin = std::sqrt(trc.getX() * trc.getX() + trc.getY() * trc.getY());
  if (rMin > minR) {
    minR = rMin;
  }
  // prepare space points from the track param
  std::vector<PNT> pnts;
  int nSteps = std::max(2, int((maxR - minR) / maxStep));
  const auto prop = o2::base::Propagator::Instance();
  float xMin = trc.getX(), xMax = maxR * maxR - trc.getY() * trc.getY();
  if (xMax > 0) {
    xMax = std::sqrt(xMax);
  }

  float dx = (xMax - xMin) / nSteps;
  auto tp = trc;
  float dxmin = std::abs(xMin - tp.getX()), dxmax = std::abs(xMax - tp.getX());

  if (dxmin > dxmax) { // start from closest end
    std::swap(xMin, xMax);
    dx = -dx;
  }
  if (!prop->propagateTo(tp, xMin, false, 0.99, maxStep, o2::base::PropagatorF::MatCorrType::USEMatCorrNONE)) {
    return pnts;
  }
  auto xyz = tp.getXYZGlo();
  pnts.emplace_back(PNT{xyz.X(), xyz.Y(), xyz.Z()});
  for (int is = 0; is < nSteps; is++) {
    if (!prop->propagateTo(tp, tp.getX() + dx, false, 0.99, 999., o2::base::PropagatorF::MatCorrType::USEMatCorrNONE)) {
      return pnts;
    }
    xyz = tp.getXYZGlo();
    if (xyz.Z() < minZ) {
      return pnts;
    }
    if (xyz.Z() > maxZ) {
      return pnts;
    }
    if (xyz.X() * xyz.X() + xyz.Y() * xyz.Y() > maxR2) {
      return pnts;
    }
    pnts.emplace_back(PNT{xyz.X(), xyz.Y(), xyz.Z()});
  }
  return pnts;
}

void EveWorkflowHelper::addTrackToEvent(const o2::track::TrackParCov& tr, GID gid, float trackTime, float dz, GID::Source source)
{
  if (source == GID::NSources) {
    source = (o2::dataformats::GlobalTrackID::Source)gid.getSource();
  }
  auto vTrack = mEvent.addTrack({.time = trackTime,
                                 .charge = tr.getCharge(),
                                 .PID = tr.getPID(),
                                 .startXYZ = {tr.getX(), tr.getY(), tr.getZ()},
                                 .phi = tr.getPhi(),
                                 .theta = tr.getTheta(),
                                 .eta = tr.getEta(),
                                 .gid = gid.asString(),
                                 .source = source});
  auto pnts = getTrackPoints(tr, minmaxR[source].first, minmaxR[source].second, 4, minmaxZ[source].first, minmaxZ[source].second);

  for (size_t ip = 0; ip < pnts.size(); ip++) {
    vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
  }
}

void EveWorkflowHelper::prepareITSClusters(const o2::itsmft::TopologyDictionary* dict)
{
  const auto& ITSClusterROFRec = mRecoCont.getITSClustersROFRecords();
  const auto& clusITS = mRecoCont.getITSClusters();
  if (clusITS.size() && ITSClusterROFRec.size()) {
    const auto& patterns = mRecoCont.getITSClustersPatterns();
    auto pattIt = patterns.begin();
    mITSClustersArray.reserve(clusITS.size());
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, dict);
  }
}

void EveWorkflowHelper::prepareMFTClusters(const o2::itsmft::TopologyDictionary* dict) // do we also have something as ITS...dict?
{
  const auto& MFTClusterROFRec = this->mRecoCont.getMFTClustersROFRecords();
  const auto& clusMFT = this->mRecoCont.getMFTClusters();
  if (clusMFT.size() && MFTClusterROFRec.size()) {
    const auto& patterns = this->mRecoCont.getMFTClustersPatterns();
    auto pattIt = patterns.begin();
    this->mMFTClustersArray.reserve(clusMFT.size());
    o2::mft::ioutils::convertCompactClusters(clusMFT, pattIt, this->mMFTClustersArray, dict);
  }
}

void EveWorkflowHelper::drawITSTPC(GID gid, float trackTime, GID::Source source)
{
  // LOG(info) << "EveWorkflowHelper::drawITSTPC " << gid;
  const auto& track = mRecoCont.getTPCITSTrack(gid);
  addTrackToEvent(track, gid, trackTime, 0., source);
  drawITSClusters(track.getRefITS(), trackTime);
  drawTPCClusters(track.getRefTPC(), trackTime * mMUS2TPCTimeBins);
}

void EveWorkflowHelper::drawITSTPCTOF(GID gid, float trackTime)
{
  const auto& track = mRecoCont.getITSTPCTOFTrack(gid);
  addTrackToEvent(track, gid, trackTime, 0.);
  drawITSClusters(track.getRefITS(), trackTime);
  drawTPCClusters(track.getRefTPC(), trackTime * mMUS2TPCTimeBins);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawTPCTRD(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawTPCTRD " << gid;
  const auto& tpcTrdTrack = mRecoCont.getTPCTRDTrack<o2::trd::TrackTRD>(gid);
  addTrackToEvent(tpcTrdTrack, gid, trackTime, 0.);
  drawTPCClusters(tpcTrdTrack.getRefGlobalTrackId(), trackTime * mMUS2TPCTimeBins);
  drawTRDClusters(tpcTrdTrack, trackTime);
}

void EveWorkflowHelper::drawITSTPCTRD(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawITSTPCTRD " << gid;
  const auto& itsTpcTrdTrack = mRecoCont.getITSTPCTRDTrack<o2::trd::TrackTRD>(gid);
  drawITSTPC(itsTpcTrdTrack.getRefGlobalTrackId(), trackTime, GID::ITSTPCTRD);
  drawTRDClusters(itsTpcTrdTrack, trackTime);
}

void EveWorkflowHelper::drawITSTPCTRDTOF(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawITSTPCTRDTOF " << gid;
  const auto& match = mRecoCont.getITSTPCTRDTOFMatches()[gid.getIndex()];
  auto gidITSTPCTRD = match.getTrackRef();
  drawITSTPCTRD(gidITSTPCTRD, trackTime);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawTPCTRDTOF(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawTPCTRDTOF " << gid;
  const auto& match = mRecoCont.getTPCTRDTOFMatches()[gid.getIndex()];
  auto gidTPCTRD = match.getTrackRef();
  drawTPCTRD(gidTPCTRD, trackTime);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawTPCTOF(GID gid, float trackTime)
{
  //  LOG(info) << "EveWorkflowHelper::drawTPCTRDTOF " << gid;
  const auto& trTPCTOF = mRecoCont.getTPCTOFTrack(gid);
  const auto& match = mRecoCont.getTPCTOFMatch(gid.getIndex());
  addTrackToEvent(trTPCTOF, gid, trackTime, 0);
  drawTPCClusters(match.getTrackRef(), trackTime * mMUS2TPCTimeBins);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawAOD(EveWorkflowHelper::AODFullTrack const& track, float trackTime)
{
  std::array<float, 5> const arraypar = {track.y(), track.z(), track.snp(),
                                         track.tgl(), track.signed1Pt()};
  std::array<float, 15> const covpar = {track.cYY(), track.cZY(), track.cZZ(),
                                        track.cSnpY(), track.cSnpZ(),
                                        track.cSnpSnp(), track.cTglY(), track.cTglZ(),
                                        track.cTglSnp(), track.cTglTgl(),
                                        track.c1PtY(), track.c1PtZ(), track.c1PtSnp(),
                                        track.c1PtTgl(), track.c1Pt21Pt2()};

  auto const tr = o2::track::TrackParCov(track.x(), track.alpha(), arraypar, covpar);

  addTrackToEvent(tr, GID::ITSTPCTRDTOF, trackTime, 0., GID::ITSTPCTRD);
}

void EveWorkflowHelper::drawTOFClusters(GID gid, float trackTime)
{
  auto tOFClustersArray = mRecoCont.getTOFClusters();
  if (!gid.includesDet(o2::dataformats::GlobalTrackID::Source::TOF)) {
    return;
  }
  const o2::dataformats::MatchInfoTOF& match = mRecoCont.getTOFMatch(gid);
  int tofcl = match.getIdxTOFCl();
  int sector = tOFClustersArray[tofcl].getSector();
  float x = tOFClustersArray[tofcl].getX();
  float y = tOFClustersArray[tofcl].getY();
  float z = tOFClustersArray[tofcl].getZ();

  // rotation
  float alpha = o2::math_utils::sector2Angle(sector);
  float xGlb = x * cos(alpha) - y * sin(alpha);
  float yGlb = y * cos(alpha) + x * sin(alpha);
  float zGlb = z;
  drawPoint(xGlb, yGlb, zGlb, trackTime);
}

void EveWorkflowHelper::drawITSClusters(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawITSClusters" << gid;
  if (gid.getSource() == GID::ITS) { // this is for for full standalone tracks
    // LOG(info) << "EveWorkflowHelper::drawITSClusters ITS " << gid;
    const auto& trc = mRecoCont.getITSTrack(gid);
    auto refs = mRecoCont.getITSTracksClusterRefs();
    int ncl = trc.getNumberOfClusters();
    int offset = trc.getFirstClusterEntry();
    for (int icl = 0; icl < ncl; icl++) {
      const auto& pnt = mITSClustersArray[refs[icl + offset]];
      const auto glo = mITSGeom->getMatrixT2G(pnt.getSensorID()) * pnt.getXYZ();
      drawPoint(glo.X(), glo.Y(), glo.Z(), trackTime);
    }
  } else if (gid.getSource() == GID::ITSAB) { // this is for ITS tracklets from ITS-TPC afterburner
    // LOG(info) << "EveWorkflowHelper::drawITSClusters ITSAB " << gid;
    const auto& trc = mRecoCont.getITSABRef(gid);
    const auto& refs = mRecoCont.getITSABClusterRefs();
    int ncl = trc.getNClusters();
    int offset = trc.getFirstEntry();
    for (int icl = 0; icl < ncl; icl++) {
      const auto& pnt = mITSClustersArray[refs[icl + offset]];
      const auto glo = mITSGeom->getMatrixT2G(pnt.getSensorID()) * pnt.getXYZ();
      drawPoint(glo.X(), glo.Y(), glo.Z(), trackTime);
    }
  }
}

// TPC cluseters for given TPC track (gid)
void EveWorkflowHelper::drawTPCClusters(GID gid, float trackTimeTB)
{
  const auto& trc = mRecoCont.getTPCTrack(gid);
  auto mTPCTracksClusIdx = mRecoCont.getTPCTracksClusterRefs();
  auto mTPCClusterIdxStruct = &mRecoCont.getTPCClusters();

  // store the TPC cluster positions
  for (int iCl = trc.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    const auto& clTPC = trc.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);

    std::array<float, 3> xyz;
    this->mTPCFastTransform->TransformIdeal(sector, row, clTPC.getPad(), clTPC.getTime(), xyz[0], xyz[1], xyz[2], trackTimeTB); // in sector coordinate
    o2::math_utils::rotateZ(xyz, o2::math_utils::sector2Angle(sector % o2::tpc::SECTORSPERSIDE));                               // lab coordinate (global)
    mEvent.addCluster(xyz[0], xyz[1], xyz[2], trackTimeTB / mMUS2TPCTimeBins);
  }
}

void EveWorkflowHelper::drawMFTClusters(GID gid, float trackTime)
{
  const auto& mftTrack = mRecoCont.getMFTTrack(gid);
  auto noOfClusters = mftTrack.getNumberOfPoints();       // number of clusters in MFT Track
  auto offset = mftTrack.getExternalClusterIndexOffset(); // first external cluster index offset:
  auto refs = mRecoCont.getMFTTracksClusterRefs();        // list of references to clusters, offset:offset+no
  for (int icl = noOfClusters - 1; icl > -1; --icl) {
    const auto& thisCluster = mMFTClustersArray[refs[offset + icl]];
    drawPoint(thisCluster.getX(), thisCluster.getY(), thisCluster.getZ(), trackTime);
  }
}

void EveWorkflowHelper::drawTPC(GID gid, float trackTime)
{
  const auto& tr = mRecoCont.getTPCTrack(gid);
  // this is a hack to suppress the noise
  //  if (std::abs(tr.getEta()) < 0.05) {
  //    return;
  //  }
  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime * 8 * o2::constants::lhc::LHCBunchSpacingMUS),
                                 .charge = tr.getCharge(),
                                 .PID = tr.getPID(),
                                 .startXYZ = {tr.getX(), tr.getY(), tr.getZ()},
                                 .phi = tr.getPhi(),
                                 .theta = tr.getTheta(),
                                 .eta = tr.getEta(),
                                 .gid = gid.asString(),
                                 .source = GID::TPC});
  auto source = gid.getSource();
  auto pnts = getTrackPoints(tr, minmaxR[source].first, minmaxR[source].second, 4, minmaxZ[source].first, minmaxZ[source].second);
  float dz = 0.0;
  for (size_t ip = 0; ip < pnts.size(); ip++) {
    vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
  }
  drawTPCClusters(gid, trackTime);
}

void EveWorkflowHelper::drawITS(GID gid, float trackTime)
{
  const auto& tr = mRecoCont.getITSTrack(gid);
  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime * 8 * o2::constants::lhc::LHCBunchSpacingMUS),
                                 .charge = tr.getCharge(),
                                 .PID = tr.getPID(),
                                 .startXYZ = {tr.getX(), tr.getY(), tr.getZ()},
                                 .phi = tr.getPhi(),
                                 .theta = tr.getTheta(),
                                 .eta = tr.getEta(),
                                 .gid = gid.asString(),
                                 .source = GID::ITS});
  auto source = gid.getSource();
  auto pnts = getTrackPoints(tr, minmaxR[source].first, minmaxR[source].second, 1.0, minmaxZ[source].first, minmaxZ[source].second);
  float dz = 0.0;
  for (size_t ip = 0; ip < pnts.size(); ip++) {
    vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
  }
  drawITSClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMFT(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawMFT " << gid;
  auto tr = mRecoCont.getMFTTrack(gid);

  std::vector<float> zPositions = {-40.f, -45.f, -65.f, -85.f}; // Selected z positions to draw the track
  tr.propagateToZlinear(zPositions[0]);                         // Fix the track starting position.

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = (int)tr.getCharge(),
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)tr.getX(), (float)tr.getY(), (float)tr.getZ()},
                                 .phi = (float)tr.getPhi(),
                                 .theta = (float)tr.getTanl(),
                                 .eta = (float)0,
                                 .gid = gid.asString(),
                                 .source = GID::MFT});
  for (auto zPos : zPositions) {
    tr.propagateToZlinear(zPos);
    vTrack->addPolyPoint((float)tr.getX(), (float)tr.getY(), (float)tr.getZ());
  }
  drawMFTClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMCH(GID gid, float trackTime)
{
  //  LOG(info) << "EveWorkflowHelper::drawMCH " << gid;
  const auto& track = mRecoCont.getMCHTrack(gid);

  auto noOfClusters = track.getNClusters();                  // number of clusters in MCH Track
  auto offset = track.getFirstClusterIdx();                  // first external cluster index offset:
  const auto& mchClusters = mRecoCont.getMCHTrackClusters(); // list of references to clusters, offset:offset+no

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = 0,
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)track.getX(), (float)track.getY(), (float)track.getZ()},
                                 .phi = (float)0,
                                 .theta = (float)0,
                                 .eta = (float)0,
                                 .gid = gid.asString(),
                                 .source = GID::MCH});

  for (int icl = noOfClusters - 1; icl > -1; --icl) {
    const auto& cluster = mchClusters[offset + icl];
    vTrack->addPolyPoint(cluster.x, cluster.y, cluster.z);
  }
  drawMCHClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMCHClusters(GID gid, float trackTime)
{
  const auto& mchTrack = mRecoCont.getMCHTrack(gid);
  auto noOfClusters = mchTrack.getNClusters();               // number of clusters in MCH Track
  auto offset = mchTrack.getFirstClusterIdx();               // first external cluster index offset:
  const auto& mchClusters = mRecoCont.getMCHTrackClusters(); // list of references to clusters, offset:offset+no
  for (int icl = noOfClusters - 1; icl > -1; --icl) {
    const auto& cluster = mchClusters[offset + icl];
    drawPoint(cluster.x, cluster.y, cluster.z, trackTime);
  }
}

void EveWorkflowHelper::drawMID(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawMID --------------------------------" << gid;
  const auto& midTrack = mRecoCont.getMIDTrack(gid);         // MID track
  const auto& midClusters = mRecoCont.getMIDTrackClusters(); // MID clusters

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = (int)0,
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)midTrack.getPositionX(), (float)midTrack.getPositionY(), (float)midTrack.getPositionZ()},
                                 .phi = (float)0,
                                 .theta = (float)0,
                                 .eta = (float)0,
                                 .gid = gid.asString(),
                                 .source = GID::MID});

  for (int ich = 0; ich < 4; ++ich) {
    auto icl = midTrack.getClusterMatched(ich);
    if (icl >= 0) {
      auto& cluster = midClusters[icl];
      vTrack->addPolyPoint(cluster.xCoor, cluster.yCoor, cluster.zCoor);
    }
  }
  drawMIDClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMIDClusters(GID gid, float trackTime)
{
  const auto& midTrack = mRecoCont.getMIDTrack(gid);         // MID track
  const auto& midClusters = mRecoCont.getMIDTrackClusters(); // MID clusters

  for (int ich = 0; ich < 4; ++ich) {
    auto icl = midTrack.getClusterMatched(ich);
    if (icl >= 0) {
      auto& cluster = midClusters[icl];
      drawPoint(cluster.xCoor, cluster.yCoor, cluster.zCoor, trackTime);
    }
  }
}

void EveWorkflowHelper::drawTRDClusters(const o2::trd::TrackTRD& tpcTrdTrack, float trackTime)
{
  const auto& tpcTrdTracks = mRecoCont.getTPCTRDTracks<o2::trd::TrackTRD>();
  const auto& tpcTrdTriggerRecords = mRecoCont.getTPCTRDTriggers();
  const auto& itsTpcTrdTracks = mRecoCont.getITSTPCTRDTracks<o2::trd::TrackTRD>();
  const auto& itsTpcTrdTriggerRecords = mRecoCont.getITSTPCTRDTriggers();
  const auto& trdTracklets = mRecoCont.getTRDTracklets();
  const auto& trdCalibratedTracklets = mRecoCont.getTRDCalibratedTracklets();

  for (int iLayer = 0; iLayer < 6; ++iLayer) {
    if (tpcTrdTrack.getTrackletIndex(iLayer) >= 0) {
      // there is a TRD space point in this layer
      const auto& tracklet = trdTracklets[tpcTrdTrack.getTrackletIndex(iLayer)];
      const auto& spacePoint = trdCalibratedTracklets[tpcTrdTrack.getTrackletIndex(iLayer)];
      // get position in sector coordinates from the spacePoint
      float x = spacePoint.getX(), y = spacePoint.getY(), z = spacePoint.getZ();
      // in order to rotate the space points into the global coordinate system we need to know the TRD chamber number
      int iChamber = tracklet.getDetector();
      // with that we can determine the sector and thus the rotation angle alpha
      int sector = iChamber / 30;
      float alpha = o2::math_utils::sector2Angle(sector);
      // now the rotation is simply
      float xGlb = x * cos(alpha) - y * sin(alpha);
      float yGlb = y * cos(alpha) + x * sin(alpha);
      float zGlb = z;
      drawPoint(xGlb, yGlb, zGlb, trackTime);
    }
  }
}

EveWorkflowHelper::EveWorkflowHelper()
{
  o2::mch::TrackExtrap::setField();
  this->mMFTGeom = o2::mft::GeometryTGeo::Instance();
  this->mMFTGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  this->mITSGeom = o2::its::GeometryTGeo::Instance();
  this->mITSGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::L2G));
  this->mTPCFastTransform = (o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  const auto& elParams = o2::tpc::ParameterElectronics::Instance();
  mMUS2TPCTimeBins = 1. / elParams.ZbinWidth;
}
