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
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
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
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonDataFormat/IRFrame.h"
#include "MFTBase/GeometryTGeo.h"
#include "ITSBase/GeometryTGeo.h"
#include "PHOSBase/Geometry.h"
#include <TGeoBBox.h>
#include <tuple>
#include <gsl/span>

using namespace o2::event_visualisation;

struct TrackTimeNode {
  GID trackGID;
  float trackTime;
};

void EveWorkflowHelper::selectTracks(const CalibObjectsConst* calib,
                                     GID::mask_t maskCl, GID::mask_t maskTrk, GID::mask_t maskMatch, bool trackSorting)
{
  std::vector<TrackTimeNode> trackTimeNodes;
  std::vector<Bracket> itsROFBrackets;

  if (mEnabledFilters.test(Filter::ITSROF)) {
    const auto irFrames = getRecoContainer().getIRFramesITS();

    static int BCDiffErrCount = 0;
    constexpr int MAXBCDiffErrCount = 5;

    auto bcDiffToTFTimeMUS = [startIR = getRecoContainer().startIR](const o2::InteractionRecord& ir) {
      auto bcd = ir.differenceInBC(startIR);
      if (uint64_t(bcd) > o2::constants::lhc::LHCMaxBunches * 256 && BCDiffErrCount < MAXBCDiffErrCount) {
        LOGP(alarm, "ATTENTION: wrong bunches diff. {} for current IR {} wrt 1st TF orbit {}", bcd, ir, startIR);
        BCDiffErrCount++;
      }
      return bcd * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
    };

    for (const auto& irFrame : irFrames) {
      itsROFBrackets.emplace_back(bcDiffToTFTimeMUS(irFrame.getMin()), bcDiffToTFTimeMUS(irFrame.getMax()));
    }
  }

  auto correctTrackTime = [this](auto& _tr, float t0, float terr) {
    if constexpr (isTPCTrack<decltype(_tr)>()) {
      // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      t0 *= this->mTPCBin2MUS;
      terr *= this->mTPCBin2MUS;
    } else if constexpr (isITSTrack<decltype(_tr)>()) {
      t0 += 0.5f * this->mITSROFrameLengthMUS;          // ITS time is supplied in \mus as beginning of ROF
      terr *= this->mITSROFrameLengthMUS;               // error is supplied as a half-ROF duration, convert to \mus
    } else if constexpr (isMFTTrack<decltype(_tr)>()) { // Same for MFT
      t0 += 0.5f * this->mMFTROFrameLengthMUS;
      terr *= this->mMFTROFrameLengthMUS;
    } else if constexpr (isGlobalFwdTrack<decltype(_tr)>()) {
      t0 = _tr.getTimeMUS().getTimeStamp();
      terr = _tr.getTimeMUS().getTimeStampError() * mPVParams->nSigmaTimeTrack; // gaussian errors must be scaled by requested n-sigma
    } else {
      terr *= mPVParams->nSigmaTimeTrack; // gaussian errors must be scaled by requested n-sigma
    }
    // for all other tracks the time is in \mus with gaussian error
    terr += mPVParams->timeMarginTrackTime;

    return Bracket{t0 - terr, t0 + terr};
  };

  auto isInsideITSROF = [&itsROFBrackets](const Bracket& br) {
    for (const auto& ir : itsROFBrackets) {
      const auto overlap = ir.getOverlap(br);

      if (overlap.isValid()) {
        return true;
      }
    }

    return false;
  };

  auto creator = [maskTrk, this, &correctTrackTime, &isInsideITSROF, &trackTimeNodes](auto& trk, GID gid, float time, float terr) {
    if (!maskTrk[gid.getSource()]) {
      return true;
    }

    auto bracket = correctTrackTime(trk, time, terr);

    if (mEnabledFilters.test(Filter::TimeBracket) && mTimeBracket.getOverlap(bracket).isInvalid()) {
      return true;
    }

    if (mEnabledFilters.test(Filter::ITSROF) && !isInsideITSROF(bracket)) {
      return true;
    }

    TrackTimeNode node;
    node.trackGID = gid;
    node.trackTime = bracket.mean();
    trackTimeNodes.push_back(node);

    return true;
  };

  this->mRecoCont.createTracksVariadic(creator);

  if (trackSorting) {
    std::sort(trackTimeNodes.begin(), trackTimeNodes.end(),
              [](TrackTimeNode a, TrackTimeNode b) {
                return a.trackTime > b.trackTime;
              });
  }

  std::size_t trackCount = trackTimeNodes.size();
  if (mEnabledFilters.test(Filter::TotalNTracks) && trackCount >= mMaxNTracks) {
    trackCount = mMaxNTracks;
  }

  for (auto node : gsl::span<const TrackTimeNode>(trackTimeNodes.data(), trackCount)) {
    mTrackSet.trackGID.push_back(node.trackGID);
    mTrackSet.trackTime.push_back(node.trackTime);
  }
}

void EveWorkflowHelper::draw()
{
  this->drawPHOS();

  for (size_t it = 0; it < mTrackSet.trackGID.size(); it++) {
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
                             o2::header::DataHeader::RunNumberType runNumber, o2::framework::DataProcessingHeader::CreationTime creation)
{
  mEvent.setWorkflowVersion(o2_eve_version);
  mEvent.setRunNumber(runNumber);
  std::time_t timeStamp = std::time(nullptr);
  std::string asciiTimeStamp = std::asctime(std::localtime(&timeStamp));
  asciiTimeStamp.pop_back(); // remove trailing \n
  mEvent.setWorkflowParameters(asciiTimeStamp + " t:" + trkMask.to_string() + " c:" + clMask.to_string());

  std::time_t creationTime = creation / 1000; // convert to seconds
  std::string asciiCreationTime = std::asctime(std::localtime(&creationTime));
  asciiCreationTime.pop_back(); // remove trailing \n
  mEvent.setCollisionTime(asciiCreationTime);

  FileProducer producer(jsonPath, numberOfFiles);
  VisualisationEventSerializer::getInstance()->toFile(mEvent, producer.newFileName());
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

void EveWorkflowHelper::addTrackToEvent(const o2::track::TrackParCov& tr, GID gid, float trackTime, float dz, GID::Source source, float maxStep)
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
  auto pnts = getTrackPoints(tr, minmaxR[source].first, minmaxR[source].second, maxStep, minmaxZ[source].first, minmaxZ[source].second);

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

void EveWorkflowHelper::drawPHOS()
{
  for (auto phos : mRecoCont.getPHOSCells()) {
    char relativeLocalPositionInModule[3]; // relative (local) position within module
    float x, z;
    o2::phos::Geometry::absToRelNumbering(phos.getAbsId(), relativeLocalPositionInModule);
    o2::phos::Geometry::absIdToRelPosInModule(phos.getAbsId(), x, z);
    TVector3 gPos;

    // convert local position in module to global position in ALICE including actual mis-aslignment read with GetInstance("Run3")
    this->mPHOSGeom->local2Global(relativeLocalPositionInModule[0], x, z, gPos);

    auto vCalo = mEvent.addCalo({.time = static_cast<float>(phos.getTime()),
                                 .energy = phos.getEnergy(),
                                 .phi = (float)gPos.Phi(),
                                 .eta = (float)gPos.Eta(),
                                 .PID = 0,
                                 .gid = GID::getSourceName(GID::PHS),
                                 .source = GID::PHS});
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

void EveWorkflowHelper::drawAODBarrel(EveWorkflowHelper::AODBarrelTrack const& track, float trackTime)
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

  addTrackToEvent(tr, GID{0, detectorMapToGIDSource(track.detectorMap())}, trackTime, 0.);
}

void EveWorkflowHelper::drawAODMFT(AODMFTTrack const& track, float trackTime)
{
  auto tr = o2::track::TrackParFwd();

  tr.setZ(track.z());
  tr.setParameters({track.x(), track.y(), track.phi(), track.tgl(), track.signed1Pt()});

  std::vector<float> zPositions = {-40.f, -45.f, -65.f, -85.f}; // Selected z positions to draw the track
  tr.propagateParamToZlinear(zPositions[0]);                    // Fix the track starting position.

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = (int)tr.getCharge(),
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)tr.getX(), (float)tr.getY(), (float)tr.getZ()},
                                 .phi = (float)tr.getPhi(),
                                 .theta = (float)tr.getTheta(),
                                 .eta = (float)tr.getEta(),
                                 .gid = GID::getSourceName(GID::MFT),
                                 .source = GID::MFT});

  for (auto zPos : zPositions) {
    tr.propagateParamToZlinear(zPos);
    vTrack->addPolyPoint((float)tr.getX(), (float)tr.getY(), (float)tr.getZ());
  }
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
    this->mTPCFastTransform->TransformIdeal(sector, row, clTPC.getPad(), clTPC.getTime(), xyz[0], xyz[1], xyz[2], trc.getTime0()); // in sector coordinate
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

  if (mEnabledFilters.test(Filter::EtaBracket) && mEtaBracket.isOutside(tr.getEta()) == Bracket::Relation::Inside) {
    return;
  }

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
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
  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
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
                                 .theta = (float)tr.getTheta(),
                                 .eta = (float)tr.getEta(),
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

EveWorkflowHelper::EveWorkflowHelper(const FilterSet& enabledFilters, std::size_t maxNTracks, const Bracket& timeBracket, const Bracket& etaBracket) : mEnabledFilters(enabledFilters), mMaxNTracks(maxNTracks), mTimeBracket(timeBracket), mEtaBracket(etaBracket)
{
  o2::mch::TrackExtrap::setField();
  this->mMFTGeom = o2::mft::GeometryTGeo::Instance();
  this->mMFTGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  this->mITSGeom = o2::its::GeometryTGeo::Instance();
  this->mITSGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::L2G));
  this->mPHOSGeom = o2::phos::Geometry::GetInstance("");
  this->mTPCFastTransform = (o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  const auto& elParams = o2::tpc::ParameterElectronics::Instance();
  mMUS2TPCTimeBins = 1. / elParams.ZbinWidth;
  mTPCBin2MUS = elParams.ZbinWidth;

  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};

  const auto& alpParamsITS = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  mITSROFrameLengthMUS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParamsITS.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsITS.roFrameLengthTrig * 1.e-3;

  const auto& alpParamsMFT = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
  mMFTROFrameLengthMUS = grp->isDetContinuousReadOut(o2::detectors::DetID::MFT) ? alpParamsMFT.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsMFT.roFrameLengthTrig * 1.e-3;

  mPVParams = &o2::vertexing::PVertexerParams::Instance();
}

GID::Source EveWorkflowHelper::detectorMapToGIDSource(uint8_t dm)
{
  switch (dm) {
    case static_cast<uint8_t>(o2::aod::track::ITS):
      return GID::ITS;
    case static_cast<uint8_t>(o2::aod::track::TPC):
      return GID::TPC;
    case static_cast<uint8_t>(o2::aod::track::TRD):
      return GID::TRD;
    case static_cast<uint8_t>(o2::aod::track::TOF):
      return GID::TOF;
    case static_cast<uint8_t>(o2::aod::track::ITS) | static_cast<uint8_t>(o2::aod::track::TPC):
      return GID::ITSTPC;
    case static_cast<uint8_t>(o2::aod::track::TPC) | static_cast<uint8_t>(o2::aod::track::TOF):
      return GID::TPCTOF;
    case static_cast<uint8_t>(o2::aod::track::TPC) | static_cast<uint8_t>(o2::aod::track::TRD):
      return GID::TPCTRD;
    case static_cast<uint8_t>(o2::aod::track::ITS) | static_cast<uint8_t>(o2::aod::track::TPC) | static_cast<uint8_t>(o2::aod::track::TRD):
      return GID::ITSTPCTRD;
    case static_cast<uint8_t>(o2::aod::track::ITS) | static_cast<uint8_t>(o2::aod::track::TPC) | static_cast<uint8_t>(o2::aod::track::TOF):
      return GID::ITSTPCTOF;
    case static_cast<uint8_t>(o2::aod::track::TPC) | static_cast<uint8_t>(o2::aod::track::TRD) | static_cast<uint8_t>(o2::aod::track::TOF):
      return GID::TPCTRDTOF;
    default:
      return GID::ITSTPCTRDTOF;
  }
}
