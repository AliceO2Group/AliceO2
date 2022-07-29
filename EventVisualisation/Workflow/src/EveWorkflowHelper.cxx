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
#include "ReconstructionDataFormats/VtxTrackRef.h"
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
#include "EMCALBase/Geometry.h"
#include <TGeoBBox.h>
#include <tuple>
#include <gsl/span>

using namespace o2::event_visualisation;

const std::unordered_map<GID::Source, EveWorkflowHelper::PropagationRange> EveWorkflowHelper::propagationRanges = {
  {GID::ITS, EveWorkflowHelper::prITS},
  {GID::TPC, EveWorkflowHelper::prTPC},
  {GID::ITSTPC, {EveWorkflowHelper::prITS.minR, EveWorkflowHelper::prTPC.maxR, EveWorkflowHelper::prTPC.minZ, EveWorkflowHelper::prTPC.maxZ}},
  {GID::TPCTOF, {EveWorkflowHelper::prTPC.minR, EveWorkflowHelper::prTOF.maxR, EveWorkflowHelper::prTOF.minZ, EveWorkflowHelper::prTOF.maxZ}},
  {GID::TPCTRD, {EveWorkflowHelper::prTPC.minR, EveWorkflowHelper::prTRD.maxR, EveWorkflowHelper::prTRD.minZ, EveWorkflowHelper::prTRD.maxZ}},
  {GID::ITSTPCTRD, {EveWorkflowHelper::prITS.minR, EveWorkflowHelper::prTRD.maxR, EveWorkflowHelper::prTRD.minZ, EveWorkflowHelper::prTRD.maxZ}},
  {GID::ITSTPCTOF, {EveWorkflowHelper::prITS.minR, EveWorkflowHelper::prTOF.maxR, EveWorkflowHelper::prTOF.minZ, EveWorkflowHelper::prTOF.maxZ}},
  {GID::TPCTRDTOF, {EveWorkflowHelper::prTPC.minR, EveWorkflowHelper::prTOF.maxR, EveWorkflowHelper::prTOF.minZ, EveWorkflowHelper::prTOF.maxZ}},
  {GID::ITSTPCTRDTOF, {EveWorkflowHelper::prITS.minR, EveWorkflowHelper::prTOF.maxR, EveWorkflowHelper::prTOF.minZ, EveWorkflowHelper::prTOF.maxZ}},
};

o2::mch::TrackParam EveWorkflowHelper::forwardTrackToMCHTrack(const o2::track::TrackParFwd& track)
{
  const auto phi = track.getPhi();
  const auto sinPhi = std::sin(phi);
  const auto tgL = track.getTgl();

  const auto SlopeX = std::cos(phi) / tgL;
  const auto SlopeY = sinPhi / tgL;
  const auto InvP_yz = track.getInvQPt() / std::sqrt(sinPhi * sinPhi + tgL * tgL);

  const std::array<Double_t, 5> params{track.getX(), SlopeX, track.getY(), SlopeY, InvP_yz};
  const std::array<Double_t, 15> cov{
    1,
    0, 1,
    0, 0, 1,
    0, 0, 0, 1,
    0, 0, 0, 0, 1};

  return {track.getZ(), params.data(), cov.data()};
}

float EveWorkflowHelper::findLastMIDClusterPosition(const o2::mid::Track& track)
{
  const auto& midClusters = mRecoCont.getMIDTrackClusters();

  int icl = -1;

  // Find last cluster position
  for (std::size_t ich = 0; ich < 4; ++ich) {
    auto cur_icl = track.getClusterMatched(ich);

    if (cur_icl >= 0) {
      icl = cur_icl;
    }
  }

  if (icl >= 0) {
    const auto& cluster = midClusters[icl];
    return cluster.zCoor;
  } else {
    return track.getPositionZ();
  }
}

float EveWorkflowHelper::findLastMCHClusterPosition(const o2::mch::TrackMCH& track)
{
  const auto& mchClusters = mRecoCont.getMCHTrackClusters();

  auto noOfClusters = track.getNClusters();
  auto offset = track.getFirstClusterIdx();

  const auto& lastCluster = mchClusters[offset + noOfClusters - 1];

  return lastCluster.getZ();
}

void EveWorkflowHelper::selectTracks(const CalibObjectsConst* calib,
                                     GID::mask_t maskCl, GID::mask_t maskTrk, GID::mask_t maskMatch)
{
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
    } else if constexpr (!(isMCHTrack<decltype(_tr)>() || isMIDTrack<decltype(_tr)>() || isGlobalFwdTrack<decltype(_tr)>())) {
      // for all other tracks the time is in \mus with gaussian error
      terr *= mPVParams->nSigmaTimeTrack; // gaussian errors must be scaled by requested n-sigma
    }

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

  auto flagTime = [](float time, GID::Src_t src) {
    auto flag = static_cast<int>(time) + TIME_OFFSET;

    // if it's a tracklet, give it a lower priority in time sort by setting the highest bit
    if (src <= GID::MID) {
      flag |= (1 << 31);
    }

    return flag;
  };

  auto creator = [maskTrk, this, &correctTrackTime, &isInsideITSROF, &flagTime](auto& trk, GID gid, float time, float terr) {
    const auto src = gid.getSource();
    mTotalTracks[src]++;

    if (!maskTrk[src]) {
      return true;
    }

    auto bracket = correctTrackTime(trk, time, terr);

    if (mEnabledFilters.test(Filter::TimeBracket) && mTimeBracket.getOverlap(bracket).isInvalid()) {
      return true;
    }

    if (mEnabledFilters.test(Filter::ITSROF) && !isInsideITSROF(bracket)) {
      return true;
    }

    mGIDTrackTime[gid] = flagTime(bracket.mean(), src);

    // If the mode is disabled,
    // add every track to a symbolic "zero" primary vertex
    if (!mPrimaryVertexMode) {
      mPrimaryVertexGIDs[0].push_back(gid);
    }

    return true;
  };

  this->mRecoCont.createTracksVariadic(creator);

  if (mPrimaryVertexMode) {
    const auto trackIndex = mRecoCont.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
    const auto vtxRefs = mRecoCont.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
    const auto totalPrimaryVertices = vtxRefs.size() - 1;              // The last entry is for unassigned tracks, ignore them

    for (std::size_t iv = 0; iv < totalPrimaryVertices; iv++) {
      const auto& vtref = vtxRefs[iv];
      int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
      for (; it < itLim; it++) {
        const auto tvid = trackIndex[it];

        if (!mRecoCont.isTrackSourceLoaded(tvid.getSource())) {
          continue;
        }

        // TODO: fix TPC tracks?

        // If a track was not rejected, associate it with its primary vertex
        if (mGIDTrackTime.find(tvid) != mGIDTrackTime.end()) {
          mPrimaryVertexGIDs[iv].push_back(tvid);
        }
      }
    }
  }
}

void EveWorkflowHelper::draw(std::size_t primaryVertexIdx, bool sortTracks)
{
  auto unflagTime = [](unsigned int time) {
    return static_cast<float>(static_cast<int>(time & ~(1 << 31)) - TIME_OFFSET);
  };

  auto& tracks = mPrimaryVertexGIDs.at(primaryVertexIdx);

  if (sortTracks) {
    std::sort(tracks.begin(), tracks.end(),
              [&](const GID& a, const GID& b) {
                return mGIDTrackTime.at(a) < mGIDTrackTime.at(b);
              });
  }

  this->drawPHOS();
  this->drawEMCAL();

  auto trackCount = tracks.size();

  if (mEnabledFilters.test(Filter::TotalNTracks) && trackCount >= mMaxNTracks) {
    trackCount = mMaxNTracks;
  }

  for (size_t it = 0; it < trackCount; it++) {
    const auto& gid = tracks[it];
    auto tim = unflagTime(mGIDTrackTime.at(gid));
    mTotalAcceptedTracks.insert(gid);
    // LOG(info) << "EveWorkflowHelper::draw " << gid.asString();
    switch (gid.getSource()) {
      case GID::ITS:
        drawITS(gid, tim);
        break;
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
      case GID::ITSTPC:
        drawITSTPC(gid, tim);
        break;
      case GID::TPCTOF:
        drawTPCTOF(gid, tim);
        break;
      case GID::TPCTRD:
        drawTPCTRD(gid, tim);
        break;
      case GID::MFTMCH:
        drawMFTMCH(gid, tim);
        break;
      case GID::MFTMCHMID:
        drawMFTMCHMID(gid, tim);
        break;
      case GID::ITSTPCTRD:
        drawITSTPCTRD(gid, tim);
        break;
      case GID::ITSTPCTOF:
        drawITSTPCTOF(gid, tim);
        break;
      case GID::TPCTRDTOF:
        drawTPCTRDTOF(gid, tim);
        break;
      case GID::ITSTPCTRDTOF:
        drawITSTPCTRDTOF(gid, tim);
        break;
      case GID::MCHMID:
        drawMCHMID(gid, tim);
        break;
      default:
        LOG(info) << "Track type " << gid.getSource() << " not handled";
    }
  }
}

void EveWorkflowHelper::save(const std::string& jsonPath, const std::string& ext, int numberOfFiles,
                             o2::dataformats::GlobalTrackID::mask_t trkMask, o2::dataformats::GlobalTrackID::mask_t clMask,
                             o2::header::DataHeader::RunNumberType runNumber, o2::framework::DataProcessingHeader::CreationTime creation)
{
  mEvent.setEveVersion(o2_eve_version);
  mEvent.setRunNumber(runNumber);
  std::time_t timeStamp = std::time(nullptr);
  std::string asciiTimeStamp = std::asctime(std::localtime(&timeStamp));
  asciiTimeStamp.pop_back(); // remove trailing \n
  mEvent.setWorkflowParameters(asciiTimeStamp + " t:" + trkMask.to_string() + " c:" + clMask.to_string());

  std::time_t creationTime = creation / 1000; // convert to seconds
  std::string asciiCreationTime = std::asctime(std::localtime(&creationTime));
  asciiCreationTime.pop_back(); // remove trailing \n
  mEvent.setCollisionTime(asciiCreationTime);

  FileProducer producer(jsonPath, ext, numberOfFiles);
  VisualisationEventSerializer::getInstance(ext)->toFile(mEvent, producer.newFileName());
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

void EveWorkflowHelper::addTrackToEvent(const o2::track::TrackPar& tr, GID gid, float trackTime, float dz, GID::Source source, float maxStep)
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

  const auto it = propagationRanges.find(source);

  const bool rangeNotFound = (it == propagationRanges.cend());
  if (rangeNotFound) {
    LOGF(error, "Track source %s has no defined propagation ranges", GID::getSourceName(source));
    return;
  }

  const auto& prange = it->second;

  auto pnts = getTrackPoints(tr, prange.minR, prange.maxR, maxStep, prange.minZ, prange.maxZ);

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

void EveWorkflowHelper::drawEMCAL()
{
  // LOG(info) <<  mRecoCont.getEMCALCells().size() << "----------------------------------------------------------------------------- !mRecoCont.getEMCALCells().size()";
  auto triggers = mRecoCont.getEMCALTriggers();
  for (auto trigger : triggers) {
    // trigger.getBCData().toLong() // timestamp in ns
    // trigger.getFirstEntry()
    // trigger.getFirstEntry() range in the emcal cell vector in RecoContainer
    //   this will be towers belongs to BC (the same for emcal/phos) ++mattermost
  }

  for (auto emcal : mRecoCont.getEMCALCells()) {
    int id = emcal.getTower();
    // supermodule ID, module number, index of cell in module in phi, index of cell in module in eta
    auto index = this->mEMCALGeom->GetCellIndex(id);
    // Point3D with x,y,z coordinates of cell with absId inside SM
    auto relPosCell = this->mEMCALGeom->RelPosCellInSModule(id);
    TGeoNode* node = gGeoManager->GetTopVolume()->FindNode("XEN1");
    auto nSupermodules = this->mEMCALGeom->GetNumberOfSuperModules();
    auto fPhiTileSize = this->mEMCALGeom->GetPhiTileSize();
    auto fEtaTileSize = this->mEMCALGeom->GetEtaTileSize();
    auto sm = std::get<0>(index);
    auto module_number = std::get<1>(index);
    auto index_module_phi = std::get<2>(index);
    auto index_module_eta = std::get<3>(index);

    const TGeoHMatrix* matrix = this->mEMCALGeom->GetMatrixForSuperModuleFromGeoManager(sm);
    const Double_t* translation = matrix->GetTranslation();
    const Double_t* rotation = matrix->GetRotationMatrix();

    /*
    LOG(info) << "EMCAL -----------------------------------------------------------------------------------------------";
    LOG(info) << "EMCAL               id: "<< id  <<            "                  emcal.getTower()";
    LOG(info) << "EMCAL       relPosCell: "<< relPosCell  <<    "                  this->mEMCALGeom->RelPosCellInSModule(id);";
    LOG(info) << "EMCAL    nSupermodules: "<< nSupermodules  << "                  this->mEMCALGeom->GetNumberOfSuperModules()";
    LOG(info) << "EMCAL     fPhiTileSize: "<< fPhiTileSize <<   "                  this->mEMCALGeom->GetPhiTileSize()";
    LOG(info) << "EMCAL     fEtaTileSize: "<< fEtaTileSize <<   "                  this->mEMCALGeom->GetEtaTileSize();" ;
    LOG(info) << "EMCAL             node: "<< node ;
    LOG(info) << "EMCAL               sm: "<< sm ;
    LOG(info) << "EMCAL    module_number: "<< module_number ;
    LOG(info) << "EMCAL index_module_phi: "<< index_module_phi ;
    LOG(info) << "EMCAL index_module_eta: "<< index_module_eta ;
    LOG(info) << "EMCAL      translation: "<< "["<<translation[0]<<","<<translation[1]<<","<<translation[2]<<"]"  ;
    */
    double rPos[] = {relPosCell.X(), relPosCell.Y(), relPosCell.Z()};
    double gPos[3];
    matrix->LocalToMaster(rPos, gPos);
    TVector3 vPos(gPos);

    auto vCalo = mEvent.addCalo({.time = static_cast<float>(emcal.getTimeStamp()),
                                 .energy = emcal.getEnergy(),
                                 .phi = (float)vPos.Phi(),
                                 .eta = (float)vPos.Eta(),
                                 .PID = 0,
                                 .gid = GID::getSourceName(GID::EMC),
                                 .source = GID::EMC});
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

void EveWorkflowHelper::drawITSTPCTOF(GID gid, float trackTime, GID::Source source)
{
  const auto& track = mRecoCont.getITSTPCTOFTrack(gid);
  addTrackToEvent(track, gid, trackTime, 0., source);
  drawITSClusters(track.getRefITS(), trackTime);
  drawTPCClusters(track.getRefTPC(), trackTime * mMUS2TPCTimeBins);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawTPCTRD(GID gid, float trackTime, GID::Source source)
{
  // LOG(info) << "EveWorkflowHelper::drawTPCTRD " << gid;
  const auto& tpcTrdTrack = mRecoCont.getTPCTRDTrack<o2::trd::TrackTRD>(gid);
  addTrackToEvent(tpcTrdTrack, gid, trackTime, 0., source);
  drawTPCClusters(tpcTrdTrack.getRefGlobalTrackId(), trackTime * mMUS2TPCTimeBins);
  drawTRDClusters(tpcTrdTrack, trackTime);
}

void EveWorkflowHelper::drawITSTPCTRD(GID gid, float trackTime, GID::Source source)
{
  // LOG(info) << "EveWorkflowHelper::drawITSTPCTRD " << gid;
  const auto& itsTpcTrdTrack = mRecoCont.getITSTPCTRDTrack<o2::trd::TrackTRD>(gid);
  drawITSTPC(itsTpcTrdTrack.getRefGlobalTrackId(), trackTime, source);
  drawTRDClusters(itsTpcTrdTrack, trackTime);
}

void EveWorkflowHelper::drawITSTPCTRDTOF(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawITSTPCTRDTOF " << gid;
  const auto& match = mRecoCont.getITSTPCTRDTOFMatches()[gid.getIndex()];
  auto gidITSTPCTRD = match.getTrackRef();
  drawITSTPCTRD(gidITSTPCTRD, trackTime, GID::ITSTPCTRDTOF);
  drawTOFClusters(gid, trackTime);
}

void EveWorkflowHelper::drawTPCTRDTOF(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawTPCTRDTOF " << gid;
  const auto& match = mRecoCont.getTPCTRDTOFMatches()[gid.getIndex()];
  auto gidTPCTRD = match.getTrackRef();
  drawTPCTRD(gidTPCTRD, trackTime, GID::TPCTRDTOF);
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

void EveWorkflowHelper::drawMFTMCH(GID gid, float trackTime)
{
  const auto& trMFTMCH = mRecoCont.getGlobalFwdTrack(gid);

  const auto& trackParam = forwardTrackToMCHTrack(trMFTMCH);

  const auto mchGID = GID{static_cast<unsigned int>(trMFTMCH.getMCHTrackID()), GID::MCH};

  const auto& mchTrack = mRecoCont.getMCHTrack(mchGID);

  const auto endZ = findLastMCHClusterPosition(mchTrack);

  drawForwardTrack(trackParam, gid.asString(), static_cast<GID::Source>(gid.getSource()), mftZPositions.front(), endZ, trackTime);

  drawMFTClusters(GID{static_cast<unsigned int>(trMFTMCH.getMFTTrackID()), GID::MFT}, trackTime);
  drawMCHClusters(mchGID, trackTime);
}

void EveWorkflowHelper::drawMFTMCHMID(GID gid, float trackTime)
{
  const auto& trMFTMCHMID = mRecoCont.getGlobalFwdTrack(gid);

  const auto& trackParam = forwardTrackToMCHTrack(trMFTMCHMID);

  const auto midGID = GID{static_cast<unsigned int>(trMFTMCHMID.getMIDTrackID()), GID::MID};

  const auto& midTrack = mRecoCont.getMIDTrack(midGID);

  const auto endZ = findLastMIDClusterPosition(midTrack);

  drawForwardTrack(trackParam, gid.asString(), static_cast<GID::Source>(gid.getSource()), mftZPositions.front(), endZ, trackTime);

  drawMFTClusters(GID{static_cast<unsigned int>(trMFTMCHMID.getMFTTrackID()), GID::MFT}, trackTime);
  drawMCHClusters(GID{static_cast<unsigned int>(trMFTMCHMID.getMCHTrackID()), GID::MCH}, trackTime);
  drawMIDClusters(midGID, trackTime);
}

void EveWorkflowHelper::drawMCHMID(GID gid, float trackTime)
{
  const auto& match = mRecoCont.getMCHMIDMatches()[gid.getIndex()];
  const auto& mchTrack = mRecoCont.getMCHTrack(match.getMCHRef());
  const auto& midTrack = mRecoCont.getMIDTrack(match.getMIDRef());

  auto trackParam = mch::TrackParam(mchTrack.getZ(), mchTrack.getParameters(), mchTrack.getCovariances());

  const auto endZ = findLastMIDClusterPosition(midTrack);

  drawForwardTrack(trackParam, gid.asString(), static_cast<GID::Source>(gid.getSource()), trackParam.getZ(), endZ, trackTime);

  drawMCHClusters(match.getMCHRef(), trackTime);
  drawMIDClusters(match.getMIDRef(), trackTime);
}

void EveWorkflowHelper::drawAODBarrel(EveWorkflowHelper::AODBarrelTrack const& track, float trackTime)
{
  const std::array<float, 5> arraypar = {track.y(), track.z(), track.snp(),
                                         track.tgl(), track.signed1Pt()};

  const auto tr = o2::track::TrackPar(track.x(), track.alpha(), arraypar);

  addTrackToEvent(tr, GID{0, detectorMapToGIDSource(track.detectorMap())}, trackTime, 0.);
}

void EveWorkflowHelper::drawAODMFT(AODMFTTrack const& track, float trackTime)
{
  auto tr = o2::track::TrackParFwd();

  tr.setZ(track.z());
  tr.setParameters({track.x(), track.y(), track.phi(), track.tgl(), track.signed1Pt()});

  drawMFTTrack(tr, trackTime);
}

void EveWorkflowHelper::drawAODFwd(AODForwardTrack const& track, float trackTime)
{
  o2::track::TrackParFwd trackFwd;
  trackFwd.setZ(track.z());
  trackFwd.setParameters({track.x(), track.y(), track.phi(), track.tgl(), track.signed1Pt()});

  const auto trackParam = forwardTrackToMCHTrack(trackFwd);

  float endZ = 0;
  GID gid;

  switch (track.trackType()) {
    case o2::aod::fwdtrack::GlobalMuonTrack:
    case o2::aod::fwdtrack::GlobalMuonTrackOtherMatch:
      gid = GID::MFTMCHMID;
      endZ = midZPositions.back();
      break;
    case o2::aod::fwdtrack::GlobalForwardTrack:
      gid = GID::MFTMCH;
      endZ = mchZPositions.back();
      break;
    case o2::aod::fwdtrack::MuonStandaloneTrack:
      gid = GID::MCHMID;
      endZ = midZPositions.back();
      break;
    case o2::aod::fwdtrack::MCHStandaloneTrack:
      gid = GID::MCH;
      endZ = mchZPositions.back();
      break;
  }

  drawForwardTrack(trackParam, gid.asString(), static_cast<GID::Source>(gid.getSource()), trackParam.getZ(), endZ, trackTime);
}

void EveWorkflowHelper::drawMFTTrack(o2::track::TrackParFwd tr, float trackTime)
{
  tr.propagateParamToZlinear(mftZPositions.front()); // Fix the track starting position.

  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = (int)tr.getCharge(),
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)tr.getX(), (float)tr.getY(), (float)tr.getZ()},
                                 .phi = (float)tr.getPhi(),
                                 .theta = (float)tr.getTheta(),
                                 .eta = (float)tr.getEta(),
                                 .gid = GID::getSourceName(GID::MFT),
                                 .source = GID::MFT});

  for (auto zPos : mftZPositions) {
    tr.propagateParamToZlinear(zPos);
    vTrack->addPolyPoint((float)tr.getX(), (float)tr.getY(), (float)tr.getZ());
  }
}

void EveWorkflowHelper::drawForwardTrack(mch::TrackParam track, const std::string& gidString, GID::Source source, float startZ, float endZ, float trackTime)
{
  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime),
                                 .charge = 0,
                                 .PID = o2::track::PID::Muon,
                                 .startXYZ = {(float)track.getNonBendingCoor(), (float)track.getBendingCoor(), (float)track.getZ()},
                                 .phi = (float)0,
                                 .theta = (float)0,
                                 .eta = (float)0,
                                 .gid = gidString,
                                 .source = source});

  static constexpr auto stepDensity = 50.; // one vertex per 50 cm should be sufficiently dense

  const auto nSteps = static_cast<std::size_t>(std::abs(endZ - startZ) / stepDensity);

  const auto dZ = (endZ - startZ) / nSteps;

  for (std::size_t i = 0; i < nSteps; ++i) {
    const auto z = startZ + i * dZ;
    vTrack->addPolyPoint(track.getNonBendingCoor(), track.getBendingCoor(), z);
    mch::TrackExtrap::extrapToZCov(track, z);
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
    o2::math_utils::rotateZ(xyz, o2::math_utils::sector2Angle(sector % o2::tpc::SECTORSPERSIDE));                                  // lab coordinate (global)
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

  addTrackToEvent(tr, gid, trackTime, 4.f, GID::TPC);

  drawTPCClusters(gid, trackTime);
}

void EveWorkflowHelper::drawITS(GID gid, float trackTime)
{
  const auto& tr = mRecoCont.getITSTrack(gid);
  addTrackToEvent(tr, gid, trackTime, 1.f, GID::ITS);

  drawITSClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMFT(GID gid, float trackTime)
{
  // LOG(info) << "EveWorkflowHelper::drawMFT " << gid;
  auto tr = mRecoCont.getMFTTrack(gid);

  drawMFTTrack(tr, trackTime);
  drawMFTClusters(gid, trackTime);
}

void EveWorkflowHelper::drawMCH(GID gid, float trackTime)
{
  //  LOG(info) << "EveWorkflowHelper::drawMCH " << gid;
  const auto& track = mRecoCont.getMCHTrack(gid);
  auto trackParam = mch::TrackParam(track.getZ(), track.getParameters(), track.getCovariances());

  const auto endZ = findLastMCHClusterPosition(track);

  drawForwardTrack(trackParam, gid.asString(), GID::MCH, track.getZ(), endZ, trackTime);

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

EveWorkflowHelper::EveWorkflowHelper(const FilterSet& enabledFilters, std::size_t maxNTracks, const Bracket& timeBracket, const Bracket& etaBracket, bool primaryVertexMode) : mEnabledFilters(enabledFilters), mMaxNTracks(maxNTracks), mTimeBracket(timeBracket), mEtaBracket(etaBracket), mPrimaryVertexMode(primaryVertexMode)
{
  o2::mch::TrackExtrap::setField();
  this->mMFTGeom = o2::mft::GeometryTGeo::Instance();
  this->mMFTGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  this->mITSGeom = o2::its::GeometryTGeo::Instance();
  this->mITSGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::L2G));
  this->mEMCALGeom = o2::emcal::Geometry::GetInstance("");
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

  for (int i = 0; i < GID::Source::NSources; i++) {
    mTotalTracks[i] = 0;
  }
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
