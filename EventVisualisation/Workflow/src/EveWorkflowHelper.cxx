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
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/Propagator.h"
#include <type_traits>

using namespace o2::event_visualisation;

void EveWorkflowHelper::selectTracks(const CalibObjectsConst* calib,
                                     GID::mask_t maskCl, GID::mask_t maskTrk, GID::mask_t maskMatch)
{
  auto creator = [maskTrk, this](auto& trk, GID gid, float time, float) {
    if (!maskTrk[gid.getSource()]) {
      return true;
    }
    mTrackSet.trackGID.push_back(gid);
    mTrackSet.trackTime.push_back(time);
    return true;
  };
  this->mRecoCont.createTracksVariadic(creator);
}

void EveWorkflowHelper::draw(std::string jsonPath, int numberOfFiles, int numberOfTracks)
{
  EveWorkflowHelper::prepareITSClusters();

  size_t nTracks = mTrackSet.trackGID.size();
  if (numberOfTracks != -1 && numberOfTracks < nTracks) {
    nTracks = numberOfTracks; // less than available
  }
  for (size_t it = 0; it < nTracks; it++) {
    const auto& gid = mTrackSet.trackGID[it];
    auto tim = mTrackSet.trackTime[it];

    if (gid.getSource() == GID::TPC) {
      drawTPC(gid, tim);
    }
    if (gid.getSource() == GID::ITS) {
      drawITS(gid, tim);
    }
    if (gid.getSource() == GID::ITSTPC) {
      drawITSTPC(gid, tim);
    } else if (gid.getSource() == GID::ITSTPCTOF) {
      drawITSTPCTOF(gid, tim);
    }
  }
  FileProducer producer(jsonPath, numberOfFiles);
  mEvent.toFile(producer.newFileName());
}

void EveWorkflowHelper::drawITSTPC(GID gid, float trackTime)
{
  const auto& track = mRecoCont.getTPCITSTrack(gid);
  auto pnts = getTrackPoints(track, minmaxR[gid.getSource()].first, minmaxR[gid.getSource()].second, 4);
  addTrackToEvent([this, trackTime](GID gid) { return mRecoCont.getTPCITSTrack(gid); }, trackTime, 0.);
  GID gidTPC = track.getRefTPC();
  GID gidITS = track.getRefITS();
  drawITSClusters(gidITS, trackTime);
  drawTPCClusters(gidTPC, trackTime);
}

void EveWorkflowHelper::drawITSTPCTOF(GID gid, float trackTime)
{
  const auto& track = mRecoCont.getITSTPCTOFTrack(gid);
  addTrackToEvent([this, trackTime](GID gid) { return mRecoCont.getITSTPCTOFTrack(gid); }, trackTime, 0.);
  GID gidTPC = track.getRefTPC();
  GID gidITS = track.getRefITS();

  drawITSClusters(gidITS, trackTime);
  drawTPCClusters(gidTPC, trackTime);
}

std::vector<PNT> EveWorkflowHelper::getTrackPoints(const o2::track::TrackPar& trc, float minR, float maxR, float maxStep)
{
  // adjust minR according to real track start fro track starting point
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
  LOG(INFO) << "R: " << minR << " " << maxR << " || X: " << xMin << " " << xMax;
  float dx = (xMax - xMin) / nSteps;
  auto tp = trc;
  float dxmin = std::abs(xMin - tp.getX()), dxmax = std::abs(xMax - tp.getX());
  bool res = false;
  if (dxmin > dxmax) { //start from closest end
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
    pnts.emplace_back(PNT{xyz.X(), xyz.Y(), xyz.Z()});
  }
  return pnts;
}

void EveWorkflowHelper::drawPoint(o2::BaseCluster<float> pnt)
{
  mEvent.addCluster(pnt.getX(), pnt.getY(), pnt.getZ());
}

void EveWorkflowHelper::prepareITSClusters(std::string dictfile)
{
  o2::itsmft::TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", "bin");
    dict.readBinaryFile(dictfile);
  }
  const auto& ITSClusterROFRec = mRecoCont.getITSClustersROFRecords();
  const auto& clusITS = mRecoCont.getITSClusters();
  if (clusITS.size() && ITSClusterROFRec.size()) {
    const auto& patterns = mRecoCont.getITSClustersPatterns();
    auto pattIt = patterns.begin();
    mITSClustersArray.reserve(clusITS.size());
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, dict);
  }
}

void EveWorkflowHelper::drawITSClusters(GID gid, float trackTime)
{

  const auto& trc = mRecoCont.getITSTrack(gid);
  auto refs = mRecoCont.getITSTracksClusterRefs();
  int entry0 = trc.getClusterEntry(gid.getIndex()); // correct?
  int ncl = trc.getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    const auto& pnt = mITSClustersArray[refs[icl]];
    drawPoint(pnt);
  }
}

void EveWorkflowHelper::drawTPCClusters(GID gid, float trackTime)
{
  const auto& trc = mRecoCont.getTPCTrack(gid);
  auto refs = mRecoCont.getTPCTracksClusterRefs();

  /*
    int entry0 = trc.getClusterEntry(gid.getIndex());                 // correct?
    int ncl = trc.getNumberOfClusters();
    for (int icl=0;icl<ncl;icl++) {
        const auto& pnt = mITSClustersArray[ refs[icl] ];
        drawPoint(pnt);
    }
     */
}

void EveWorkflowHelper::drawTPC(GID gid, float trackTime)
{
  const auto& tr = mRecoCont.getTPCTrack(gid);
  auto vTrack = mEvent.addTrack({.time = static_cast<float>(trackTime * 8 * o2::constants::lhc::LHCBunchSpacingMUS),
                                 .charge = tr.getCharge(),
                                 .PID = tr.getPID(),
                                 .startXYZ = {tr.getX(), tr.getY(), tr.getZ()},
                                 .phi = tr.getPhi(),
                                 .theta = tr.getTheta(),
                                 .source = GID::TPC});
  auto pnts = getTrackPoints(tr, minmaxR[gid.getSource()].first, minmaxR[gid.getSource()].second, 4);
  float dz = 0.0;
  for (size_t ip = 0; ip < pnts.size(); ip++) {
    vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
  }
}

void EveWorkflowHelper::drawITS(GID gid, float trackTime)
{
  addTrackToEvent([this, trackTime](GID gid) { return mRecoCont.getITSTrack(gid); }, trackTime, 0.);
}

template <typename Functor>
void EveWorkflowHelper::addTrackToEvent(Functor source, GID gid, float trackTime, float dz)
{
  const auto& tr = source(gid);

  auto vTrack = mEvent.addTrack({.time = trackTime,
                                 .charge = tr.getCharge(),
                                 .PID = tr.getPID(),
                                 .startXYZ = {tr.getX(), tr.getY(), tr.getZ()},
                                 .phi = tr.getPhi(),
                                 .theta = tr.getTheta(),
                                 .source = (o2::dataformats::GlobalTrackID::Source)gid.getSource()});
  auto pnts = getTrackPoints(tr, minmaxR[gid.getSource()].first, minmaxR[gid.getSource()].second, 4);

  for (size_t ip = 0; ip < pnts.size(); ip++) {
    vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
  }
}
