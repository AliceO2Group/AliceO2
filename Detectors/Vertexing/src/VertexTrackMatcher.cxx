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

/// \file VertexTrackMatcher.cxx
/// \brief Class for vertex track association
/// \author ruben.shahoyan@cern.ch

#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsVertexing/VertexTrackMatcher.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include <unordered_map>
#include <numeric>

using namespace o2::vertexing;

//___________________________________________________________________
void VertexTrackMatcher::init()
{
  mPVParams = &o2::vertexing::PVertexerParams::Instance();
  updateTimeDependentParams(); // RS FIXME eventually should be set from CCDB for every TF
}

void VertexTrackMatcher::updateTimeDependentParams()
{
  // tpc time bin in microseconds
  if (mMaxTPCDriftTimeMUS == 0) {
    auto& gasParam = o2::tpc::ParameterGas::Instance();
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    auto& detParam = o2::tpc::ParameterDetector::Instance();
    mTPCBin2MUS = elParam.ZbinWidth;
    mMaxTPCDriftTimeMUS = detParam.TPClength / gasParam.DriftV;
  }
  if (mITSROFrameLengthMUS == 0) {
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mITSROFrameLengthMUS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParams.roFrameLengthTrig * 1.e-3;
    LOG(INFO) << "VertexTrackMatcher::ITSROFrameLengthMUS = " << mITSROFrameLengthMUS;
  }
  if (mMFTROFrameLengthMUS == 0) {
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    mMFTROFrameLengthMUS = grp->isDetContinuousReadOut(o2::detectors::DetID::MFT) ? alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParams.roFrameLengthTrig * 1.e-3;
    LOG(INFO) << "VertexTrackMatcher::MFTROFrameLengthMUS = " << mMFTROFrameLengthMUS;
  }
}

void VertexTrackMatcher::process(const o2::globaltracking::RecoContainer& recoData,
                                 std::vector<VTIndex>& trackIndex,
                                 std::vector<VRef>& vtxRefs)
{
  updateTimeDependentParams();

  auto vertices = recoData.getPrimaryVertices();
  auto v2tfitIDs = recoData.getPrimaryVertexContributors();
  auto v2tfitRefs = recoData.getPrimaryVertexContributorsRefs();

  int nv = vertices.size(), nv1 = nv + 1;
  TmpMap tmpMap(nv1);
  auto& orphans = tmpMap.back(); // in the last element we store unassigned track indices

  // register vertex contributors
  std::unordered_map<GIndex, bool> vcont;
  std::vector<VtxTBracket> vtxOrdBrack; // vertex indices and brackets sorted in tmin
  float maxVtxSpan = 0;
  for (int iv = 0; iv < nv; iv++) {
    int idMin = v2tfitRefs[iv].getFirstEntry(), idMax = idMin + v2tfitRefs[iv].getEntries();
    auto& vtxIds = tmpMap[iv]; // global IDs of contibuting tracks
    vtxIds.reserve(v2tfitRefs[iv].getEntries());
    for (int id = idMin; id < idMax; id++) {
      auto gid = v2tfitIDs[id];
      vtxIds.emplace_back(gid).setPVContributor();
      vcont[gid] = true;
    }
    const auto& vtx = vertices[iv];
    const auto& vto = vtxOrdBrack.emplace_back(VtxTBracket{
      {float((vtx.getIRMin().differenceInBC(recoData.startIR) - 0.5f) * o2::constants::lhc::LHCBunchSpacingMUS),
       float((vtx.getIRMax().differenceInBC(recoData.startIR) + 0.5f) * o2::constants::lhc::LHCBunchSpacingMUS)},
      iv});
    if (vto.tBracket.delta() > maxVtxSpan) {
      maxVtxSpan = vto.tBracket.delta();
    }
  }
  // sort vertices in tmin
  std::sort(vtxOrdBrack.begin(), vtxOrdBrack.end(), [](const VtxTBracket& a, const VtxTBracket& b) { return a.tBracket.getMin() < b.tBracket.getMin(); });

  extractTracks(recoData, vcont); // extract all track t-brackets, excluding those tracks which contribute to vertex (already attached)

  int ivStart = 0, nAssigned = 0, nAmbiguous = 0;
  std::vector<int> vtxList; // list of vertices which match to checked track
  for (const auto& tro : mTBrackets) {
    vtxList.clear();
    for (int iv = ivStart; iv < nv; iv++) {
      const auto& vto = vtxOrdBrack[iv];
      auto res = tro.tBracket.isOutside(vto.tBracket);
      if (res == TBracket::Below) {                                       // vertex preceeds the track
        if (tro.tBracket.getMin() > vto.tBracket.getMin() + maxVtxSpan) { // all following vertices will be preceeding all following tracks times
          ivStart = ++iv;
          break;
        }
        continue; // following vertex with longer span might still match this track
      }
      if (res == TBracket::Above) { // track preceeds the vertex, so will preceed also all following vertices
        break;
      }
      // track matches to vertex, register
      vtxList.push_back(vto.origID); // flag matching vertex
    }
    if (vtxList.size()) {
      nAssigned++;
      for (auto v : vtxList) {
        tmpMap[v].emplace_back(tro.origID).setAmbiguous();
      }
      if (vtxList.size() > 1) { // did track match to multiple vertices?
        nAmbiguous++;
      }
    } else {
      orphans.emplace_back(tro.origID); // register unassigned track
    }
  }

  // build final vector of global indices
  trackIndex.clear();
  vtxRefs.clear();

  for (int iv = 0; iv < nv1; iv++) {
    auto& trvec = tmpMap[iv];
    // sort entries in each vertex track indices list according to the source
    std::sort(trvec.begin(), trvec.end(), [](VTIndex a, VTIndex b) { return a.getSource() < b.getSource(); });

    auto entry0 = trackIndex.size();   // start of entries for this vertex
    auto& vr = vtxRefs.emplace_back();
    vr.setVtxID(iv < nv ? iv : -1); // flag table for unassigned tracks by VtxID = -1
    int oldSrc = -1;
    for (const auto gid0 : trvec) {
      int src = gid0.getSource();
      while (oldSrc < src) {
        oldSrc++;
        vr.setFirstEntryOfSource(oldSrc, trackIndex.size()); // register start of new source
      }
      trackIndex.push_back(gid0);
    }
    while (++oldSrc < GIndex::NSources) {
      vr.setFirstEntryOfSource(oldSrc, trackIndex.size());
    }
    vr.setEnd(trackIndex.size());
    LOG(INFO) << vr;
  }
  LOG(INFO) << "Assigned " << nAssigned << " (" << nAmbiguous << " ambigously) out of " << mTBrackets.size() << " non-contributor tracks + " << vcont.size() << " contributors";
}

//________________________________________________________
void VertexTrackMatcher::extractTracks(const o2::globaltracking::RecoContainer& data, const std::unordered_map<GIndex, bool>& vcont)
{
  // Scan all inputs and create tracks

  mTBrackets.clear();

  auto creator = [this, &vcont](auto& _tr, GIndex _origID, float t0, float terr) {
    if (vcont.find(_origID) != vcont.end()) { // track is contributor to vertex, already accounted
      return true;
    }
    if constexpr (isTPCTrack<decltype(_tr)>()) {
      // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      t0 *= this->mTPCBin2MUS;
      terr *= this->mTPCBin2MUS;
    } else if (isITSTrack<decltype(_tr)>()) {
      t0 += 0.5 * this->mITSROFrameLengthMUS; // ITS time is supplied in \mus as beginning of ROF
      terr *= this->mITSROFrameLengthMUS;     // error is supplied as a half-ROF duration, convert to \mus
    } else if (isMFTTrack<decltype(_tr)>()) { // Same for MFT
      t0 += 0.5 * this->mMFTROFrameLengthMUS;
      terr *= this->mMFTROFrameLengthMUS;
    }
    // for all other tracks the time is in \mus with gaussian error
    mTBrackets.emplace_back(TrackTBracket{{t0 - terr, t0 + terr}, _origID});
    return true;
  };

  data.createTracksVariadic(creator);

  // sort in increasing min.time
  std::sort(mTBrackets.begin(), mTBrackets.end(), [](const TrackTBracket& a, const TrackTBracket& b) { return a.tBracket.getMin() < b.tBracket.getMin(); });

  LOG(INFO) << "collected " << mTBrackets.size() << " non-contributor and " << vcont.size() << " contributor seeds";
}
