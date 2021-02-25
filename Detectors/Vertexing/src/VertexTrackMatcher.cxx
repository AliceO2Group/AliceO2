// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file VertexTrackMatcher.cxx
/// \brief Class for vertex track association
/// \author ruben.shahoyan@cern.ch

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
  updateTPCTimeDependentParams(); // RS FIXME eventually should be set from CCDB for every TF
}

void VertexTrackMatcher::updateTPCTimeDependentParams()
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
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName())};
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mITSROFrameLengthMUS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMS : alpParams.roFrameLengthTrig * 1.e-3;
  }
}

void VertexTrackMatcher::process(const gsl::span<const PVertex>& vertices,
                                 const gsl::span<const VTIndex>& v2tfitIDs,
                                 const gsl::span<const VRef>& v2tfitRefs,
                                 const o2::globaltracking::RecoContainer& recoData,
                                 std::vector<VTIndex>& trackIndex,
                                 std::vector<VRef>& vtxRefs)
{
  updateTPCTimeDependentParams();

  int nv = vertices.size();
  TmpMap tmpMap(nv);
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
      {float((vtx.getIRMin().differenceInBC(mStartIR) - 0.5f) * o2::constants::lhc::LHCBunchSpacingMS),
       float((vtx.getIRMax().differenceInBC(mStartIR) + 0.5f) * o2::constants::lhc::LHCBunchSpacingMS)},
      iv});
    if (vto.tBracket.delta() > maxVtxSpan) {
      maxVtxSpan = vto.tBracket.delta();
    }
  }
  // sort vertices in tmin
  std::sort(vtxOrdBrack.begin(), vtxOrdBrack.end(), [](const VtxTBracket& a, const VtxTBracket& b) { return a.tBracket.getMin() < b.tBracket.getMin(); });

  extractTracks(recoData, vcont); // extract all track t-brackets, excluding those tracks which contribute to vertex (already attached)

  int ivStart = 0;
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
    bool ambigous = vtxList.size() > 1; // did track match to multiple vertices?
    for (auto v : vtxList) {
      auto& ref = tmpMap[v].emplace_back(tro.origID);
      if (ambigous) {
        ref.setAmbiguous();
      }
    }
  }

  // build final vector of global indices
  trackIndex.clear();
  vtxRefs.clear();

  for (int iv = 0; iv < nv; iv++) {
    auto& trvec = tmpMap[iv];
    // sort entries in each vertex track indices list according to the source
    std::sort(trvec.begin(), trvec.end(), [](VTIndex a, VTIndex b) { return a.getSource() < b.getSource(); });

    auto entry0 = trackIndex.size();   // start of entries for this vertex
    auto& vr = vtxRefs.emplace_back();
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
    LOG(INFO) << "Vertxex " << iv << " Tracks " << vr;
  }
}

//________________________________________________________
void VertexTrackMatcher::extractTracks(const o2::globaltracking::RecoContainer& data, const std::unordered_map<GIndex, bool>& vcont)
{
  // Scan all inputs and create tracks

  mTBrackets.clear();

  std::function<void(const o2::track::TrackParCov& _tr, float t0, float terr, GIndex _origID)> creator =
    [this, &vcont](const o2::track::TrackParCov& _tr, float t0, float terr, GIndex _origID) {
      if (vcont.find(_origID) != vcont.end()) { // track is contributor to vertex, already accounted
        return;
      }
      if (_origID.getSource() == GIndex::TPC) { // convert TPC bins to \mus
        t0 *= this->mTPCBin2MUS;
        terr *= this->mTPCBin2MUS;
      } else if (_origID.getSource() == GIndex::ITS) { // error is supplied a half-ROF duration, convert to \mus
        t0 += this->mITSROFrameLengthMUS;
        terr *= 0.5 * this->mITSROFrameLengthMUS;
      } else {
        //terr *= this->mMatchParams->nSigmaTError;
      }
      mTBrackets.emplace_back(TrackTBracket{{t0 - terr, t0 + terr}, _origID});
    };

  data.createTracks(creator);

  // sort in increasing min.time
  std::sort(mTBrackets.begin(), mTBrackets.end(), [](const TrackTBracket& a, const TrackTBracket& b) { return a.tBracket.getMin() < b.tBracket.getMin(); });

  LOG(INFO) << "collected " << mTBrackets.size() << " seeds";
}
