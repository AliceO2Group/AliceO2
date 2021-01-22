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
#include <unordered_map>
#include <numeric>
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"

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
  auto& gasParam = o2::tpc::ParameterGas::Instance();
  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  auto& detParam = o2::tpc::ParameterDetector::Instance();
  mTPCBin2MUS = elParam.ZbinWidth;
  mMaxTPCDriftTimeMUS = detParam.TPClength / gasParam.DriftV;
}

void VertexTrackMatcher::process(const gsl::span<const PVertex>& vertices,
                                 const gsl::span<const GIndex>& v2tfitIDs,
                                 const gsl::span<const VRef>& v2tfitRefs,
                                 const gsl::span<const TrackTPCITS>& tpcitsTracks,
                                 const gsl::span<const TrackITS>& itsTracks,
                                 const gsl::span<const ITSROFR>& itsROFR,
                                 const gsl::span<const TrackTPC>& tpcTracks,
                                 std::vector<GIndex>& trackIndex,
                                 std::vector<VRef>& vtxRefs)
{

  int nv = vertices.size();
  TmpMap tmpMap(nv);
  // 1st register vertex contributors
  // TPC/ITS and TPC tracks are not sorted in time, do this and exclude indiced of tracks used in the vertex fit
  std::vector<int> idTPCITS(tpcitsTracks.size()); // indices of TPCITS tracks sorted in time
  std::iota(idTPCITS.begin(), idTPCITS.end(), 0);
  for (int iv = 0; iv < nv; iv++) {
    int idMin = v2tfitRefs[iv].getFirstEntry(), idMax = idMin + v2tfitRefs[iv].getEntries();
    auto& vtxIds = tmpMap[iv]; // global IDs of contibuting tracks
    vtxIds.reserve(v2tfitRefs[iv].getEntries());
    for (int id = idMin; id < idMax; id++) {
      auto gid = v2tfitIDs[id];
      vtxIds.push_back(gid);
      // flag already accounted tracks
      idTPCITS[gid.getIndex()] = -1; // RS Attention: this will not work once not only ITSTPC contributes to vertex, FIXME!!!
    }
  }

  std::vector<int> idVtxIRMin(vertices.size());   // indices of vertices sorted in IRmin
  std::vector<int> flgITS(itsTracks.size(), 0);
  std::vector<int> idTPC(tpcTracks.size()); // indices of TPC tracks sorted in min time
  std::iota(idTPC.begin(), idTPC.end(), 0);
  std::iota(idVtxIRMin.begin(), idVtxIRMin.end(), 0);
  for (const auto& tpcits : tpcitsTracks) { // flag standalone ITS and TPC tracks used in global matches, we exclude them from association to vertex
    flgITS[tpcits.getRefITS()] = -1;
    idTPC[tpcits.getRefTPC()] = -1;
  }
  std::sort(idVtxIRMin.begin(), idVtxIRMin.end(), [&vertices](int i, int j) { // sort according to IRMin
    return vertices[i].getIRMin() < vertices[j].getIRMin();
  });
  std::sort(idTPCITS.begin(), idTPCITS.end(), [&tpcitsTracks](int i, int j) { // sort according to central time
    float tI = (i < 0) ? 1e9 : tpcitsTracks[i].getTimeMUS().getTimeStamp();
    float tJ = (j < 0) ? 1e9 : tpcitsTracks[j].getTimeMUS().getTimeStamp();
    return tI < tJ;
  });

  std::vector<TBracket> tpcTimes;
  tpcTimes.reserve(tpcTracks.size());
  for (const auto& trc : tpcTracks) {
    tpcTimes.emplace_back(tpcTimeBin2MUS(trc.getTime0() - trc.getDeltaTBwd()), tpcTimeBin2MUS(trc.getTime0() + trc.getDeltaTFwd()));
  }
  std::sort(idTPC.begin(), idTPC.end(), [&tpcTimes](int i, int j) { // sort according to max time
    float tI = (i < 0) ? 1e9 : tpcTimes[i].getMax();
    float tJ = (j < 0) ? 1e9 : tpcTimes[j].getMax();
    return tI < tJ;
  });

  attachTPCITS(tmpMap, tpcitsTracks, idTPCITS, vertices);
  attachITS(tmpMap, itsTracks, itsROFR, flgITS, vertices, idVtxIRMin);
  attachTPC(tmpMap, tpcTimes, idTPC, vertices, idVtxIRMin);

  // build vector of global indices
  trackIndex.clear();
  vtxRefs.clear();
  // just to reuse these vectors for ambiguous attachment counting
  memset(idTPCITS.data(), 0, sizeof(int) * idTPCITS.size());
  memset(idTPC.data(), 0, sizeof(int) * idTPC.size());
  memset(flgITS.data(), 0, sizeof(int) * flgITS.size());
  std::array<std::vector<int>*, GIndex::NSources> vptr;
  vptr[GIndex::ITSTPC] = &idTPCITS;
  vptr[GIndex::ITS] = &flgITS;
  vptr[GIndex::TPC] = &idTPC;
  // flag tracks attached to >1 vertex
  for (int iv = 0; iv < nv; iv++) {
    const auto& trvec = tmpMap[iv];
    for (auto gid : trvec) {
      (*vptr[gid.getSource()])[gid.getIndex()]++;
    }
  }

  for (int iv = 0; iv < nv; iv++) {
    auto& trvec = tmpMap[iv];
    // sort entries in each vertex track indices list according to the source
    std::sort(trvec.begin(), trvec.end(), [](GIndex a, GIndex b) { return a.getSource() < b.getSource(); });

    auto entry0 = trackIndex.size();   // start of entries for this vertex
    auto& vr = vtxRefs.emplace_back(); //entry0, 0);
    int oldSrc = -1;
    for (const auto gid0 : trvec) {
      int src = gid0.getSource();
      while (oldSrc < src) {
        oldSrc++;
        vr.setFirstEntryOfSource(oldSrc, trackIndex.size()); // register start of new source
      }
      auto& gid = trackIndex.emplace_back(gid0);
      if ((*vptr[src])[gid.getIndex()] > 1) {
        gid.setAmbiguous();
      }
    }
    while (++oldSrc < GIndex::NSources) {
      vr.setFirstEntryOfSource(oldSrc, trackIndex.size());
    }
    vr.setEnd(trackIndex.size());
    LOG(INFO) << "Vertxex " << iv << " Tracks " << vr;
  }
}

///______________________________________________________________________________________
void VertexTrackMatcher::attachTPCITS(TmpMap& tmpMap, const gsl::span<const TrackTPCITS>& tpcits, const std::vector<int>& idTPCITS, const gsl::span<const PVertex>& vertices)
{
  int itrCurr = 0, nvt = vertices.size(), ntr = tpcits.size();
  // indices of tracks used for vertex fit will be in the end, find their start and max error
  float maxErr = 0;
  for (int i = 0; i < ntr; i++) {
    if (idTPCITS[i] < 0) {
      ntr = i;
      break;
    }
    float err = tpcits[idTPCITS[i]].getTimeMUS().getTimeStampError();
    if (maxErr < err) {
      maxErr = err;
    }
  }
  auto maxErr2 = maxErr * maxErr;
  for (int ivtCurr = 0; ivtCurr < nvt; ivtCurr++) {
    const auto& vtxT = vertices[ivtCurr].getTimeStamp();
    auto rangeT = mPVParams->nSigmaTimeCut * std::sqrt(maxErr2 + vtxT.getTimeStampError() * vtxT.getTimeStampError());
    float minTime = vtxT.getTimeStamp() - rangeT, maxTime = vtxT.getTimeStamp() + rangeT;
    // proceed to the 1st track having compatible time
    int itr = itrCurr;
    while (itr < ntr) {
      const auto& trcT = tpcits[idTPCITS[itr]].getTimeMUS();
      if (trcT.getTimeStamp() < minTime) {
        itrCurr = itr;
      } else if (trcT.getTimeStamp() > maxTime) {
        break;
      } else if (compatibleTimes(vtxT, trcT)) {
        tmpMap[ivtCurr].emplace_back(idTPCITS[itr], GIndex::ITSTPC);
      }
      itr++;
    }
  }
}

///______________________________________________________________________________________
void VertexTrackMatcher::attachITS(TmpMap& tmpMap, const gsl::span<const TrackITS>& itsTracks, const gsl::span<const ITSROFR>& itsROFR, const std::vector<int>& flITS,
                                   const gsl::span<const PVertex>& vertices, std::vector<int>& idxVtx)
{
  int irofCurr = 0, nvt = vertices.size(), ntr = itsTracks.size();
  int nROFs = itsROFR.size();

  for (int ivtCurr = 0; ivtCurr < nvt; ivtCurr++) {
    const auto& vtx = vertices[idxVtx[ivtCurr]]; // we iterate over vertices in order of their getIRMin()
    int irof = irofCurr;
    while (irof < nROFs) {
      const auto& rofr = itsROFR[irof];
      auto irMin = rofr.getBCData(), irMax = irMin + mITSROFrameLengthInBC;
      if (irMax < vtx.getIRMin()) {
        irofCurr = irof;
      } else if (irMin > vtx.getIRMax()) {
        break;
      } else {
        if (irMax == vtx.getIRMin()) {
          irofCurr = irof;
        }
        int maxItr = rofr.getNEntries() + rofr.getFirstEntry();
        for (int itr = rofr.getFirstEntry(); itr < maxItr; itr++) {
          if (flITS[itr] != -1) {
            tmpMap[ivtCurr].emplace_back(itr, GIndex::ITS);
          }
        }
      }
      irof++;
    }
  }
}

///______________________________________________________________________________________
void VertexTrackMatcher::attachTPC(TmpMap& tmpMap, const std::vector<TBracket>& tpcTimes, const std::vector<int>& idTPC,
                                   const gsl::span<const PVertex>& vertices, std::vector<int>& idVtx)
{
  int itrCurr = 0, nvt = vertices.size(), ntr = idTPC.size();
  // indices of tracks used for vertex fit will be in the end, find their start and max error
  for (int i = 0; i < ntr; i++) {
    if (idTPC[i] < 0) {
      ntr = i;
      break;
    }
  }
  for (int ivtCurr = 0; ivtCurr < nvt; ivtCurr++) {
    const auto& vtx = vertices[idVtx[ivtCurr]];
    TBracket tV(vtx.getIRMin().differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3,
                vtx.getIRMax().differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3);
    // proceed to the 1st track having compatible time, idTPC provides track IDs ordered in max time
    int itr = itrCurr;
    while (itr < ntr) {
      const auto& tT = tpcTimes[idTPC[itr]];
      auto rel = tV.isOutside(tT);
      if (rel == TBracket::Below) { // tmax of track < tmin of vtx
        itrCurr = itr;
      } else if (rel == TBracket::Inside) {
        tmpMap[ivtCurr].emplace_back(idTPC[itr], GIndex::TPC);
      } else if (tT.getMax() - mMaxTPCDriftTimeMUS > tV.getMax()) {
        break;
      }
      itr++;
    }
  }
}

//______________________________________________
void VertexTrackMatcher::setITSROFrameLengthInBC(int nbc)
{
  mITSROFrameLengthInBC = nbc;
}

//______________________________________________
bool VertexTrackMatcher::compatibleTimes(const TimeEst& vtxT, const TimeEst& trcT) const
{
  float err2 = vtxT.getTimeStampError() * vtxT.getTimeStampError() + trcT.getTimeStampError() * trcT.getTimeStampError();
  float dfred = (vtxT.getTimeStamp() - trcT.getTimeStamp()) * mPVParams->nSigmaTimeCut;
  return dfred * dfred < err2;
}
