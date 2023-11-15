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

/// \file StrangenessTracker.h
/// \brief
///

#ifndef _ALICEO2_STRANGENESS_TRACKER_
#define _ALICEO2_STRANGENESS_TRACKER_

#include <gsl/gsl>

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "StrangenessTracking/IndexTableUtils.h"
#include "StrangenessTracking/StrangenessTrackingConfigParam.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/DecayNBodyIndex.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"

#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

#include "DCAFitter/DCAFitterN.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace strangeness_tracking
{

enum DauType : int {
  kV0DauPos = 0,
  kV0DauNeg = 1,
  kBach = 2
};

struct ClusAttachments {

  std::array<unsigned int, 7> arr;
};

class StrangenessTracker
{
 public:
  using StrangeTrack = o2::dataformats::StrangeTrack;
  using PID = o2::track::PID;
  using TrackITS = o2::its::TrackITS;
  using ITSCluster = o2::BaseCluster<float>;
  using V0 = o2::dataformats::V0;
  using V0Index = o2::dataformats::V0Index;
  using Cascade = o2::dataformats::Cascade;
  using CascadeIndex = o2::dataformats::CascadeIndex;
  using GIndex = o2::dataformats::VtxTrackIndex;
  using DCAFitter2 = o2::vertexing::DCAFitterN<2>;
  using DCAFitter3 = o2::vertexing::DCAFitterN<3>;
  using MCLabContCl = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using MCLabSpan = gsl::span<const o2::MCCompLabel>;
  using VBracket = o2::math_utils::Bracket<int>;

  StrangenessTracker() = default;
  ~StrangenessTracker() = default;

  bool loadData(const o2::globaltracking::RecoContainer& recoData);
  bool matchDecayToITStrack(float decayR, StrangeTrack& strangeTrack, ClusAttachments& structClus, const TrackITS& itsTrack, std::vector<o2::track::TrackParCovF>& daughterTracks, int iThread = 0);
  void prepareITStracks();
  void process();
  void processV0(int iv0, const V0& v0, const V0Index& v0Idx, int iThread = 0);
  void processCascade(int icasc, const Cascade& casc, const CascadeIndex& cascIdx, const V0& cascV0, int iThread = 0);
  bool updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track);

  std::vector<ClusAttachments>& getClusAttachments(int iThread = 0) { return mClusAttachments[iThread]; };
  std::vector<StrangeTrack>& getStrangeTrackVec(int iThread = 0) { return mStrangeTrackVec[iThread]; };
  std::vector<o2::MCCompLabel>& getStrangeTrackLabels(int iThread = 0) { return mStrangeTrackLabels[iThread]; };
  size_t getNTracks(int ithread = 0) const { return ithread < (int)mStrangeTrackVec.size() ? mStrangeTrackVec[ithread].size() : 0; }

  float getBz() const { return mBz; }
  void setBz(float d) { mBz = d; }
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }
  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType& type) { mCorrType = type; }
  void setConfigParams(const StrangenessTrackingParamConfig* params) { mStrParams = params; }
  void setMCTruthOn(bool v) { mMCTruthON = v; }
  bool getMCTruthOn() const { return mMCTruthON; }

  void clear()
  {
    for (int i = 0; i < mNThreads; i++) {
      mDaughterTracks[i].clear();
      mClusAttachments[i].clear();
      mStrangeTrackVec[i].clear();
      if (mMCTruthON) {
        mStrangeTrackLabels[i].clear();
      }
    }
    mTracksIdxTable.clear();
    mSortedITStracks.clear();
    mSortedITSindexes.clear();
    mITSvtxBrackets.clear();
    mInputITSclusters.clear();
    mInputClusterSizes.clear();
  }

  void setupThreads(int nThreads = 1)
  {
    mNThreads = nThreads;
    mFitterV0.resize(nThreads);
    mFitter3Body.resize(nThreads);
    mStrangeTrackVec.resize(nThreads);
    mClusAttachments.resize(nThreads);
    mStrangeTrackLabels.resize(nThreads);
    mDaughterTracks.resize(nThreads);
  }

  void setupFitters()
  {
    for (auto& fitter : mFitterV0) {
      fitter.setBz(mBz);
      fitter.setUseAbsDCA(true);
    }
    for (auto& fitter : mFitter3Body) {
      fitter.setBz(mBz);
      fitter.setUseAbsDCA(true);
    }
  }

  double calcV0alpha(const V0& v0)
  {
    std::array<float, 3> momT, momP, momN;
    v0.getProng(0).getPxPyPzGlo(momP);
    v0.getProng(1).getPxPyPzGlo(momN);
    v0.getPxPyPzGlo(momT);
    float qNeg = momN[0] * momT[0] + momN[1] * momT[1] + momN[2] * momT[2];
    float qPos = momP[0] * momT[0] + momP[1] * momT[1] + momP[2] * momT[2];
    return (qPos - qNeg) / (qPos + qNeg);
  };

  double calcMotherMass(const std::array<float, 3>& pDauFirst, const std::array<float, 3>& pDauSecond, PID pidDauFirst, PID pidDauSecond)
  {
    double m2DauFirst = PID::getMass2(pidDauFirst);
    double m2DauSecond = PID::getMass2(pidDauSecond);
    double p2DauFirst = (pDauFirst[0] * pDauFirst[0]) + (pDauFirst[1] * pDauFirst[1]) + (pDauFirst[2] * pDauFirst[2]);
    double p2DauSecond = (pDauSecond[0] * pDauSecond[0]) + (pDauSecond[1] * pDauSecond[1]) + (pDauSecond[2] * pDauSecond[2]);
    float ePos = std::sqrt(p2DauFirst + m2DauFirst), eNeg = std::sqrt(p2DauSecond + m2DauSecond);

    double e2Mother = (ePos + eNeg) * (ePos + eNeg);
    double pxMother = (pDauFirst[0] + pDauSecond[0]);
    double pyMother = (pDauFirst[1] + pDauSecond[1]);
    double pzMother = (pDauFirst[2] + pDauSecond[2]);
    double p2Mother = (pxMother * pxMother) + (pyMother * pyMother) + (pzMother * pzMother);
    return std::sqrt(e2Mother - p2Mother);
  }

  bool recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, V0& newV0, int iThread = 0)
  {
    int nCand;
    try {
      nCand = mFitterV0[iThread].process(posTrack, negTrack);
    } catch (std::runtime_error& e) {
      return false;
    }
    if (!nCand || !mFitterV0[iThread].propagateTracksToVertex()) {
      return false;
    }

    const auto& v0XYZ = mFitterV0[iThread].getPCACandidatePos();

    auto& propPos = mFitterV0[iThread].getTrack(0, 0);
    auto& propNeg = mFitterV0[iThread].getTrack(1, 0);

    std::array<float, 3> pP, pN;
    propPos.getPxPyPzGlo(pP);
    propNeg.getPxPyPzGlo(pN);
    std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
    newV0 = V0(v0XYZ, pV0, mFitterV0[iThread].calcPCACovMatrixFlat(0), propPos, propNeg, PID::HyperTriton);
    return true;
  };

  std::vector<ITSCluster> getTrackClusters(const TrackITS& itsTrack)
  {
    std::vector<ITSCluster> outVec;
    outVec.reserve(7);
    auto firstClus = itsTrack.getFirstClusterEntry();
    auto ncl = itsTrack.getNumberOfClusters();
    for (int icl = 0; icl < ncl; icl++) {
      outVec.push_back(mInputITSclusters[mInputITSidxs[firstClus + icl]]);
    }
    return outVec;
  };

  std::vector<int> getTrackClusterSizes(const TrackITS& itsTrack)
  {
    std::vector<int> outVec;
    outVec.reserve(7);
    auto firstClus = itsTrack.getFirstClusterEntry();
    auto ncl = itsTrack.getNumberOfClusters();
    for (int icl = 0; icl < ncl; icl++) {
      outVec.push_back(mInputClusterSizes[mInputITSidxs[firstClus + icl]]);
    }
    return outVec;
  };

  void getClusterSizes(std::vector<int>& clusSizeVec, const gsl::span<const o2::itsmft::CompClusterExt> ITSclus, gsl::span<const unsigned char>::iterator& pattIt, const o2::itsmft::TopologyDictionary* mdict)
  {
    for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
      auto& clus = ITSclus[iClus];
      auto pattID = clus.getPatternID();
      int npix;
      o2::itsmft::ClusterPattern patt;

      if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict->isGroup(pattID)) {
        patt.acquirePattern(pattIt);
        npix = patt.getNPixels();
      } else {

        npix = mdict->getNpixels(pattID);
        patt = mdict->getPattern(pattID);
      }
      clusSizeVec[iClus] = npix;
    }
    // LOG(info) << " Patt Npixel: " << pattVec[0].getNPixels();
  }

  float getMatchingChi2(o2::track::TrackParCovF v0, const TrackITS& itsTrack)
  {
    if (v0.rotate(itsTrack.getParamOut().getAlpha()) && v0.propagateTo(itsTrack.getParamOut().getX(), mBz)) {
      return v0.getPredictedChi2(itsTrack.getParamOut());
    }
    return -100;
  };

  o2::MCCompLabel getStrangeTrackLabel(const TrackITS& itsTrack, const StrangeTrack& strangeTrack, const ClusAttachments& structClus) // ITS label with fake flag recomputed
  {
    bool isFake = false;
    auto itsTrkLab = mITSTrkLabels[strangeTrack.mITSRef];
    for (unsigned int iLay = 0; iLay < 7; iLay++) {
      if (itsTrack.hasHitOnLayer(iLay) && itsTrack.isFakeOnLayer(iLay) && structClus.arr[iLay] == 0) {
        isFake = true;
        break;
      }
    }
    itsTrkLab.setFakeFlag(isFake);
    return itsTrkLab;
  }

 protected:
  bool mMCTruthON = false;                            /// flag availability of MC truth
  int mNThreads = 1;                                  /// number of threads (externally driven)
  gsl::span<const TrackITS> mInputITStracks;          // input ITS tracks
  std::vector<VBracket> mITSvtxBrackets;              // time brackets for ITS tracks
  std::vector<int> mTracksIdxTable;                   // index table for ITS tracks
  std::vector<int> mInputClusterSizes;                // input cluster sizes
  std::vector<ITSCluster> mInputITSclusters;          // input ITS clusters
  gsl::span<const int> mInputITSidxs;                 // input ITS track-cluster indexes
  gsl::span<const V0> mInputV0tracks;                 // input V0 of decay daughters
  gsl::span<const V0Index> mInputV0Indices;           // input V0 indices of decay daughters
  gsl::span<const Cascade> mInputCascadeTracks;       // input cascade of decay daughters
  gsl::span<const CascadeIndex> mInputCascadeIndices; // input cascade indices of decay daughters
  const MCLabContCl* mITSClsLabels = nullptr;         /// input ITS Cluster MC labels
  MCLabSpan mITSTrkLabels;                            /// input ITS Track MC labels

  std::vector<o2::its::TrackITS> mSortedITStracks; // sorted ITS tracks
  std::vector<int> mSortedITSindexes;              // indexes of sorted ITS tracks
  IndexTableUtils mUtils;                          // structure for computing eta/phi matching selections

  std::vector<std::vector<StrangeTrack>> mStrangeTrackVec;       // structure containing updated mother and daughter tracks (per thread)
  std::vector<std::vector<ClusAttachments>> mClusAttachments;    // # of attached tracks, -1 not attached, 0 for the mother, > 0 for the daughters (per thread)
  std::vector<std::vector<o2::MCCompLabel>> mStrangeTrackLabels; // vector of MC labels for mother track (per thread)

  const StrangenessTrackingParamConfig* mStrParams = nullptr;
  float mBz = -5; // Magnetic field
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  std::vector<DCAFitter2> mFitterV0;    // optional DCA Fitter for recreating V0 with hypertriton mass hypothesis (per thread)
  std::vector<DCAFitter3> mFitter3Body; // optional DCA Fitter for final 3 Body refit (per thread)

  o2::base::PropagatorImpl<float>::MatCorrType mCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE; // use mat correction

  std::vector<std::vector<o2::track::TrackParCovF>> mDaughterTracks; // vector of daughter tracks (per thread)
  ClusAttachments mStructClus;                                       // # of attached tracks, 1 for mother, 2 for daughter

  ClassDefNV(StrangenessTracker, 1);
};

} // namespace strangeness_tracking
} // namespace o2

#endif //  _ALICEO2_STRANGENESS_TRACKER_
