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

/// \file CookedTracker.h
/// \brief Definition of the "Cooked Matrix" ITS tracker
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACKER_H
#define ALICEO2_ITS_COOKEDTRACKER_H

//-------------------------------------------------------------------------
//                   A stand-alone ITS tracker
//    The pattern recongintion based on the "cooked covariance" approach
//-------------------------------------------------------------------------

#include <vector>
#include <tuple>
#include "ITSBase/GeometryTGeo.h"
#include "MathUtils/Cartesian.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ITSReconstruction/CookedConfigParam.h"

using Point3Df = o2::math_utils::Point3D<float>;

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}
namespace itsmft
{
class TopologyDictionary;
class CompClusterExt;
} // namespace itsmft
namespace its
{
class CookedTracker
{
  using Cluster = o2::itsmft::Cluster;
  using CompClusterExt = o2::itsmft::CompClusterExt;
  using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

 public:
  CookedTracker(Int_t nThreads = 1);
  CookedTracker(const CookedTracker&) = delete;
  CookedTracker& operator=(const CookedTracker& tr) = delete;
  ~CookedTracker() = default;

  void setConfigParams()
  {
    const auto& par = CookedConfigParam::Instance();
    LOG(INFO) << " Setting configurable parameters...";

    gzWin = par.zWin;
    gminPt = par.minPt;
    gmaxDCAxy = par.maxDCAxy;
    gmaxDCAz = par.maxDCAz;
    gSigma2 = par.sigma * par.sigma;
    gRoadY = par.roadY;
    gRoadZ = par.roadZ;
    gminNumberOfClusters = par.minNumberOfClusters;
  }
  void setParameters(const std::vector<float>& par)
  {
    gzWin = par[0];
    gminPt = par[1];
    gmaxDCAxy = par[3];
    gmaxDCAz = par[4];
    gSeedingLayer1 = par[5];
    gSeedingLayer2 = par[6];
    gSeedingLayer3 = par[7];
    gSigma2 = par[8] * par[8];
    gmaxChi2PerCluster = par[9];
    gmaxChi2PerTrack = par[10];
    gRoadY = par[11];
    gRoadZ = par[12];
    gminNumberOfClusters = par[13];
  }
  void setParametersCosmics()
  {
    // seed "windows" in z and phi: makeSeeds
    gzWin = 84.; // length of the L3
    gminPt = 10.;
    // Maximal accepted impact parameters for the seeds
    gmaxDCAxy = 19.4; // radius of the L3
    gmaxDCAz = 42.;   // half-lenght of the L3
    // Space point resolution
    gSigma2 = 0.2 * 0.2;
    // Tracking "road" from layer to layer
    gRoadY = 1.5; // Chip size in Y
    gRoadZ = 3.0; // Chip size in Z
  }

  void setVertices(const std::vector<Vertex>& vertices)
  {
    mVertices = &vertices;
  }

  Double_t getX() const { return mX; }
  Double_t getY() const { return mY; }
  Double_t getZ() const { return mZ; }
  Double_t getSigmaX() const { return mSigmaX; }
  Double_t getSigmaY() const { return mSigmaY; }
  Double_t getSigmaZ() const { return mSigmaZ; }
  o2::MCCompLabel cookLabel(TrackITSExt& t, Float_t wrong) const;
  void setExternalIndices(TrackITSExt& t) const;
  Double_t getBz() const;
  void setBz(Double_t bz) { mBz = bz; }

  void setNumberOfThreads(Int_t n) { mNumOfThreads = n; }
  Int_t getNumberOfThreads() const { return mNumOfThreads; }

  using TrackInserter = std::function<int(const TrackITSExt& t)>;
  // These functions must be implemented
  template <typename U, typename V>
  void process(gsl::span<const CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& it, const o2::itsmft::TopologyDictionary& dict, U& tracks, V& clusIdx, o2::itsmft::ROFRecord& rof)
  {
    TrackInserter inserter = [&tracks, &clusIdx, this](const TrackITSExt& t) -> int {
      // convert internal track to output format
      auto& trackNew = tracks.emplace_back(t);
      int noc = t.getNumberOfClusters();
      int clEntry = clusIdx.size();
      for (int i = 0; i < noc; i++) {
        const Cluster* c = this->getCluster(t.getClusterIndex(i));
        Int_t idx = c - &mClusterCache[0]; // Index of this cluster in event
        clusIdx.emplace_back(this->mFirstInFrame + idx);
      }
      trackNew.setClusterRefs(clEntry, noc);
      trackNew.setPattern(0x7f); // this tracker finds only complete tracks
      return tracks.size();
    };
    process(clusters, it, dict, inserter, rof);
  }
  void process(gsl::span<const CompClusterExt> const& clusters, gsl::span<const unsigned char>::iterator& it, const o2::itsmft::TopologyDictionary& dict, TrackInserter& inserter, o2::itsmft::ROFRecord& rof);
  const Cluster* getCluster(Int_t index) const;

  void setGeometry(o2::its::GeometryTGeo* geom);
  void setMCTruthContainers(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabels, std::vector<o2::MCCompLabel>* trkLabels)
  {
    mClsLabels = clsLabels;
    mTrkLabels = trkLabels;
  }

  void setContinuousMode(bool mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }

  static void setMostProbablePt(float pt) { mMostProbablePt = pt; }
  static auto getMostProbablePt() { return mMostProbablePt; }

  // internal helper classes
  class ThreadData;
  class Layer;

 protected:
  static constexpr int kNLayers = 7;
  int loadClusters();
  void unloadClusters();
  std::tuple<int, int> processLoadedClusters(TrackInserter& inserter);

  std::vector<TrackITSExt> trackInThread(Int_t first, Int_t last);
  o2::its::TrackITSExt cookSeed(const Point3Df& r1, Point3Df& r2, const Point3Df& tr3, float rad2, float rad3, float_t alpha, float_t bz);
  void makeSeeds(std::vector<TrackITSExt>& seeds, Int_t first, Int_t last);
  void trackSeeds(std::vector<TrackITSExt>& seeds);

  Bool_t attachCluster(Int_t& volID, Int_t nl, Int_t ci, TrackITSExt& t, const TrackITSExt& o) const;

  void makeBackPropParam(std::vector<TrackITSExt>& seeds) const;
  bool makeBackPropParam(TrackITSExt& track) const;

 private:
  /*** Tracking parameters ***/
  // seed "windows" in z and phi: makeSeeds
  static Float_t gzWin;
  static Float_t gminPt;
  static Float_t mMostProbablePt; ///< settable most probable pt
  // Maximal accepted impact parameters for the seeds
  static Float_t gmaxDCAxy;
  static Float_t gmaxDCAz;
  // Layers for the seeding
  static Int_t gSeedingLayer1;
  static Int_t gSeedingLayer2;
  static Int_t gSeedingLayer3;
  // Space point resolution
  static Float_t gSigma2;
  // Max accepted chi2
  static Float_t gmaxChi2PerCluster;
  static Float_t gmaxChi2PerTrack;
  // Tracking "road" from layer to layer
  static Float_t gRoadY;
  static Float_t gRoadZ;
  // Minimal number of attached clusters
  static Int_t gminNumberOfClusters;

  bool mContinuousMode = true;                                                    ///< triggered or cont. mode
  const o2::its::GeometryTGeo* mGeom = nullptr;                                   /// interface to geometry
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; /// Cluster MC labels
  std::vector<o2::MCCompLabel>* mTrkLabels = nullptr;                             /// Track MC labels
  std::uint32_t mFirstInFrame = 0;                                                ///< Index of the 1st cluster of a frame (within the loaded vector of clusters)

  Int_t mNumOfThreads; ///< Number of tracking threads

  Double_t mBz; ///< Effective Z-component of the magnetic field (kG)

  const std::vector<Vertex>* mVertices = nullptr;
  Double_t mX = 0.; ///< X-coordinate of the primary vertex
  Double_t mY = 0.; ///< Y-coordinate of the primary vertex
  Double_t mZ = 0.; ///< Z-coordinate of the primary vertex

  Double_t mSigmaX = 2.; ///< error of the primary vertex position in X
  Double_t mSigmaY = 2.; ///< error of the primary vertex position in Y
  Double_t mSigmaZ = 2.; ///< error of the primary vertex position in Z

  static Layer sLayers[kNLayers];  ///< Layers filled with clusters
  std::vector<TrackITSExt> mSeeds; ///< Track seeds

  std::vector<Cluster> mClusterCache;

  ClassDefNV(CookedTracker, 1);
};

class CookedTracker::Layer
{
 public:
  Layer();
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer& tr) = delete;

  void init();
  Bool_t insertCluster(const Cluster* c);
  void setR(Double_t r) { mR = r; }
  void unloadClusters();
  void selectClusters(std::vector<Int_t>& s, Float_t phi, Float_t dy, Float_t z, Float_t dz);
  Int_t findClusterIndex(Float_t z) const;
  Float_t getR() const { return mR; }
  const Cluster* getCluster(Int_t i) const { return mClusters[i]; }
  Float_t getAlphaRef(Int_t i) const { return mAlphaRef[i]; }
  Float_t getClusterPhi(Int_t i) const { return mPhi[i]; }
  Int_t getNumberOfClusters() const { return mClusters.size(); }
  void setGeometry(o2::its::GeometryTGeo* geom) { mGeom = geom; }

 protected:
  enum { kNSectors = 21 };

  Float_t mR;                                             ///< mean radius of this layer
  const o2::its::GeometryTGeo* mGeom = nullptr;           ///< interface to geometry
  std::vector<const Cluster*> mClusters;                  ///< All clusters
  std::vector<Float_t> mAlphaRef;                         ///< alpha of the reference plane
  std::vector<Float_t> mPhi;                              ///< cluster phi
  std::vector<std::pair<int, float>> mSectors[kNSectors]; ///< Cluster indices sector-by-sector
};
} // namespace its
} // namespace o2
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
