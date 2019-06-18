// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ITSBase/GeometryTGeo.h"
#include "MathUtils/Cartesian3D.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"

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
class Cluster;
}

namespace its
{
class CookedTracker
{
  using Cluster = o2::itsmft::Cluster;

 public:
  CookedTracker(Int_t nThreads = 1);
  CookedTracker(const CookedTracker&) = delete;
  CookedTracker& operator=(const CookedTracker& tr) = delete;
  ~CookedTracker() = default;

  void setVertices(std::vector<std::array<Double_t, 3>>& vertices) { mVertices = std::move(vertices); }

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

  // These functions must be implemented
  void process(const std::vector<Cluster>& clusters, std::vector<TrackITS>& tracks, std::vector<int>& clusIdx,
               std::vector<o2::itsmft::ROFRecord>& rofs);
  void processFrame(std::vector<TrackITS>& tracks, std::vector<int>& clusIdx);
  const Cluster* getCluster(Int_t index) const;

  void setGeometry(o2::its::GeometryTGeo* geom);
  void setMCTruthContainers(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabels,
                            o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkLabels)
  {
    mClsLabels = clsLabels;
    mTrkLabels = trkLabels;
  }

  void setContinuousMode(bool mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }

  // internal helper classes
  class ThreadData;
  class Layer;

 protected:
  static constexpr int kNLayers = 7;
  void addOutputTrack(const TrackITSExt& t, std::vector<TrackITS>& tracks, std::vector<int>& clusIdx);
  int loadClusters(const std::vector<Cluster>& clusters, const o2::itsmft::ROFRecord& rof);
  void unloadClusters();

  std::vector<TrackITSExt> trackInThread(Int_t first, Int_t last);
  void makeSeeds(std::vector<TrackITSExt>& seeds, Int_t first, Int_t last);
  void trackSeeds(std::vector<TrackITSExt>& seeds);

  Bool_t attachCluster(Int_t& volID, Int_t nl, Int_t ci, TrackITSExt& t, const TrackITSExt& o) const;

  void makeBackPropParam(std::vector<TrackITSExt>& seeds) const;
  bool makeBackPropParam(TrackITSExt& track) const;

 private:
  bool mContinuousMode = true;                                                    ///< triggered or cont. mode
  const o2::its::GeometryTGeo* mGeom = nullptr;                                   /// interface to geometry
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; /// Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTrkLabels = nullptr;       /// Track MC labels
  std::uint32_t mFirstInFrame = 0; ///< Index of the 1st cluster of a frame (within the loaded vector of clusters)

  Int_t mNumOfThreads; ///< Number of tracking threads

  Double_t mBz; ///< Effective Z-component of the magnetic field (kG)

  std::vector<std::array<Double_t, 3>> mVertices;
  Double_t mX = 0.; ///< X-coordinate of the primary vertex
  Double_t mY = 0.; ///< Y-coordinate of the primary vertex
  Double_t mZ = 0.; ///< Z-coordinate of the primary vertex

  Double_t mSigmaX = 2.; ///< error of the primary vertex position in X
  Double_t mSigmaY = 2.; ///< error of the primary vertex position in Y
  Double_t mSigmaZ = 2.; ///< error of the primary vertex position in Z

  static Layer sLayers[kNLayers]; ///< Layers filled with clusters
  std::vector<TrackITSExt> mSeeds; ///< Track seeds

  const Cluster* mFirstCluster = nullptr; ///< Pointer to the 1st cluster in event

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

  Float_t mR;                                   ///< mean radius of this layer
  const o2::its::GeometryTGeo* mGeom = nullptr; /// interface to geometry
  std::vector<const Cluster*> mClusters;        ///< All clusters
  std::vector<Float_t> mAlphaRef;               ///< alpha of the reference plane
  std::vector<Float_t> mPhi;                    ///< cluster phi
  std::vector<std::pair<int, float>> mSectors[kNSectors]; ///< Cluster indices sector-by-sector
};
} // namespace its
} // namespace o2
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
