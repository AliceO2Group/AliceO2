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

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace ITSMFT
{
class Cluster;
}

namespace ITS
{
class CookedTracker
{
  using Cluster = o2::ITSMFT::Cluster;

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
  o2::MCCompLabel cookLabel(TrackITS& t, Float_t wrong) const;
  void setExternalIndices(TrackITS& t) const;
  Double_t getBz() const;
  void setBz(Double_t bz) { mBz = bz; }

  void setNumberOfThreads(Int_t n) { mNumOfThreads = n; }
  Int_t getNumberOfThreads() const { return mNumOfThreads; }

  // These functions must be implemented
  void process(const std::vector<Cluster>& clusters, std::vector<TrackITS>& tracks);
  void processFrame(std::vector<TrackITS>& tracks);
  // Int_t propagateBack(std::vector<TrackITS> *event);
  // Int_t RefitInward(std::vector<TrackITS> *event);
  // Bool_t refitAt(Double_t x, TrackITS *seed, const TrackITS *t);
  const Cluster* getCluster(Int_t index) const;

  void setGeometry(o2::ITS::GeometryTGeo* geom);
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
  int loadClusters(const std::vector<Cluster>& clusters);
  void unloadClusters();

  std::vector<TrackITS> trackInThread(Int_t first, Int_t last);
  void makeSeeds(std::vector<TrackITS>& seeds, Int_t first, Int_t last);
  void trackSeeds(std::vector<TrackITS>& seeds);

  Bool_t attachCluster(Int_t& volID, Int_t nl, Int_t ci, TrackITS& t, const TrackITS& o) const;

  void makeBackPropParam(std::vector<TrackITS>& seeds) const;
  bool makeBackPropParam(TrackITS& track) const;

 private:
  bool mContinuousMode = true;                                                    ///< triggered or cont. mode
  const o2::ITS::GeometryTGeo* mGeom = nullptr;                                   /// interface to geometry
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; /// Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTrkLabels = nullptr;       /// Track MC labels

  std::uint32_t mROFrame = 0; ///< last frame processed

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
  std::vector<TrackITS> mSeeds;   ///< Track seeds

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
  Int_t findClusterIndex(Double_t z) const;
  Float_t getR() const { return mR; }
  const Cluster* getCluster(Int_t i) const { return mClusters[i]; }
  Float_t getAlphaRef(Int_t i) const { return mAlphaRef[i]; }
  Float_t getClusterPhi(Int_t i) const { return mPhi[i]; }
  Int_t getNumberOfClusters() const { return mClusters.size(); }
  void setGeometry(o2::ITS::GeometryTGeo* geom) { mGeom = geom; }

 protected:
  enum { kNSectors = 21 };

  Float_t mR;                                   ///< mean radius of this layer
  const o2::ITS::GeometryTGeo* mGeom = nullptr; /// interface to geometry
  std::vector<const Cluster*> mClusters;        ///< All clusters
  std::vector<Float_t> mAlphaRef;               ///< alpha of the reference plane
  std::vector<Float_t> mPhi;                    ///< cluster phi
  std::vector<Int_t> mSectors[kNSectors];       ///< Cluster indices sector-by-sector
};
} // namespace ITS
} // namespace o2
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
