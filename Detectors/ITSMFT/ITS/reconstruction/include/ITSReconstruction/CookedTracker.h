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
#include "ITSReconstruction/CookedTrack.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
  template<typename T>
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
  CookedTracker(Int_t nThreads=1);
  CookedTracker(const CookedTracker&) = delete;
  CookedTracker& operator=(const CookedTracker& tr) = delete;
  ~CookedTracker() = default;

  void setVertex(const Double_t* xyz, const Double_t* ers = nullptr)
  {
    mX = xyz[0];
    mY = xyz[1];
    mZ = xyz[2];
    if (ers) {
      mSigmaX = ers[0];
      mSigmaY = ers[1];
      mSigmaZ = ers[2];
    }
  }
  Double_t getX() const { return mX; }
  Double_t getY() const { return mY; }
  Double_t getZ() const { return mZ; }
  Double_t getSigmaX() const { return mSigmaX; }
  Double_t getSigmaY() const { return mSigmaY; }
  Double_t getSigmaZ() const { return mSigmaZ; }
  o2::MCCompLabel cookLabel(CookedTrack& t, Float_t wrong) const;
  void setExternalIndices(CookedTrack& t) const;
  Double_t getBz() const;
  void setBz(Double_t bz) { mBz = bz; }

  void setNumberOfThreads(Int_t n) { mNumOfThreads=n; }
  Int_t getNumberOfThreads() const { return mNumOfThreads; }
  
  // These functions must be implemented
  void process(const std::vector<Cluster> &clusters, std::vector<CookedTrack> &tracks);
  void processFrame(std::vector<CookedTrack> &tracks);
  // Int_t propagateBack(std::vector<CookedTrack> *event);
  // Int_t RefitInward(std::vector<CookedTrack> *event);
  // Bool_t refitAt(Double_t x, CookedTrack *seed, const CookedTrack *t);
  const Cluster* getCluster(Int_t index) const;

  void setGeometry(o2::ITS::GeometryTGeo* geom);
  void setMCTruthContainers(const o2::dataformats::MCTruthContainer<o2::MCCompLabel> *clsLabels,
                            o2::dataformats::MCTruthContainer<o2::MCCompLabel> *trkLabels) {
    mClsLabels=clsLabels;
    mTrkLabels=trkLabels;
  }

  void setContinuousMode(bool mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }

  
  // internal helper classes
  class ThreadData;
  class Layer;

 protected:
  static constexpr int kNLayers = 7;
  int loadClusters(const std::vector<Cluster> &clusters);
  void unloadClusters();
  
  std::vector<CookedTrack> trackInThread(Int_t first, Int_t last);
  void makeSeeds(std::vector<CookedTrack> &seeds, Int_t first, Int_t last);
  void trackSeeds(std::vector<CookedTrack> &seeds);

  Bool_t attachCluster(Int_t& volID, Int_t nl, Int_t ci, CookedTrack& t, const CookedTrack& o) const;

  void makeBackPropParam(std::vector<CookedTrack> &seeds) const;
  bool makeBackPropParam(CookedTrack& track) const;
  
 private:

  bool mContinuousMode = true; ///< triggered or cont. mode
  const o2::ITS::GeometryTGeo* mGeom = nullptr; /// interface to geometry
  const
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mClsLabels = nullptr; /// Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mTrkLabels = nullptr; /// Track MC labels

  std::uint32_t mROFrame=0;  ///< last frame processed
  
  Int_t mNumOfThreads; ///< Number of tracking threads
  
  Double_t mBz;///< Effective Z-component of the magnetic field (kG)
  Double_t mX; ///< X-coordinate of the primary vertex
  Double_t mY; ///< Y-coordinate of the primary vertex
  Double_t mZ; ///< Z-coordinate of the primary vertex

  Double_t mSigmaX; ///< error of the primary vertex position in X
  Double_t mSigmaY; ///< error of the primary vertex position in Y
  Double_t mSigmaZ; ///< error of the primary vertex position in Z

  static Layer sLayers[kNLayers];  ///< Layers filled with clusters
  std::vector<CookedTrack> mSeeds; ///< Track seeds
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
  void selectClusters(std::vector<Int_t> &s, Float_t phi, Float_t dy, Float_t z, Float_t dz);
  Int_t findClusterIndex(Double_t z) const;
  Float_t getR() const { return mR; }
  const Cluster* getCluster(Int_t i) const { return mClusters[i]; }
  Float_t getAlphaRef(Int_t i) const { return mAlphaRef[i]; }
  Float_t getClusterPhi(Int_t i) const { return mPhi[i]; }
  Int_t getNumberOfClusters() const { return mClusters.size(); }
  void  setGeometry(o2::ITS::GeometryTGeo* geom) { mGeom = geom; }

 protected:
  enum {kNSectors=21};

  Float_t mR; ///< mean radius of this layer
  const o2::ITS::GeometryTGeo* mGeom = nullptr; /// interface to geometry
  std::vector<const Cluster*>mClusters;          ///< All clusters
  std::vector<Float_t> mAlphaRef;          ///< alpha of the reference plane
  std::vector<Float_t> mPhi;               ///< cluster phi
  std::vector<Int_t> mSectors[kNSectors];  ///< Cluster indices sector-by-sector
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
