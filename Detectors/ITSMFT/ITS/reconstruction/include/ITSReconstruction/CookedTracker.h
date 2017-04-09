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

class TClonesArray;

namespace o2
{
namespace ITS
{
class Cluster;
class CookedTrack;

class CookedTracker
{
 public:
  CookedTracker(Int_t nThreads=1);
  CookedTracker(const CookedTracker&) = delete;
  CookedTracker& operator=(const CookedTracker& tr) = delete;
  virtual ~CookedTracker();

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
  void cookLabel(CookedTrack& t, Float_t wrong) const;
  void setExternalIndices(CookedTrack& t) const;
  Double_t getBz() const;
  void setBz(Double_t bz) { mBz = bz; }

  void setNumberOfThreads(Int_t n) { mNumOfThreads=n; }
  Int_t getNumberOfThreads() const { return mNumOfThreads; }
  
  // These functions must be implemented
  void process(const TClonesArray& clusters, TClonesArray& tracks);
  // Int_t propagateBack(std::vector<CookedTrack> *event);
  // Int_t RefitInward(std::vector<CookedTrack> *event);
  // Bool_t refitAt(Double_t x, CookedTrack *seed, const CookedTrack *t);
  Cluster* getCluster(Int_t index) const;

  // internal helper classes
  class ThreadData;
  class Layer;

 protected:
  enum {kNLayers=7};
  void loadClusters(const TClonesArray& clusters);
  void unloadClusters();
  
  std::vector<CookedTrack> trackInThread(Int_t first, Int_t last);
  void makeSeeds(std::vector<CookedTrack> &seeds, Int_t first, Int_t last);
  void trackSeeds(std::vector<CookedTrack> &seeds);

  Bool_t attachCluster(Int_t& volID, Int_t nl, Int_t ci, CookedTrack& t, const CookedTrack& o) const;

 private:
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
  Bool_t insertCluster(Cluster* c);
  void setR(Double_t r) { mR = r; }
  void unloadClusters();
  void selectClusters(std::vector<Int_t> &s, Float_t phi, Float_t dy, Float_t z, Float_t dz);
  Int_t findClusterIndex(Double_t z) const;
  Float_t getR() const { return mR; }
  Cluster* getCluster(Int_t i) const { return mClusters[i]; }
  Float_t getXRef(Int_t i) const { return mXRef[i]; }
  Float_t getAlphaRef(Int_t i) const { return mAlphaRef[i]; }
  Float_t getClusterPhi(Int_t i) const { return mPhi[i]; }
  Int_t getNumberOfClusters() const { return mClusters.size(); }

 protected:
  enum {kNSectors=21};

  Float_t mR; ///< mean radius of this layer

  std::vector<Cluster*>mClusters;          ///< All clusters
  std::vector<Float_t> mXRef;              ///< x of the reference plane
  std::vector<Float_t> mAlphaRef;          ///< alpha of the reference plane
  std::vector<Float_t> mPhi;               ///< cluster phi
  std::vector<Int_t> mSectors[kNSectors];  ///< Cluster indices sector-by-sector
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
