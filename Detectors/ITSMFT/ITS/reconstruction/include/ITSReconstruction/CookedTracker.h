/// \file CookedTracker.h
/// \brief Definition of the "Cooked Matrix" ITS tracker
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACKER_H
#define ALICEO2_ITS_COOKEDTRACKER_H

//-------------------------------------------------------------------------
//                   A stand-alone ITS tracker
//    The pattern recongintion based on the "cooked covariance" approach
//-------------------------------------------------------------------------

class std::vector;

class TTree;
class TClonesArray;
class TObjArray;

class CookedTrack;

namespace AliceO2 {
namespace ITS {

class Cluster;

class CookedTracker {
public:
  enum {
     kNLayers=7, kMaxClusterPerLayer=150000, kMaxSelected=kMaxClusterPerLayer/10
  };
  CookedTracker();
  virtual ~CookedTracker();

  // These functions must be implemented 
  Int_t Clusters2Tracks(std::vector *event);
  Int_t PropagateBack(std::vector *event);
  Int_t RefitInward(std::vector *event);
  Int_t LoadClusters(TTree *ct);
  void UnloadClusters();
  Cluster *GetCluster(Int_t index) const;

  // Other public functions
  Bool_t
  RefitAt(Double_t x, CookedTrack *seed, const CookedTrack *t);
  void SetSAonly(Bool_t sa=kTRUE) {fSAonly=sa;}
  Bool_t GetSAonly() const {return fSAonly;}

  // internal helper classes
  class AliITSUthreadData;
  class AliITSUlayer;

protected:
  CookedTracker(const CookedTracker&);
  // Other protected functions
  Int_t MakeSeeds();
  Bool_t AddCookedSeed(const Float_t r1[3], Int_t l1, Int_t i1,
                       const Float_t r2[3], Int_t l2, Int_t i2,
                       const Cluster *c3,Int_t l3, Int_t i3);

  void LoopOverSeeds(Int_t inx[], Int_t n);

  Bool_t AttachCluster(Int_t &volID, Int_t nl, Int_t ci,
         AliKalmanTrack &t, const AliKalmanTrack &o) const;

private:
  CookedTracker &operator=(const CookedTracker &tr);

  // Data members
  // Internal tracker arrays, layers, modules, etc
  static AliITSUlayer fgLayers[kNLayers];// Layers
    
  TObjArray *fSeeds; // Track seeds

  Bool_t fSAonly; // kTRUE if the standalone tracking only

};



// The helper classes
class CookedTracker::AliITSUthreadData {
  public:
    AliITSUthreadData();
   ~AliITSUthreadData() {}
    void ResetSelectedClusters() {fI=0;}
    Int_t *Index() {return fIndex;}
    Int_t &Nsel() {return fNsel;}
    Int_t GetNextClusterIndex() {
      while (fI<fNsel) {
         Int_t ci=fIndex[fI++];
         if (!fUsed[ci]) return ci;
      }
      return -1;
    }
    void UseCluster(Int_t i) { fUsed[i]=kTRUE; }

  private:
    AliITSUthreadData(const AliITSUthreadData&);
    AliITSUthreadData &operator=(const AliITSUthreadData &tr);
    Int_t fIndex[kMaxSelected]; // Indices of selected clusters
    Int_t fNsel;      // Number of selected clusters
    Int_t fI;         // Running index for the selected clusters
    Bool_t fUsed[kMaxClusterPerLayer]; // Cluster usage flags
};

class CookedTracker::AliITSUlayer {
  public:
    AliITSUlayer();

    void InsertClusters(TClonesArray *clusters, Bool_t seedingLayer, Bool_t sa);
    void SetR(Double_t r) {fR=r;}
    void DeleteClusters();
    void SelectClusters(Int_t &i, Int_t idx[], Float_t phi, Float_t dy, Float_t z, Float_t dz);
    Int_t FindClusterIndex(Double_t z) const;
    Float_t GetR() const {return fR;}
    Cluster *GetCluster(Int_t i) const { return fClusters[i]; } 
    Float_t GetXRef(Int_t i) const { return fXRef[i]; } 
    Float_t GetAlphaRef(Int_t i) const { return fAlphaRef[i]; } 
    Float_t GetClusterPhi(Int_t i) const { return fPhi[i]; } 
    Int_t GetNumberOfClusters() const {return fN;}

  protected:
    AliITSUlayer(const AliITSUlayer&);
    AliITSUlayer &operator=(const AliITSUlayer &tr);  
    Int_t InsertCluster(Cluster *c);

    Float_t fR;                // mean radius of this layer

    Cluster *fClusters[kMaxClusterPerLayer]; // All clusters
    Float_t fXRef[kMaxClusterPerLayer];     // x of the reference plane
    Float_t fAlphaRef[kMaxClusterPerLayer]; // alpha of the reference plane
    Float_t fPhi[kMaxClusterPerLayer]; // cluster phi 
    Int_t fN; // Total number of clusters 
};
}
}
#endif /* ALICEO2_ITS_COOKEDTRACKER_H */
