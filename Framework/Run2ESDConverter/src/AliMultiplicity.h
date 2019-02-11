#ifndef ALIMULTIPLICITY_H
#define ALIMULTIPLICITY_H

#include <TBits.h>
#include <TMath.h>
#include "AliVMultiplicity.h"
class AliRefArray;

////////////////////////////////////////////////////////
////   Class containing multiplicity information      //
////   to stored in the ESD                           //
////////////////////////////////////////////////////////

class AliMultiplicity : public AliVMultiplicity {

 public:
  //
  AliMultiplicity();               // default constructor
  AliMultiplicity(Int_t ntr,Float_t *th, Float_t *ph, Float_t *dth, Float_t *dph, Int_t *labels,
         Int_t* labelsL2, Int_t ns, Float_t *ts, Float_t *ps, Int_t *labelss, Short_t nfcL1, Short_t nfcL2, const TBits & fFastOrFiredChips);
  AliMultiplicity(Int_t ntr, Int_t ns, Short_t nfcL1, Short_t nfcL2, const TBits & fFastOr);
  AliMultiplicity(const AliMultiplicity& m);
  AliMultiplicity& operator=(const AliMultiplicity& m);
  virtual void Copy(TObject &obj) const;
  virtual void Clear(Option_t* opt="");
  virtual ~AliMultiplicity();
  // methods to access tracklet information
  Bool_t  GetMultTrackRefs()                  const {return TestBit(kMultTrackRefs);}
  void    SetMultTrackRefs(Bool_t v)                {SetBit(kMultTrackRefs,v);}
  Bool_t  GetScaleDThetaBySin2T()             const {return TestBit(kScaleDThtbySin2);}
  void    SetScaleDThetaBySin2T(Bool_t v)           {SetBit(kScaleDThtbySin2,v);}
  //
  // methods supported on AliVMultiplicity level >>>
  //
  virtual  Int_t    GetNumberOfTracklets() const {if (fNtracks==0 && fNtracksOnline>0) { return fNtracksOnline; } return fNtracks;}
  virtual  void     SetNumberOfTracklets( Int_t tr) { fNtracksOnline = tr; } //used online (HLT)
  virtual  Double_t GetTheta(Int_t i)      const { 
    if(i>=0 && i<fNtracks) return fTh[i];
    Error("GetTheta","Invalid track number %d",i); return -9999.;
  }
  virtual  Double_t GetPhi(Int_t i)        const { 
    if(i>=0 && i<fNtracks) return fPhi[i];
    Error("GetPhi","Invalid track number %d",i); return -9999.;
  }
  virtual Double_t GetDeltaPhi(Int_t i)    const {
    if(i>=0 && i<fNtracks) return fDeltPhi[i];
    Error("GetDeltaPhi","Invalid track number %d",i); return -9999.;
  }
  virtual Int_t GetLabel(Int_t i, Int_t layer) const;
  virtual void  SetLabel(Int_t i, Int_t layer, Int_t label);
  //
  // array getters
  virtual Double_t* GetTheta()       const {return (Double_t*)fTh;}
  virtual Double_t* GetPhi()         const {return (Double_t*)fPhi;}
  virtual Double_t* GetDeltPhi()     const {return (Double_t*)fDeltPhi;}
  virtual Int_t*    GetLabels()      const {return (Int_t*)fLabels;}  
  virtual Int_t*    GetLabels2()     const {return (Int_t*)fLabelsL2;}
  //
  virtual void Print(Option_t *opt="") const;
  //
  // methods supported on AliVMultiplicity level <<<
  //
  Double_t GetDeltaTheta(Int_t i)          const {
    if(fDeltTh && i>=0 && i<fNtracks) return fDeltTh[i];
    Error("GetDeltaTheta","DeltaTheta not available in data or Invalid track number %d(max %d)",i, fNtracks); return -9999.;
  }

  Double_t  CalcDist(Int_t it)  const;
  Float_t GetThetaAll(int icl, int lr) const;
  Float_t GetPhiAll(int icl, int lr) const;
  Int_t   GetLabelAll(int icl, int lr) const;

  Int_t GetLabelSingle(Int_t i) const;
  Int_t GetLabelSingleLr(Int_t i, Int_t layer) const;
  void  SetLabelSingle(Int_t i, Int_t label);

  Bool_t FreeClustersTracklet(Int_t i, Int_t mode) const;
  Bool_t FreeSingleCluster(Int_t i, Int_t mode)    const;
  
  // methods to access single cluster information
  Int_t SetNumberOfSingleClustersSPD2(Int_t n) {return fNsingleSPD2 = n;}
  Int_t GetNumberOfSingleClusters() const {return fNsingle;}
  Int_t GetNumberOfSingleClustersLr(Int_t lr) const;
  Bool_t AreSPD2SinglesStored()         const {return TestBit(kSPD2Sng);}
  void   SetSPD2SinglesStored(Bool_t v=kTRUE) {return SetBit(kSPD2Sng,v);}

  Double_t GetThetaSingle(Int_t i) const { 
    if(i>=0 && i<fNsingle) return fThsingle[i];
    Error("GetThetaSingle","Invalid cluster number %d",i); return -9999.;
  }

  Double_t GetThetaSingleLr(Int_t i, Int_t lr) const { 
    if (lr==1) {
      if (!AreSPD2SinglesStored()) {Error("GetThetaSingle","Invalid cluster number %d for lr %d",i,lr); return -9999.;}
      else i += GetNumberOfSingleClustersLr(0);
    }
    if(i>=0 && i<fNsingle) return fThsingle[i];
    Error("GetThetaSingle","Invalid cluster number %d",i); return -9999.;
  }
  
  Double_t GetPhiSingle(Int_t i) const { 
    if(i>=0 && i<fNsingle) return fPhisingle[i];
    Error("GetPhisingle","Invalid cluster number %d",i); return -9999.;
  }
  
  Double_t GetPhiSingleLr(Int_t i, Int_t lr) const { 
    if (lr==1) {
      if (!AreSPD2SinglesStored()) {Error("GetPhiSingle","Invalid cluster number %d for lr %d",i,lr); return -9999.;}
      else i += GetNumberOfSingleClustersLr(0);
    }
    if(i>=0 && i<fNsingle) return fPhisingle[i];
    Error("GetPhisingle","Invalid cluster number %d",i); return -9999.;
  }

  UInt_t GetNumberOfITSClusters(Int_t layer) const { return layer<6 ? fITSClusters[layer] : 0; }
  UInt_t GetNumberOfITSClusters(Int_t layMin, Int_t layMax) const ;
  UInt_t GetNumberOfSPDClusters() const {return GetNumberOfITSClusters(0) + GetNumberOfITSClusters(1);}
  virtual void SetITSClusters(Int_t layer, UInt_t clusters) { fITSClusters[layer] = clusters; }
  //
  virtual Short_t GetNumberOfFiredChips(Int_t layer) const { return fFiredChips[layer]; }
  virtual void SetFiredChips(Int_t layer, Short_t firedChips) { fFiredChips[layer] = firedChips; }
  //
  virtual void   SetFastOrFiredChips(UInt_t chipKey){fFastOrFiredChips.SetBitNumber(chipKey);}
  virtual void   SetFastOrFiredChipMap(const TBits& fochips) {fFastOrFiredChips = fochips;}
  virtual const  TBits & GetFastOrFiredChips() const {return fFastOrFiredChips;}
  virtual Bool_t TestFastOrFiredChips(UInt_t chipKey) const {return fFastOrFiredChips.TestBitNumber(chipKey);}

  virtual void   SetFiredChipMap(const TBits & firedChips) {fClusterFiredChips = firedChips;}
  virtual void   SetFiredChipMap(UInt_t chipKey){fClusterFiredChips.SetBitNumber(chipKey);}
  virtual const  TBits & GetFiredChipMap() const {return fClusterFiredChips;}
  virtual Bool_t TestFiredChipMap(UInt_t chipKey) const {return fClusterFiredChips.TestBitNumber(chipKey);}

  Bool_t GetTrackletTrackIDs(Int_t i, Int_t mode, Int_t &spd1, Int_t &spd2) const;
  Int_t  GetTrackletTrackIDsLay(Int_t lr,Int_t i, Int_t mode, UInt_t* refs, UInt_t maxRef) const;
  Bool_t GetSingleClusterTrackID(Int_t i, Int_t mode, Int_t &tr) const;
  Int_t  GetSingleClusterTrackIDs(Int_t i, Int_t mode, UInt_t* refs, UInt_t maxRef) const;

  // array getters
  Double_t* GetDeltTheta()   const {return (Double_t*)fDeltTh;}
  Double_t* GetThetaSingle() const {return (Double_t*)fThsingle;}
  Double_t* GetPhiSingle()   const {return (Double_t*)fPhisingle;}
  Int_t*    GetLabelsSingle()      const {return (Int_t*)fLabelssingle;} 

  void AttachTracklet2TrackRefs(AliRefArray* l1t1,AliRefArray* l1t2,AliRefArray* l2t1,AliRefArray* l2t2) {
    fTCl2Tracks[0][0] = l1t1; fTCl2Tracks[0][1] = l1t2; fTCl2Tracks[1][0] = l2t1; fTCl2Tracks[1][1] = l2t2; 
  }
  void AttachCluster2TrackRefs(AliRefArray* l1t1,AliRefArray* l1t2) {
    fSCl2Tracks[0] = l1t1; fSCl2Tracks[1] = l1t2;
  }
  void SetTrackletData(Int_t id, const Float_t* tlet, UInt_t trSPD1=0, UInt_t trSPD2=0);
  void SetSingleClusterData(Int_t id, const Float_t* scl,UInt_t tr=0);
  void CompactBits();
  //
  void    SetDPhiWindow2(Float_t v=-1)            {fDPhiWindow2 = v;}
  void    SetDThetaWindow2(Float_t v=-1)          {fDThetaWindow2 = v;}
  void    SetDPhiShift(Float_t v=-1)              {fDPhiShift = v;}
  void    SetNStdDev(Float_t v=1)                 {fNStdDev = v;}
  //
  Float_t GetDPhiWindow2()                  const {return fDPhiWindow2;}
  Float_t GetDThetaWindow2()                const {return fDThetaWindow2;}
  Float_t GetDPhiShift()                    const {return fDPhiShift;}
  Float_t GetNStdDev()                      const {return fNStdDev;}

  virtual Float_t GetCentroidX() const {return fCentroidXY[0];}
  virtual Float_t GetCentroidY() const {return fCentroidXY[1];}
  virtual void SetCentroidXY(float x, float y) {
    fCentroidXY[0] = x;
    fCentroidXY[1] = y;
  }
  
  //
  protected:
  void Duplicate(const AliMultiplicity &m);  // used by copy ctr.

  Int_t fNtracks;            // Number of tracklets
  Int_t fNsingle;            // Number of clusters on SPD layer 1 and 2 (if storage of spd2 singles requested), not associated with a tracklet on otherSPD 
  Int_t fNsingleSPD2;        // Number of clusters on SPD layer 2 not associated (if stored)
  //
  Float_t fDPhiWindow2;      // sigma^2 in dphi used in reco
  Float_t fDThetaWindow2;    // sigma^2 in dtheta used in reco
  Float_t fDPhiShift;        // bending shift used
  Float_t fNStdDev;          // number of standard deviations kept
  //
  Int_t *fLabels;            //[fNtracks] array with labels of cluster in L1 used for tracklet
  Int_t *fLabelsL2;          //[fNtracks] array with labels of cluster in L2 used for tracklet
  UInt_t* fUsedClusS;        //[fNsingle] id+1 of the tracks using cluster, coded as (TPC/ITS+ITS_SA)+(ITS_SA_PURE<<16) !!! Outphased for multiple refs
  ULong64_t* fUsedClusT;     //[fNtracks] id+1 of the tracks using clusters, coded as (TPC/ITS+ITS_SA)+(ITS_SA_PURE<<16) for SPD1 and SPD2 in low and high parts
  AliRefArray *fTCl2Tracks[2][2]; // container with multiple tracklet_cluster->track references
  AliRefArray *fSCl2Tracks[2];    // container with multiple single_cluster->track references
  Double32_t *fTh;           //[fNtracks] array with theta values
  Double32_t *fPhi;          //[fNtracks] array with phi values
  Double32_t *fDeltTh;       //[fNtracks] array with delta theta values
  Double32_t *fDeltPhi;      //[fNtracks] array with delta phi values
  Double32_t *fThsingle;     //[fNsingle] array with theta values of L1 clusters
  Double32_t *fPhisingle;    //[fNsingle] array with phi values of L1 clusters
  Int_t *fLabelssingle;      //[fNsingle] array with labels of clusters in L1 not used for tracklets 
  Short_t fFiredChips[2];    // Number of fired chips in the two SPD layers
  UInt_t fITSClusters[6];    // Number of ITS cluster per layer
  TBits fFastOrFiredChips;   // Map of FastOr fired chips
  TBits fClusterFiredChips;  // Map of fired chips (= at least one cluster)
  Int_t fNtracksOnline;      //!Number of SPD tracklets set on the fly in online processing (HLT)
  Float_t fCentroidXY[2];    // tracklets centroid in X,Y 
  
  ClassDef(AliMultiplicity,21);
};

inline Int_t AliMultiplicity::GetLabel(Int_t i, Int_t layer) const
{
    if(i>=0 && i<fNtracks) {
	if (layer == 0) {
	    return fLabels[i];
	} else if (layer == 1) {
	    if (fLabelsL2) {
		return fLabelsL2[i];
	    } else {
		Warning("GetLabel", "No information for layer 2 available !");
		return -9999;
	    }
	} else {
	    Error("GetLabel","Invalid layer number %d",layer); return -9999;
	}
    } else {
	Error("GetLabel","Invalid track number %d",i); return -9999;
    }
    return -9999;
}

inline Int_t AliMultiplicity::GetLabelSingle(Int_t i) const 
{
    if(i>=0 && i<fNsingle) {
      return fLabelssingle[i];
    } else {
        Error("GetLabelSingle","Invalid cluster number %d",i); return -9999;
    }
    return -9999;
}


inline Double_t AliMultiplicity::CalcDist(Int_t i) const
{
  // calculate eliptical distance. theta is the angle of cl1, dtheta = tht(cl1)-tht(cl2)
  if (i<0 && i>=fNtracks) return -1;
  if (fDPhiWindow2<1E-9 || fDThetaWindow2<1E-9) return -1; // not stored
  double dphi   = TMath::Abs(fDeltPhi[i]) - fDPhiShift;
  double dtheta = fDeltTh[i];
  if (GetScaleDThetaBySin2T()) {
    double sinTI = TMath::Sin(fTh[i]-dtheta/2);
    sinTI *= sinTI;
    dtheta /= sinTI>1.e-6 ? sinTI : 1.e-6;
  }
  return dphi*dphi/fDPhiWindow2 + dtheta*dtheta/fDThetaWindow2;
}

inline Int_t AliMultiplicity::GetNumberOfSingleClustersLr(Int_t lr) const
{
  // return number of singles at given layer
  if (lr==0) return fNsingle - fNsingleSPD2;
  return AreSPD2SinglesStored() ? fNsingleSPD2 : -1;
}



#endif
