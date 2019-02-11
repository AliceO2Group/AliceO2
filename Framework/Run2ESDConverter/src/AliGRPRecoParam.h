#ifndef ALIGRPRECOPARAM_H
#define ALIGRPRECOPARAM_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Class with global reconstruction parameters                               //
// (initially, parameters for AliVertexerTracks)                             //
// Origin: andrea.dainese@lnl.infn.it                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


#include "AliDetectorRecoParam.h"
#include "TObjArray.h"
class AliV0HypSel;

class AliGRPRecoParam : public AliDetectorRecoParam
{
 public: 
  AliGRPRecoParam();
  virtual ~AliGRPRecoParam();

  static AliGRPRecoParam *GetLowFluxParam();// make reco parameters for low flux env.
  static AliGRPRecoParam *GetHighFluxParam();// make reco parameters for high flux env. 
  static AliGRPRecoParam *GetCosmicTestParam();// make reco parameters for cosmics env. 

  void  SetMostProbablePt(Double_t pt=0.350) { fMostProbablePt=pt; return; }
  Double_t GetMostProbablePt() const { return fMostProbablePt; }

  void  SetVertexerTracksConstraintITS(Bool_t constr=kTRUE) { fVertexerTracksConstraintITS=constr; return; }
  void  SetVertexerTracksConstraintTPC(Bool_t constr=kTRUE) { fVertexerTracksConstraintTPC=constr; return; }
  void  SetVertexerTracksCuts(Int_t mode,Int_t ncuts,Double_t* cuts);
  void  SetVertexerTracksCutsITS(Int_t ncuts,Double_t* cuts)
    { SetVertexerTracksCuts(0,ncuts,cuts); return; }
  void  SetVertexerTracksCutsTPC(Int_t ncuts,Double_t* cuts)
    { SetVertexerTracksCuts(1,ncuts,cuts); return; }
  void  SetVertexerV0Cuts(Int_t ncuts,Double_t cuts[7]);
  void  SetVertexerCascadeCuts(Int_t ncuts,Double_t cuts[8]);
  void  SetVertexerTracksTPCClusterization(Bool_t use, Double_t dzcut, Double_t nsigmazcut);
  void  SetVertexerTracksITSClusterization(Bool_t use, Double_t dzcut, Double_t nsigmazcut);
  Bool_t GetVertexerTracksConstraintITS() const { return fVertexerTracksConstraintITS; }
  Bool_t GetVertexerTracksConstraintTPC() const { return fVertexerTracksConstraintTPC; }
  Int_t GetVertexerTracksNCuts() const { return fVertexerTracksNCuts; }
  Int_t GetVertexerV0NCuts() const { return fVertexerV0NCuts; }
  Double_t GetVertexerV0EtaMax() const { return fVertexerV0EtaMax; }
  void  SetVertexerV0EtaMax(double eta)  { fVertexerV0EtaMax = eta; }
  Bool_t GetCleanOfflineV0Prongs() const { return fCleanOfflineV0Prongs;}
  void  SetCleanOfflineV0Prongs(Bool_t v) { fCleanOfflineV0Prongs = v;}
  Int_t GetVertexerCascadeNCuts() const { return fVertexerCascadeNCuts; }
  void  GetVertexerTracksCuts(Int_t mode,Double_t *cuts,Int_t n) const;
  void  GetVertexerTracksCutsITS(Double_t *cuts,Int_t n) const
  { GetVertexerTracksCuts(0,cuts,n); return; }
  void  GetVertexerTracksCutsTPC(Double_t *cuts,Int_t n) const
  { GetVertexerTracksCuts(1,cuts,n); return; }
  void  GetVertexerV0Cuts(Double_t *cuts) const;
  void  GetVertexerCascadeCuts(Double_t *cuts) const;

  ULong_t GetFlagsNotToClean() const {return fFlagsNotToClean;}
  void    SetFlagsNotToClean(ULong_t v) {fFlagsNotToClean = v;}

  Double_t GetCleanDCAZCut() const {return fCleanDCAZCut;}
  void    SetCleanDCAZCut(double v) {fCleanDCAZCut = v;}
  
  const TObjArray* GetV0HypSelArray() const {return &fV0HypSelArray;}
  void  AddV0HypSel(const AliV0HypSel& sel);
  
  AliGRPRecoParam(const AliGRPRecoParam&);
  AliGRPRecoParam& operator=(const AliGRPRecoParam&);

 protected:
  //

  Double_t fMostProbablePt; // to be used for B=0 tracking
  Bool_t   fVertexerTracksConstraintITS; // diamond constr for AliVertexerTracks
  Bool_t   fVertexerTracksConstraintTPC; // diamond constr for AliVertexerTracks
  Int_t    fVertexerTracksNCuts;  // number of cuts for AliVertexerTracks

  // cuts for AliVertexerTracks: ITS mode
  Double_t fVertexerTracksITSdcacut; // general dca
  Double_t fVertexerTracksITSdcacutIter0; // dca in iteration 0
  Double_t fVertexerTracksITSmaxd0z0; // max d0z0
  Double_t fVertexerTracksITSminCls; // min clusters
  Double_t fVertexerTracksITSmintrks; // min tracks
  Double_t fVertexerTracksITSnsigma; // n sigma for d0 cut
  Double_t fVertexerTracksITSnindetfitter; // min det to try inversion
  Double_t fVertexerTracksITSmaxtgl; // max tgl 
  Double_t fVertexerTracksITSfidR; // fiducial radius
  Double_t fVertexerTracksITSfidZ; // fiducial z
  Double_t fVertexerTracksITSalgo; // finder algo
  Double_t fVertexerTracksITSalgoIter0; // finder algo iteration 0
  //
  Double_t fVertexerTracksITSMVTukey2;          // Tukey constant for multivertexer
  Double_t fVertexerTracksITSMVSig2Ini;         // initial sig2 for multivertexer
  Double_t fVertexerTracksITSMVMaxSigma2;       // max sig2 to accept for multivertexer
  Double_t fVertexerTracksITSMVMinSig2Red;      // min sig2 to to consider multivertexer stuck (then push)
  Double_t fVertexerTracksITSMVMinDst;          // min distance between 2 iterations to stop multi-vertex search
  Double_t fVertexerTracksITSMVScanStep;        // z-scan step for multivertexer
  Double_t fVertexerTracksITSMVMaxWghNtr;       // min wdist*ncontrib between to vertices to eliminate
  Double_t fVertexerTracksITSMVFinalWBinary;    // for the final fit used binary weights
  Double_t fVertexerTracksITSMVBCSpacing;       // assumer BC spacing
  //
  Double_t fVertexerTracksITSclusterize; // pre-clusterization of tracks
  Double_t fVertexerTracksITSclusterdz;  // cut in absolute dz
  Double_t fVertexerTracksITSclusternsigmaz;  // cut in standardized dz

  // cuts for AliVertexerTracks: TPC-only mode
  Double_t fVertexerTracksTPCdcacut; // general dca
  Double_t fVertexerTracksTPCdcacutIter0; // dca in iteration 0
  Double_t fVertexerTracksTPCmaxd0z0; // max d0z0
  Double_t fVertexerTracksTPCminCls; // min clusters
  Double_t fVertexerTracksTPCmintrks; // min tracks
  Double_t fVertexerTracksTPCnsigma; // n sigma for d0 cut
  Double_t fVertexerTracksTPCnindetfitter; // min det to try inversion
  Double_t fVertexerTracksTPCmaxtgl; // max tgl 
  Double_t fVertexerTracksTPCfidR; // fiducial radius
  Double_t fVertexerTracksTPCfidZ; // fiducial z
  Double_t fVertexerTracksTPCalgo; // finder algo
  Double_t fVertexerTracksTPCalgoIter0; // finder algo iteration 0
  //
  Double_t fVertexerTracksTPCMVTukey2;          // Tukey constant for multivertexer
  Double_t fVertexerTracksTPCMVSig2Ini;         // initial sig2 for multivertexer
  Double_t fVertexerTracksTPCMVMaxSigma2;       // max sig2 to accept for multivertexer
  Double_t fVertexerTracksTPCMVMinSig2Red;      // min sig2 to to consider multivertexer stuck (then push)
  Double_t fVertexerTracksTPCMVMinDst;          // min distance between 2 iterations to stop multi-vertex search
  Double_t fVertexerTracksTPCMVScanStep;        // z-scan step for multivertexer
  Double_t fVertexerTracksTPCMVMaxWghNtr;       // min wdist*ncontrib between to vertices to eliminate
  Double_t fVertexerTracksTPCMVFinalWBinary;    // for the final fit used binary weights
  Double_t fVertexerTracksTPCMVBCSpacing;       // assumer BC spacing
  //
  Double_t fVertexerTracksTPCclusterize; // pre-clusterization of tracks
  Double_t fVertexerTracksTPCclusterdz;  // cut in absolute dz
  Double_t fVertexerTracksTPCclusternsigmaz;  // cut in standardized dz
  //
  Int_t    fVertexerV0NCuts;      // number of cuts for AliV0vertexer

  // cuts for AliV0vertexer:
  Double_t fVertexerV0Chi2max; //max chi2
  Double_t fVertexerV0DNmin;   //min imp parameter for the 1st daughter
  Double_t fVertexerV0DPmin;   //min imp parameter for the 2nd daughter
  Double_t fVertexerV0DCAmax;  //max DCA between the daughter tracks
  Double_t fVertexerV0CPAmin;  //min cosine of V0's pointing angle
  Double_t fVertexerV0Rmin;    //min radius of the fiducial volume
  Double_t fVertexerV0Rmax;    //max radius of the fiducial volume
  Double_t fVertexerV0EtaMax;  //max eta
  Bool_t   fCleanOfflineV0Prongs; // compress redundant prongs by filling 0s
  Int_t    fVertexerCascadeNCuts; // number of cuts for AliCascadeVertexer

  // cuts for AliCascadeVertexer:
  Double_t fVertexerCascadeChi2max;  //maximal allowed chi2 
  Double_t fVertexerCascadeDV0min;   //min V0 impact parameter
  Double_t fVertexerCascadeMassWin;  //"window" around the Lambda mass
  Double_t fVertexerCascadeDBachMin; //min bachelor impact parameter
  Double_t fVertexerCascadeDCAmax;   //max DCA between the V0 and the track 
  Double_t fVertexerCascadeCPAmin;   //min cosine of the cascade pointing angle
  Double_t fVertexerCascadeRmin;     //min radius of the fiducial volume
  Double_t fVertexerCascadeRmax;     //max radius of the fiducial volume
  Double_t fCleanDCAZCut;            // consider track removal if its DCA exceeds this
  ULong_t fFlagsNotToClean;         // tracks with these flags are kept in CleanESD  
  TObjArray fV0HypSelArray;         // array of V0 hypotheses selection


  
  ClassDef(AliGRPRecoParam,8) // global reco parameters
};

inline void  AliGRPRecoParam::SetVertexerTracksTPCClusterization(Bool_t use, Double_t dzcut, Double_t nsigmazcut)
{
  // set TPC tracks clustering settings
  if(use) fVertexerTracksTPCclusterize=1.;
  else fVertexerTracksTPCclusterize=0.;
  fVertexerTracksTPCclusterdz=dzcut;
  fVertexerTracksTPCclusternsigmaz=nsigmazcut;
}

inline  void  AliGRPRecoParam::SetVertexerTracksITSClusterization(Bool_t use, Double_t dzcut, Double_t nsigmazcut)
{
  // set ITS tracks clustering settings
  if(use) fVertexerTracksITSclusterize=1.;
  else fVertexerTracksITSclusterize=0.;
  fVertexerTracksITSclusterdz=dzcut;
  fVertexerTracksITSclusternsigmaz=nsigmazcut;
}


#endif
