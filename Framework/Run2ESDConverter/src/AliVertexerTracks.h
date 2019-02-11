#ifndef ALIVERTEXERTRACKS_H
#define ALIVERTEXERTRACKS_H
/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


//-------------------------------------------------------
// Class for vertex determination with tracks
//
//   Origin: AliITSVertexerTracks  
//           A.Dainese, Padova, andrea.dainese@pd.infn.it
//           M.Masera,  Torino, massimo.masera@to.infn.it 
//   Moved to STEER and adapted to ESD tracks: 
//           F.Prino, Torino, prino@to.infn.it 
//-------------------------------------------------------

/*****************************************************************************
 *                                                                           *
 * This class determines the vertex of a set of tracks.                      *
 * Different algorithms are implemented, see data member fAlgo.              *
 *                                                                           *
 *****************************************************************************/

#include <TObjArray.h>
#include <TMatrixD.h>

#include "AliLog.h"
#include "AliESDVertex.h"


class AliExternalTrackParam;
class AliVEvent;
class AliESDEvent;
class AliStrLine;

class AliVertexerTracks : public TObject {
  
 public:
  enum {kTOFBCShift=200, 
	kStrLinVertexFinderMinDist1=1,
	kStrLinVertexFinderMinDist0=2,
	kHelixVertexFinder=3,
	kVertexFinder1=4,
	kVertexFinder0=5,
	kMultiVertexer=6};
  enum {kBitUsed = BIT(16),kBitAccounted = BIT(17)};
  AliVertexerTracks(); 
  AliVertexerTracks(Double_t fieldkG); 
  virtual ~AliVertexerTracks();

  AliESDVertex* FindPrimaryVertex(const AliVEvent *vEvent);
  AliESDVertex* FindPrimaryVertex(const TObjArray *trkArrayOrig,UShort_t *idOrig);
  AliESDVertex* VertexForSelectedTracks(const TObjArray *trkArray,UShort_t *id,
					Bool_t optUseFitter=kTRUE,
					Bool_t optPropagate=kTRUE,
					Bool_t optUseDiamondConstraint=kFALSE);
  AliESDVertex* VertexForSelectedESDTracks(TObjArray *trkArray,
					Bool_t optUseFitter=kTRUE,
					Bool_t optPropagate=kTRUE,
					Bool_t optUseDiamondConstraint=kFALSE);
  AliESDVertex* RemoveTracksFromVertex(AliESDVertex *inVtx,
				       const TObjArray *trkArray,UShort_t *id,
				       const Float_t *diamondxy) const; 
  AliESDVertex* RemoveConstraintFromVertex(AliESDVertex *inVtx,
					   Float_t *diamondxyz,
					   Float_t *diamondcov) const;
  void  SetITSMode(Double_t dcacut=0.1,
		   Double_t dcacutIter0=0.1,
		   Double_t maxd0z0=0.5,
		   Int_t minCls=3,
		   Int_t mintrks=1,
		   Double_t nsigma=3.,
		   Double_t mindetfitter=100.,
		   Double_t maxtgl=1000.,
		   Double_t fidR=3.,
		   Double_t fidZ=30.,
		   Int_t finderAlgo=1,
		   Int_t finderAlgoIter0=4); 
  void  SetTPCMode(Double_t dcacut=0.1,
		   Double_t dcacutIter0=1.0,
		   Double_t maxd0z0=5.0,
		   Int_t minCls=10,
		   Int_t mintrks=1,
		   Double_t nsigma=3.,
		   Double_t mindetfitter=0.1,
		   Double_t maxtgl=1.5, 
		   Double_t fidR=3.,
		   Double_t fidZ=30.,
		   Int_t finderAlgo=1,
		   Int_t finderAlgoIter0=4); 
  void  SetCuts(Double_t *cuts, int ncuts);
  void  SetConstraintOff() { fConstraint=kFALSE; SetVtxStart(); SetVtxStartSigma(); return; }
  void  SetConstraintOn() { fConstraint=kTRUE; return; }
  void  SetDCAcut(Double_t maxdca) { fDCAcut=maxdca; return; }
  void  SetDCAcutIter0(Double_t maxdca) { fDCAcutIter0=maxdca; return; }
  void  SetFinderAlgorithm(Int_t opt=1) { fAlgo=opt; return; }
  void  SetITSrefitRequired() { fITSrefit=kTRUE; return; }
  void  SetITSpureSA(Bool_t v=kTRUE) { fITSpureSA=v; return; }
  Bool_t GetITSpureSA() { return fITSpureSA; }
  Bool_t GetITSrefitRequired() const { return fITSrefit; }
  void  SetITSrefitNotRequired() { fITSrefit=kFALSE; return; }
  void  SetFiducialRZ(Double_t r=3,Double_t z=30) { fFiducialR=r; fFiducialZ=z; return; }
  void  SetMaxd0z0(Double_t maxd0z0=0.5) { fMaxd0z0=maxd0z0; return; }
  void  SetMinClusters(Int_t n=5) { fMinClusters=n; return; }
  Int_t GetMinClusters() const { return fMinClusters; }
  void  SetMinTracks(Int_t n=1) { fMinTracks=n; return; }
  void  SetNSigmad0(Double_t n=3) { fNSigma=n; return; }
  Double_t GetNSigmad0() const { return fNSigma; }
  void  SetMinDetFitter(Double_t mindet=100.) { fMinDetFitter=mindet; return; }
  void  SetMaxTgl(Double_t maxtgl=1.) { fMaxTgl=maxtgl; return; }
  void  SetOnlyFitter() { if(!fConstraint) AliFatal("Set constraint first!"); 
     fOnlyFitter=kTRUE; return; }
  void  SetSkipTracks(Int_t n,const Int_t *skipped);
  void  SetVtxStart(Double_t x=0,Double_t y=0,Double_t z=0) 
    { fNominalPos[0]=x; fNominalPos[1]=y; fNominalPos[2]=z; return; }
  void  SetVtxStartSigma(Double_t sx=3.,Double_t sy=3.,Double_t sz=15.) 
    { fNominalCov[0]=sx*sx; fNominalCov[2]=sy*sy; fNominalCov[5]=sz*sz;
      fNominalCov[1]=0.; fNominalCov[3]=0.; fNominalCov[4]=0.; return; }
  void  SetVtxStart(AliESDVertex *vtx);
  void  SetSelectOnTOFBunchCrossing(Bool_t select=kFALSE,Bool_t keepAlsoUnflagged=kTRUE) {fSelectOnTOFBunchCrossing=select; fKeepAlsoUnflaggedTOFBunchCrossing=keepAlsoUnflagged; return;}
  //
  static Double_t GetStrLinMinDist(const Double_t *p0,const Double_t *p1,const Double_t *x0);
  static Double_t GetDeterminant3X3(Double_t matr[][3]);
  static void GetStrLinDerivMatrix(const Double_t *p0,const Double_t *p1,Double_t (*m)[3],Double_t *d);
  static void GetStrLinDerivMatrix(const Double_t *p0,const Double_t *p1,const Double_t *sigmasq,Double_t (*m)[3],Double_t *d);
  static AliESDVertex TrackletVertexFinder(const TClonesArray *lines, Int_t optUseWeights=0);
  static AliESDVertex TrackletVertexFinder(AliStrLine **lines, const Int_t knacc, Int_t optUseWeights=0);
  void     SetFieldkG(Double_t field=-999.) { fFieldkG=field; return; }
  Double_t GetFieldkG() const { 
    if(fFieldkG<-99.) AliFatal("Field value not set");
    return fFieldkG; } 
  void SetNSigmaForUi00(Double_t n=1.5) { fnSigmaForUi00=n; return; }
  Double_t GetNSigmaForUi00() const { return fnSigmaForUi00; }
  //
  void SetMVTukey2(double t=6)             {fMVTukey2     = t;}
  void SetMVSig2Ini(double t=1e3)          {fMVSig2Ini    = t;}
  void SetMVMaxSigma2(double t=3.)         {fMVMaxSigma2  = t;}
  void SetMVMinSig2Red(double t=0.005)     {fMVMinSig2Red = t;}
  void SetMVMinDst(double t=10e-4)         {fMVMinDst     = t;}
  void SetMVScanStep(double t=2.)          {fMVScanStep   = t;}
  void SetMVFinalWBinary(Bool_t v=kTRUE)   {fMVFinalWBinary = v;}
  void SetMVMaxWghNtr(double w=10.)        {fMVMaxWghNtr  = w;}
  //
  void   FindVerticesMV();
  Bool_t FindNextVertexMV();
  //
  AliESDVertex* GetCurrentVertex() const {return (AliESDVertex*)fCurrentVertex;}
  TObjArray*    GetVerticesArray() const {return (TObjArray*)fMVVertices;}   // RS to be removed
  void          AnalyzePileUp(AliESDEvent* esdEv);
  void          SetBCSpacing(Int_t ns=50) {fBCSpacing = ns;}

  // Configuration of multi-vertexing vis pre-clusterization of tracks
  void SetUseTrackClusterization(Bool_t opt=kFALSE){fClusterize=opt;}
  void SetDeltaZCutForCluster(Double_t cut){fDeltaZCutForCluster=cut;}
  void SetnSigmaZCutForCluster(Double_t cut){fnSigmaZCutForCluster=cut;}
  void SetDisableBCInCPass0(Bool_t v=kTRUE) {fDisableBCInCPass0 = v;}

  Bool_t GetDisableBCInCPass0()      const {return fDisableBCInCPass0;}
  Bool_t GetUseTrackClusterization() const {return fClusterize;}
  Double_t GetDeltaZCutForCluster() const {return fDeltaZCutForCluster;}
  Double_t GetnSigmaZCutForCluster() const {return fnSigmaZCutForCluster;}


  //
 protected:
  void     HelixVertexFinder();
  void     OneTrackVertFinder();
  Int_t    PrepareTracks(const TObjArray &trkArrayOrig,const UShort_t *idOrig,
			 Int_t optImpParCut);
  Bool_t   PropagateTrackTo(AliExternalTrackParam *track,
			    Double_t xToGo);
  Bool_t   TrackToPoint(AliExternalTrackParam *t,
		        TMatrixD &ri,TMatrixD &wWi,
			Bool_t uUi3by3=kFALSE) const;     
  void     VertexFinder(Int_t optUseWeights=0);
  void     VertexFitter(Bool_t vfit=kTRUE, Bool_t chiCalc=kTRUE,Int_t useWeights=0);
  void     StrLinVertexFinderMinDist(Int_t optUseWeights=0);
  void     TooFewTracks();

  void     FindAllVertices(Int_t nTrksOrig, const TObjArray *trkArrayOrig, Double_t* zTr, Double_t* err2zTr, UShort_t* idOrig);

  AliESDVertex fVert;         // vertex after vertex finder
  AliESDVertex *fCurrentVertex;  // ESD vertex after fitter
  UShort_t  fMode;            // 0 ITS+TPC; 1 TPC
  Double_t  fFieldkG;         // z component of field (kGauss) 
  Double_t  fNominalPos[3];   // initial knowledge on vertex position
  Double_t  fNominalCov[6];   // initial knowledge on vertex position
  TObjArray fTrkArraySel;     // array with tracks to be processed
  UShort_t  *fIdSel;          //! IDs of the tracks (AliESDtrack::GetID())
  Int_t     *fTrksToSkip;     //! track IDs to be skipped for find and fit 
  Int_t     fNTrksToSkip;     // number of tracks to be skipped 
  Bool_t    fConstraint;      // true when "mean vertex" was set in 
                              // fNominal ... and must be used in the fit
  Bool_t    fOnlyFitter;      // primary with one fitter shot only
                              // (use only with beam constraint)
  Int_t     fMinTracks;       // minimum number of tracks
  Int_t     fMinClusters;     // minimum number of ITS or TPC clusters per track
  Double_t  fDCAcut;          // maximum DCA between 2 tracks used for vertex
  Double_t  fDCAcutIter0;     // maximum DCA between 2 tracks used for vertex
  Double_t  fNSigma;          // number of sigmas for d0 cut in PrepareTracks()
  Double_t  fMaxd0z0;         // value for sqrt(d0d0+z0z0) cut 
                              // in PrepareTracks(1) if fConstraint=kFALSE
  Double_t  fMinDetFitter;    // minimum determinant to try to invertex matrix
  Double_t  fMaxTgl;          // maximum tgl of tracks
  Bool_t    fITSrefit;        // if kTRUE (default), use only kITSrefit tracks
                              // if kFALSE, use all tracks (also TPC only)
  Bool_t    fITSpureSA;       // if kFALSE (default) skip ITSpureSA tracks
                              // if kTRUE use only those
  Double_t  fFiducialR;       // radius of fiducial cylinder for tracks 
  Double_t  fFiducialZ;       // length of fiducial cylinder for tracks
  Double_t  fnSigmaForUi00;   // n. sigmas from finder in TrackToPoint
  Int_t     fAlgo;            // option for vertex finding algorythm
  Int_t     fAlgoIter0;       // this is for iteration 0  
  // fAlgo=1 (default) finds minimum-distance point among all selected tracks
  //         approximated as straight lines 
  //         and uses errors on track parameters as weights
  // fAlgo=2 finds minimum-distance point among all the selected tracks
  //         approximated as straight lines 
  // fAlgo=3 finds the average point among DCA points of all pairs of tracks
  //         treated as helices
  // fAlgo=4 finds the average point among DCA points of all pairs of tracks
  //         approximated as straight lines 
  //         and uses errors on track parameters as weights
  // fAlgo=5 finds the average point among DCA points of all pairs of tracks
  //         approximated as straight lines 
  //
  Bool_t    fSelectOnTOFBunchCrossing;  // tracks from bunch crossing 0 
  Bool_t    fKeepAlsoUnflaggedTOFBunchCrossing; // also tracks w/o bunch crossing number (-1)
  // parameters for multivertexer
  Double_t fMVWSum;                    // sum of weights for multivertexer
  Double_t fMVWE2;                     // sum of weighted chi2's for  multivertexer
  Double_t fMVTukey2;                  // Tukey constant for multivertexer
  Double_t fMVSigma2;                  // chi2 current scaling param for multivertexer
  Double_t fMVSig2Ini;                 // initial value for fMVSigma2
  Double_t fMVMaxSigma2;               // max acceptable value for final fMVSigma2
  Double_t fMVMinSig2Red;              // min reduction of fMVSigma2 to exit the loop
  Double_t fMVMinDst;                  // min distance between vertices at two iterations to exit
  Double_t fMVScanStep;                // step of vertices scan
  Double_t fMVMaxWghNtr;               // min W-distance*Ncontr_min for close vertices
  Bool_t   fMVFinalWBinary;            // for the final fit use binary weights
  Int_t    fBCSpacing;                 // BC Spacing in ns (will define the rounding of BCid)
  TObjArray* fMVVertices;              // array of found vertices

  Bool_t   fDisableBCInCPass0;         // do not use BC from TOF in CPass0
  Bool_t   fClusterize;                // flag to activate track clusterization into vertices before vertex finder
  Double_t fDeltaZCutForCluster;       // minimum distance in z between tracks to create new cluster
  Double_t fnSigmaZCutForCluster;      // minimum distacnce in number of sigma along z to create new cluster
  //
 private:
  AliVertexerTracks(const AliVertexerTracks & source);
  AliVertexerTracks & operator=(const AliVertexerTracks & source);

  ClassDef(AliVertexerTracks,18) // 3D Vertexing with tracks 
};

#endif

