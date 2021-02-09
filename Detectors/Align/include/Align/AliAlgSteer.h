#ifndef ALIALGSTEER_H
#define ALIALGSTEER_H

#include "AliGeomManager.h"
#include "AliAlgTrack.h"
#include "AliSymMatrix.h"

#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <TString.h>
#include <TArrayF.h>
#include <TArrayI.h>
#include <TH1F.h>
#include "AliAlgAux.h"

class AliESDEvent;
class AliESDtrack;
class AliESDCosmicTrack;
class AliESDVertex;
class AliAlgDet;
class AliAlgVol;
class AliAlgVtx;
class AliAlgPoint;
class AliAlgMPRecord;
class AliAlgRes;
class AliAlgResFast;
class AliAlgConstraint;
class AliAlgDOFStat;
class TTree;
class TFile;
//
class Mille;


/*--------------------------------------------------------
  Steering class for the global alignment. Responsible for feeding the track data 
  to participating detectors and preparation of the millepede input.
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgSteer : public TObject
{
 public:
  enum {kNLrSkip=4};
  enum {kITS,kTPC,kTRD,kTOF,kHMPID,kNDetectors, kUndefined};
  enum {kCosmLow,kCosmUp,kNCosmLegs};
  enum {kInpStat,kAccStat,kNStatCl};
  enum {kRun,kEventColl,kEventCosm,kTrackColl,kTrackCosm, kMaxStat};
  enum MPOut_t {kMille=BIT(0),kMPRec=BIT(1),kContR=BIT(2)};
  enum {kInitGeomDone=BIT(14),kInitDOFsDone=BIT(15),kMPAlignDone=BIT(16)};
  //
  enum {             // STAT histo entries
    kRunDone         // input runs
    ,kEvInp          // input events
    ,kEvVtx          // after vtx selection
    ,kTrackInp       // input tracks
    ,kTrackFitInp    // input to ini fit
    ,kTrackFitInpVC  // those with vertex constraint
    ,kTrackProcMatInp// input to process materials
    ,kTrackResDerInp // input to resid/deriv calculation
    ,kTrackStore     // stored tracks
    ,kTrackAcc       // tracks accepted
    ,kTrackControl   // control tracks filled
    //
    ,kNHVars
  };

  //
  AliAlgSteer(const char* configMacro=0, int refRun=-1);
  virtual ~AliAlgSteer();
  Bool_t   LoadRefOCDB();
  Bool_t   LoadRecoTimeOCDB();
  Bool_t   GetUseRecoOCDB()               const {return fUseRecoOCDB;}
  void     SetUseRecoOCDB(Bool_t v=kTRUE)       {fUseRecoOCDB=v;}

  void     InitDetectors();
  void     InitDOFs();  
  void     Terminate(Bool_t dostat=kTRUE);
  void     SetStatHistoLabels(TH1* h)                     const;
  //
  void     SetInitGeomDone()                                    {SetBit(kInitGeomDone);}
  Bool_t   GetInitGeomDone()                              const {return TestBit(kInitGeomDone);}
  //
  void     SetInitDOFsDone()                                    {SetBit(kInitDOFsDone);}
  Bool_t   GetInitDOFsDone()                              const {return TestBit(kInitDOFsDone);}
  //
  void     SetMPAlignDone()                                     {SetBit(kMPAlignDone);}
  Bool_t   GetMPAlignDone()                               const {return TestBit(kMPAlignDone);}

  void     AssignDOFs();
  //
  void     AddDetector(UInt_t id, AliAlgDet* det=0);
  void     AddDetector(AliAlgDet* det);
  //
  void     AddConstraint(const AliAlgConstraint* cs)            {fConstraints.AddLast((TObject*)cs);}
  Int_t    GetNConstraints()                              const {return fConstraints.GetEntriesFast();}
  const    TObjArray*        GetConstraints()             const {return &fConstraints;}
  const    AliAlgConstraint* GetConstraint(int i)         const {return (AliAlgConstraint*)fConstraints[i];}
  void     AddAutoConstraints();
  //
  void     AcknowledgeNewRun(Int_t run);
  void     SetRunNumber(Int_t run);
  Int_t    GetRunNumber()                                 const {return fRunNumber;}
  Bool_t   GetFieldOn()                                   const {return fFieldOn;}
  void     SetFieldOn(Bool_t v=kTRUE) {fFieldOn = v;}
  Int_t    GetTracksType()                                const {return fTracksType;}
  void     SetTracksType(Int_t t=AliAlgAux::kColl)              {fTracksType = t;}
  Bool_t   IsCosmic()                                     const {return fTracksType==AliAlgAux::kCosm;}
  Bool_t   IsCollision()                                  const {return fTracksType==AliAlgAux::kColl;}
  void     SetCosmic(Bool_t v=kTRUE)                            {fTracksType = v ? AliAlgAux::kCosm : AliAlgAux::kColl;}
  Float_t  GetStat(int cls, int tp)                       const {return fStat[cls][tp];}
  //
  void     SetESDTree(const TTree* tr)                          {fESDTree = tr;}
  const    TTree* GetESDTree()                            const {return fESDTree;}
  void     SetESDEvent(const AliESDEvent* ev);
  const    AliESDEvent* GetESDEvent()                     const {return fESDEvent;}
  void     SetESDtrack(const AliESDtrack* tr, int i=0)          {fESDTrack[i] = tr;}
  const    AliESDtrack* GetESDtrack(int i=0)              const {return fESDTrack[i];}
  //
  // Track selection
  void     SetCosmicSelStrict(Bool_t v=kTRUE)                   {fCosmicSelStrict = v;}
  Bool_t   GetCosmicSelStrict()                           const {return fCosmicSelStrict;}
  //  
  Int_t    GetMinPoints()                                 const {return fMinPoints[fTracksType][GetFieldOn()];}
  Int_t    GetMinPoints(Bool_t tp,Bool_t bON)             const {return fMinPoints[tp][bON];}
  void     SetMinPoints(Bool_t tp,Bool_t bON,int n)             {int mn=bON?4:3; fMinPoints[tp][bON]=n>mn?n:mn;}
  void     SetMinPointsColl(int vbOff=3,int vbOn=4);
  void     SetMinPointsCosm(int vbOff=3,int vbOn=4);
  //
  Double_t GetPtMin(Bool_t tp)                            const {return fPtMin[tp];}
  void     SetPtMin(Bool_t tp,double pt)                        {fPtMin[tp] = pt;}
  void     SetPtMinColl(double pt=0.7)                          {SetPtMin(AliAlgAux::kColl,pt);}
  void     SetPtMinCosm(double pt=1.0)                          {SetPtMin(AliAlgAux::kCosm,pt);}
  //
  Double_t GetEtaMax(Bool_t tp)                           const {return fEtaMax[tp];}
  void     SetEtaMax(Bool_t tp,double eta)                      {fEtaMax[tp]=eta;}
  void     SetEtaMaxColl(double eta=1.5)                        {SetEtaMax(AliAlgAux::kColl,eta);}
  void     SetEtaMaxCosm(double eta=1.5)                        {SetEtaMax(AliAlgAux::kCosm,eta);}
  //
  void     SetDefPtBOffCosm(double pt=5.0)                      {fDefPtBOff[AliAlgAux::kCosm] = pt>0.3 ? pt:0.3;}
  void     SetDefPtBOffColl(double pt=0.6)                      {fDefPtBOff[AliAlgAux::kColl] = pt>0.3 ? pt:0.3;}
  Double_t GetDefPtBOff(Bool_t tp)                              {return fDefPtBOff[tp];}
  //
  Int_t    GetMinDetAcc(Bool_t tp)                        const {return fMinDetAcc[tp];}
  void     SetMinDetAcc(Bool_t tp, int n)                       {fMinDetAcc[tp] = n;}
  void     SetMinDetAccColl(int n=1)                            {SetMinDetAcc(AliAlgAux::kColl,n);}
  void     SetMinDetAccCosm(int n=1)                            {SetMinDetAcc(AliAlgAux::kCosm,n);}
  //
  Int_t    GetVtxMinCont()                                const {return fVtxMinCont;}
  void     SetVtxMinCont(int n)                                 {fVtxMinCont = n;}
  Int_t    GetVtxMaxCont()                                const {return fVtxMaxCont;}
  void     SetVtxMaxCont(int n)                                 {fVtxMaxCont = n;}
  Int_t    GetVtxMinContVC()                              const {return fVtxMinContVC;}
  void     SetVtxMinContVC(int n)                               {fVtxMinContVC = n;}
  //
  Int_t    GetMinITSClforVC()                             const {return fMinITSClforVC;}
  void     SetMinITSClforVC(int n)                              {fMinITSClforVC = n;}
  Int_t    GetITSPattforVC()                              const {return fITSPattforVC;}
  void     SetITSPattforVC(int p)                               {fITSPattforVC=p;}
  Double_t GetMaxDCARforVC()                              const {return fMaxDCAforVC[0];}
  Double_t GetMaxDCAZforVC()                              const {return fMaxDCAforVC[1];}
  void     SetMaxDCAforVC(double dr=0.1,double dz=0.6)          {fMaxDCAforVC[0]=dr; fMaxDCAforVC[1]=dz;}
  Double_t GetMaxChi2forVC()                              const {return fMaxChi2forVC;}
  void     SetMaxChi2forVC(double chi2=10)                      {fMaxChi2forVC = chi2;}
  //
  Bool_t   CheckDetectorPattern(UInt_t patt)              const;
  Bool_t   CheckDetectorPoints(const int* npsel)          const;
  void     SetObligatoryDetector(Int_t detID, Int_t tp, Bool_t v=kTRUE);
  void     SetEventSpeciiSelection(UInt_t sel)                  {fSelEventSpecii = sel;}
  UInt_t   GetEventSpeciiSelection()                      const {return fSelEventSpecii;}
  //
  void     SetVertex(const AliESDVertex* v)                     {fVertex = v;}
  const AliESDVertex* GetVertex()                         const {return fVertex;}
  //
  //----------------------------------------
  Bool_t     ReadParameters(const char* parfile="millepede.res", Bool_t useErrors=kTRUE);
  Float_t*   GetGloParVal()                               const {return (Float_t*)fGloParVal;}
  Float_t*   GetGloParErr()                               const {return (Float_t*)fGloParErr;}
  Int_t*     GetGloParLab()                               const {return (Int_t*)fGloParLab;}
  Int_t      GetGloParLab(int i)                          const {return (Int_t)fGloParLab[i];}
  Int_t      ParID2Label(int i)                           const {return GetGloParLab(i);}
  Int_t      Label2ParID(int lab)                         const;
  AliAlgVol* GetVolOfDOFID(int id)                        const;
  AliAlgDet* GetDetOfDOFID(int id)                        const;
  //
  AliAlgPoint* GetRefPoint()                              const {return (AliAlgPoint*)fRefPoint;}
  //
  AliAlgRes* GetContResid()                               const {return (AliAlgRes*)fCResid;}
  AliAlgMPRecord* GetMPRecord()                           const {return (AliAlgMPRecord*)fMPRecord;}
  TTree*    GetMPRecTree()                                const {return fMPRecTree;}
  AliAlgTrack* GetAlgTrack()                              const {return (AliAlgTrack*)fAlgTrack;}
  Bool_t     ProcessEvent(const AliESDEvent* esdEv); 
  Bool_t     ProcessTrack(const AliESDtrack* esdTr);
  Bool_t     ProcessTrack(const AliESDCosmicTrack* esdCTr);
  UInt_t     AcceptTrack(const AliESDtrack* esdTr, Bool_t strict=kTRUE)    const;
  UInt_t     AcceptTrackCosmic(const AliESDtrack* esdPairCosm[kNCosmLegs]) const;
  Bool_t     CheckSetVertex(const AliESDVertex* vtx);
  Bool_t     AddVertexConstraint();
  Int_t      GetNDetectors()                              const {return fNDet;}
  AliAlgDet* GetDetector(Int_t i)                         const {return fDetectors[i];}
  AliAlgDet* GetDetectorByDetID(Int_t i)                  const {return fDetPos[i]<0 ? 0:fDetectors[fDetPos[i]];}
  AliAlgDet* GetDetectorByVolID(Int_t id)                 const;
  AliAlgVtx* GetVertexSensor()                            const {return fVtxSens;}
  //
  void       ResetDetectors();
  Int_t      GetNDOFs()                                   const {return fNDOFs;}
  //
  const char* GetConfMacroName()                          const {return fConfMacroName.Data();}
  //----------------------------------------
  // output related
  void     SetMPDatFileName(const char* name="mpData");
  void     SetMPParFileName(const char* name="mpParams.txt");
  void     SetMPConFileName(const char* name="mpConstraints.txt");
  void     SetMPSteerFileName(const char* name="mpSteer.txt");
  void     SetResidFileName(const char* name="mpControlRes.root");
  void     SetOutCDBPath(const char* name="local://outOCDB");
  void     SetOutCDBComment(const char* cm=0)                    {fOutCDBComment = cm;}
  void     SetOutCDBResponsible(const char* v=0)                 {fOutCDBResponsible = v;}
  void     SetOutCDBRunRange(int rmin=0,int rmax=999999999);
  Int_t*   GetOutCDBRunRange()                             const {return (int*)fOutCDBRunRange;}
  Int_t    GetOutCDBRunMin()                               const {return fOutCDBRunRange[0];}
  Int_t    GetOutCDBRunMax()                               const {return fOutCDBRunRange[1];}
  Float_t  GetControlFrac()                                const {return fControlFrac;}
  void     SetControlFrac(float v=1.)                            {fControlFrac = v;}
  void     WriteCalibrationResults()                       const;
  void     ApplyAlignmentFromMPSol();
  const  char* GetOutCDBComment()                          const {return fOutCDBComment.Data();}
  const  char* GetOutCDBResponsible()                      const {return fOutCDBResponsible.Data();}
  const  char* GetOutCDBPath()                             const {return fOutCDBPath.Data();}
  const  char* GetMPDatFileName()                          const {return fMPDatFileName.Data();}
  const  char* GetResidFileName()                          const {return fResidFileName.Data();}
  const  char* GetMPParFileName()                          const {return fMPParFileName.Data();}
  const  char* GetMPConFileName()                          const {return fMPConFileName.Data();}
  const  char* GetMPSteerFileName()                        const {return fMPSteerFileName.Data();}
  //
  Bool_t   FillMPRecData();
  Bool_t   FillMilleData();
  Bool_t   FillControlData();
  void     SetDoKalmanResid(Bool_t v=kTRUE)                      {fDoKalmanResid = v;}
  void     SetMPOutType(Int_t t)                                 {fMPOutType = t;}
  void     ProduceMPData(Bool_t v=kTRUE)                         {if (v) fMPOutType|=kMille; else fMPOutType&=~kMille;}
  void     ProduceMPRecord(Bool_t v=kTRUE)                       {if (v) fMPOutType|=kMPRec; else fMPOutType&=~kMPRec;}
  void     ProduceControlRes(Bool_t v=kTRUE)                     {if (v) fMPOutType|=kContR; else fMPOutType&=~kContR;}
  Int_t    GetMPOutType()                                  const {return fMPOutType;}
  Bool_t   GetDoKalmanResid()                              const {return fDoKalmanResid;}
  Bool_t   GetProduceMPData()                              const {return fMPOutType&kMille;}
  Bool_t   GetProduceMPRecord()                            const {return fMPOutType&kMPRec;}
  Bool_t   GetProduceControlRes()                          const {return fMPOutType&kContR;}
  void     CloseMPRecOutput();
  void     CloseMilleOutput();
  void     CloseResidOutput();
  void     InitMPRecOutput();
  void     InitMIlleOutput();
  void     InitResidOutput();
  Bool_t   StoreProcessedTrack(Int_t what);
  void     PrintStatistics()                               const;
  Bool_t   GetMilleTXT()                                   const {return !fMilleOutBin;}
  void     SetMilleTXT(Bool_t v=kTRUE)                           {fMilleOutBin = !v;}
  //
  void     GenPedeSteerFile(const Option_t *opt="")        const;
  void     WritePedeConstraints()                          const;
  void     CheckConstraints(const char* params=0);
  AliAlgDOFStat* GetDOFStat()                              const {return fDOFStat;}
  void     SetDOFStat(AliAlgDOFStat* st)                        {fDOFStat = st;}
  void     DetachDOFStat()                                      {SetDOFStat(0);}
  TH1*     GetHistoStat()                                  const {return fHistoStat;}
  void     DetachHistoStat()                                     {SetHistoStat(0);}
  void     SetHistoStat(TH1F* h)                                 {fHistoStat = h;}
  void     FillStatHisto(int type, float w=1);
  void     CreateStatHisto();
  void     FixLowStatFromDOFStat(Int_t thresh=40);
  void     LoadStat(const char* flname);
  //
  //----------------------------------------
  //
  Int_t  GetRefRunNumber()                                const {return fRefRunNumber;}
  void   SetRefRunNumber(int r=-1)                               {fRefRunNumber = r;}
  //
  void   SetRefOCDBConfigMacro(const char* nm="configRefOCDB.C") {fRefOCDBConf = nm;}
  const  char* GetRefOCDBConfigMacro()                    const {return fRefOCDBConf.Data();}
  void   SetRecoOCDBConfigMacro(const char* nm="configRecoOCDB.C") {fRecoOCDBConf = nm;}
  const  char* GetRecoOCDBConfigMacro()                   const {return fRecoOCDBConf.Data();}
  Int_t  GetRefOCDBLoaded()                               const {return fRefOCDBLoaded;}
  //
  virtual void Print(const Option_t *opt="")              const;
  void         PrintLabels()                              const;
  Char_t*      GetDOFLabelTxt(int idf)                    const;
  //
  static Char_t* GetDetNameByDetID(Int_t id)              {return (Char_t*)fgkDetectorName[id];}
  static void    MPRec2Mille(const char* mprecfile,const char* millefile="mpData.mille",Bool_t bindata=kTRUE);
  static void    MPRec2Mille(TTree* mprTree,const char* millefile="mpData.mille",Bool_t bindata=kTRUE);
  //
  AliSymMatrix* BuildMatrix(TVectorD &vec);
  Bool_t        TestLocalSolution();
  //
  // fast check of solution using derivatives
  void   CheckSol(TTree* mpRecTree, Bool_t store=kTRUE,Bool_t verbose=kFALSE,Bool_t loc=kTRUE, const char* outName="resFast");
  Bool_t CheckSol(AliAlgMPRecord* rec,AliAlgResFast *rLG=0, AliAlgResFast* rL=0,Bool_t verbose=kTRUE, Bool_t loc=kTRUE);
  //
 protected:
  //
  // --------- dummies -----------
  AliAlgSteer(const AliAlgSteer&);
  AliAlgSteer& operator=(const AliAlgSteer&);
  //
 protected:
  //
  Int_t         fNDet;                                    // number of deectors participating in the alignment
  Int_t         fNDOFs;                                   // number of degrees of freedom
  Int_t         fRunNumber;                               // current run number
  Bool_t        fFieldOn;                                 // field on flag
  Int_t         fTracksType;                              // collision/cosmic event type
  AliAlgTrack*  fAlgTrack;                                // current alignment track 
  AliAlgDet*    fDetectors[kNDetectors];                  // detectors participating in the alignment
  Int_t         fDetPos[kNDetectors];                     // entry of detector in the fDetectors array
  AliAlgVtx*    fVtxSens;                                 // fake sensor for the vertex
  TObjArray     fConstraints;                             // array of constraints
  //
  // Track selection
  UInt_t        fSelEventSpecii;                          // consider only these event specii
  UInt_t        fObligatoryDetPattern[AliAlgAux::kNTrackTypes]; // pattern of obligatory detectors
  Bool_t        fCosmicSelStrict;                         // if true, each cosmic track leg selected like separate track
  Int_t         fMinPoints[AliAlgAux::kNTrackTypes][2];   // require min points per leg (case Boff,Bon)
  Int_t         fMinDetAcc[AliAlgAux::kNTrackTypes];      // min number of detector required in track
  Double_t      fDefPtBOff[AliAlgAux::kNTrackTypes];      // nominal pt for tracks in Boff run
  Double_t      fPtMin[AliAlgAux::kNTrackTypes];          // min pT of tracks to consider
  Double_t      fEtaMax[AliAlgAux::kNTrackTypes];         // eta cut on tracks
  Int_t         fVtxMinCont;                              // require min number of contributors in Vtx
  Int_t         fVtxMaxCont;                              // require max number of contributors in Vtx  
  Int_t         fVtxMinContVC;                            // min number of contributors to use as constraint
  //
  Int_t         fMinITSClforVC;                           // use vertex constraint for tracks with enough points
  Int_t         fITSPattforVC;                            // optional request on ITS hits to allow vertex constraint
  Double_t      fMaxDCAforVC[2];                          // DCA cut in R,Z to allow vertex constraint
  Double_t      fMaxChi2forVC;                            // track-vertex chi2 cut to allow vertex constraint
  //
  //
  Float_t*      fGloParVal;                               //[fNDOFs] parameters for DOFs
  Float_t*      fGloParErr;                               //[fNDOFs] errors for DOFs
  Int_t*        fGloParLab;                               //[fNDOFs] labels for DOFs
  Int_t*        fOrderedLbl;                              //[fNDOFs] ordered labels
  Int_t*        fLbl2ID;                                  //[fNDOFs] Label order in fOrderedLbl -> parID
  //
  AliAlgPoint*   fRefPoint;                               // reference point for track definition
  //
  const TTree*       fESDTree;                            //! externally set esdTree, needed to access UserInfo list
  const AliESDEvent* fESDEvent;                           //! externally set event
  const AliESDtrack* fESDTrack[kNCosmLegs];               //! externally set ESD tracks
  const AliESDVertex* fVertex;                            //! event vertex
  //
  // statistics
  Float_t fStat[kNStatCl][kMaxStat];                      // processing statistics
  static const Char_t* fgkStatClName[kNStatCl];           // stat classes names
  static const Char_t* fgkStatName[kMaxStat];             // stat type names  
  //
  // output related
  Float_t         fControlFrac;                           //  fraction of tracks to process control residuals
  Int_t           fMPOutType;                             // What to store as an output, see StoreProcessedTrack
  Mille*          fMille;                                 //! Mille interface
  AliAlgMPRecord* fMPRecord;                              //! MP record 
  AliAlgRes*      fCResid;                                //! control residuals
  TTree*          fMPRecTree;                             //! tree to store MP record
  TTree*          fResidTree;                             //! tree to store control residuals
  TFile*          fMPRecFile;                             //! file to store MP record tree
  TFile*          fResidFile;                             //! file to store control residuals tree
  TArrayF         fMilleDBuffer;                          //! buffer for Mille Derivatives output
  TArrayI         fMilleIBuffer;                          //! buffer for Mille Indecis output
  TString         fMPDatFileName;                         //  file name for records binary data output
  TString         fMPParFileName;                         //  file name for MP params
  TString         fMPConFileName;                         //  file name for MP constraints
  TString         fMPSteerFileName;                       //  file name for MP steering
  TString         fResidFileName;                         //  file name for optional control residuals
  Bool_t          fMilleOutBin;                           //  optionally text output for Mille debugging
  Bool_t          fDoKalmanResid;                         //  calculate residuals with smoothed kalman in the ControlRes
  //
  TString         fOutCDBPath;                            // output OCDB path
  TString         fOutCDBComment;                         // optional comment to add to output cdb objects
  TString         fOutCDBResponsible;                     // optional responsible for output metadata
  Int_t           fOutCDBRunRange[2];                     // run range for output storage
  //
  AliAlgDOFStat*  fDOFStat;                               // stat of entries per dof
  TH1F*           fHistoStat;                             // histo with general statistics
  //
  // input related
  TString         fConfMacroName;                         // optional configuration macro
  TString         fRecoOCDBConf;                          // optional macro name for reco-time OCDB setup: void fun(int run)
  TString         fRefOCDBConf;                           // optional macro name for prealignment OCDB setup: void fun()
  Int_t           fRefRunNumber;                          // optional run number used for reference
  Int_t           fRefOCDBLoaded;                         // flag/counter for ref.OCDB loading
  Bool_t          fUseRecoOCDB;                           // flag to preload reco-time calib objects
  //
  static const Int_t   fgkSkipLayers[kNLrSkip];           // detector layers for which we don't need module matrices
  static const Char_t* fgkDetectorName[kNDetectors];      // names of detectors
  static const Char_t* fgkHStatName[kNHVars];             // names for stat.bins in the stat histo
  static const Char_t* fgkMPDataExt;                      // extension for MP2 binary data 
  //
  ClassDef(AliAlgSteer,2)
};

//__________________________________________________________
inline void AliAlgSteer::SetMinPointsColl(int vbOff,int vbOn)
{
  // ask min number of points per track
  SetMinPoints(AliAlgAux::kColl,kFALSE,vbOff);
  SetMinPoints(AliAlgAux::kColl,kTRUE,vbOn);
}

//__________________________________________________________
inline void AliAlgSteer::SetMinPointsCosm(int vbOff,int vbOn)
{
  // ask min number of points per track
  SetMinPoints(AliAlgAux::kCosm,kFALSE,vbOff);
  SetMinPoints(AliAlgAux::kCosm,kTRUE,vbOn);
}

#endif
