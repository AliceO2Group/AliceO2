// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSteer.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Steering class for the global alignment

/**
 * Steering class for the global alignment. Responsible for feeding the track data
 * to participating detectors and preparation of the millepede input.
 */

#ifndef ALIALGSTEER_H
#define ALIALGSTEER_H

#include "DetectorsBase/GeometryManager.h"
#include "Align/AliAlgTrack.h"
// #include "AliSymMatrix.h" FIXME(milettri): needs AliSymMatrix

#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <TString.h>
#include <TArrayF.h>
#include <TArrayI.h>
#include <TH1F.h>
#include "Align/AliAlgAux.h"

//class AliESDEvent; FIXME(milettri): needs AliESDEvent
//class AliESDtrack; FIXME(milettri): needs AliESDtrack
//class AliESDCosmicTrack; FIXME(milettri): needs AliESDCosmicTrack
//class AliESDVertex; FIXME(milettri): needs AliESDVertex

class TTree;
class TFile;
//

namespace o2
{
namespace align
{

class Mille;

class AliAlgDet;
class AliAlgVol;
class AliAlgVtx;
class AliAlgPoint;
class AliAlgMPRecord;
class AliAlgRes;
class AliAlgResFast;
class AliAlgConstraint;
class AliAlgDOFStat;

class AliAlgSteer : public TObject
{
 public:
  enum { kNLrSkip = 4 };
  enum { kITS,
         kTPC,
         kTRD,
         kTOF,
         kHMPID,
         kNDetectors,
         kUndefined };
  enum { kCosmLow,
         kCosmUp,
         kNCosmLegs };
  enum { kInpStat,
         kAccStat,
         kNStatCl };
  enum { kRun,
         kEventColl,
         kEventCosm,
         kTrackColl,
         kTrackCosm,
         kMaxStat };
  enum MPOut_t { kMille = BIT(0),
                 kMPRec = BIT(1),
                 kContR = BIT(2) };
  enum { kInitGeomDone = BIT(14),
         kInitDOFsDone = BIT(15),
         kMPAlignDone = BIT(16) };
  //
  enum {     // STAT histo entries
    kRunDone // input runs
    ,
    kEvInp // input events
    ,
    kEvVtx // after vtx selection
    ,
    kTrackInp // input tracks
    ,
    kTrackFitInp // input to ini fit
    ,
    kTrackFitInpVC // those with vertex constraint
    ,
    kTrackProcMatInp // input to process materials
    ,
    kTrackResDerInp // input to resid/deriv calculation
    ,
    kTrackStore // stored tracks
    ,
    kTrackAcc // tracks accepted
    ,
    kTrackControl // control tracks filled
    //
    ,
    kNHVars
  };

  //
  AliAlgSteer(const char* configMacro = 0, int refRun = -1);
  virtual ~AliAlgSteer();
  //  bool LoadRefOCDB(); FIXME(milettri): needs OCDB
  //  bool LoadRecoTimeOCDB(); FIXME(milettri): needs OCDB
  bool GetUseRecoOCDB() const { return fUseRecoOCDB; }
  void SetUseRecoOCDB(bool v = true) { fUseRecoOCDB = v; }

  void InitDetectors();
  void InitDOFs();
  void Terminate(bool dostat = true);
  void SetStatHistoLabels(TH1* h) const;
  //
  void SetInitGeomDone() { SetBit(kInitGeomDone); }
  bool GetInitGeomDone() const { return TestBit(kInitGeomDone); }
  //
  void SetInitDOFsDone() { SetBit(kInitDOFsDone); }
  bool GetInitDOFsDone() const { return TestBit(kInitDOFsDone); }
  //
  void SetMPAlignDone() { SetBit(kMPAlignDone); }
  bool GetMPAlignDone() const { return TestBit(kMPAlignDone); }

  void AssignDOFs();
  //
  void AddDetector(uint32_t id, AliAlgDet* det = 0);
  void AddDetector(AliAlgDet* det);
  //
  void AddConstraint(const AliAlgConstraint* cs) { fConstraints.AddLast((TObject*)cs); }
  int GetNConstraints() const { return fConstraints.GetEntriesFast(); }
  const TObjArray* GetConstraints() const { return &fConstraints; }
  const AliAlgConstraint* GetConstraint(int i) const { return (AliAlgConstraint*)fConstraints[i]; }
  void AddAutoConstraints();
  //
  void AcknowledgeNewRun(int run);
  void SetRunNumber(int run);
  int GetRunNumber() const { return fRunNumber; }
  bool GetFieldOn() const { return fFieldOn; }
  void SetFieldOn(bool v = true) { fFieldOn = v; }
  int GetTracksType() const { return fTracksType; }
  void SetTracksType(int t = AliAlgAux::kColl) { fTracksType = t; }
  bool IsCosmic() const { return fTracksType == AliAlgAux::kCosm; }
  bool IsCollision() const { return fTracksType == AliAlgAux::kColl; }
  void SetCosmic(bool v = true) { fTracksType = v ? AliAlgAux::kCosm : AliAlgAux::kColl; }
  float GetStat(int cls, int tp) const { return fStat[cls][tp]; }
  //
  void SetESDTree(const TTree* tr) { fESDTree = tr; }
  const TTree* GetESDTree() const { return fESDTree; }
  //  void SetESDEvent(const AliESDEvent* ev); FIXME(milettri): needs AliESDEvent
  //  const AliESDEvent* GetESDEvent() const { return fESDEvent; } FIXME(milettri): needs AliESDEvent
  //  void SetESDtrack(const AliESDtrack* tr, int i = 0) { fESDTrack[i] = tr; } FIXME(milettri): needs AliESDtrack
  //  const AliESDtrack* GetESDtrack(int i = 0) const { return fESDTrack[i]; } FIXME(milettri): needs AliESDtrack
  //
  // Track selection
  void SetCosmicSelStrict(bool v = true) { fCosmicSelStrict = v; }
  bool GetCosmicSelStrict() const { return fCosmicSelStrict; }
  //
  int GetMinPoints() const { return fMinPoints[fTracksType][GetFieldOn()]; }
  int GetMinPoints(bool tp, bool bON) const { return fMinPoints[tp][bON]; }
  void SetMinPoints(bool tp, bool bON, int n)
  {
    int mn = bON ? 4 : 3;
    fMinPoints[tp][bON] = n > mn ? n : mn;
  }
  void SetMinPointsColl(int vbOff = 3, int vbOn = 4);
  void SetMinPointsCosm(int vbOff = 3, int vbOn = 4);
  //
  double GetPtMin(bool tp) const { return fPtMin[tp]; }
  void SetPtMin(bool tp, double pt) { fPtMin[tp] = pt; }
  void SetPtMinColl(double pt = 0.7) { SetPtMin(AliAlgAux::kColl, pt); }
  void SetPtMinCosm(double pt = 1.0) { SetPtMin(AliAlgAux::kCosm, pt); }
  //
  double GetEtaMax(bool tp) const { return fEtaMax[tp]; }
  void SetEtaMax(bool tp, double eta) { fEtaMax[tp] = eta; }
  void SetEtaMaxColl(double eta = 1.5) { SetEtaMax(AliAlgAux::kColl, eta); }
  void SetEtaMaxCosm(double eta = 1.5) { SetEtaMax(AliAlgAux::kCosm, eta); }
  //
  void SetDefPtBOffCosm(double pt = 5.0) { fDefPtBOff[AliAlgAux::kCosm] = pt > 0.3 ? pt : 0.3; }
  void SetDefPtBOffColl(double pt = 0.6) { fDefPtBOff[AliAlgAux::kColl] = pt > 0.3 ? pt : 0.3; }
  double GetDefPtBOff(bool tp) { return fDefPtBOff[tp]; }
  //
  int GetMinDetAcc(bool tp) const { return fMinDetAcc[tp]; }
  void SetMinDetAcc(bool tp, int n) { fMinDetAcc[tp] = n; }
  void SetMinDetAccColl(int n = 1) { SetMinDetAcc(AliAlgAux::kColl, n); }
  void SetMinDetAccCosm(int n = 1) { SetMinDetAcc(AliAlgAux::kCosm, n); }
  //
  int GetVtxMinCont() const { return fVtxMinCont; }
  void SetVtxMinCont(int n) { fVtxMinCont = n; }
  int GetVtxMaxCont() const { return fVtxMaxCont; }
  void SetVtxMaxCont(int n) { fVtxMaxCont = n; }
  int GetVtxMinContVC() const { return fVtxMinContVC; }
  void SetVtxMinContVC(int n) { fVtxMinContVC = n; }
  //
  int GetMinITSClforVC() const { return fMinITSClforVC; }
  void SetMinITSClforVC(int n) { fMinITSClforVC = n; }
  int GetITSPattforVC() const { return fITSPattforVC; }
  void SetITSPattforVC(int p) { fITSPattforVC = p; }
  double GetMaxDCARforVC() const { return fMaxDCAforVC[0]; }
  double GetMaxDCAZforVC() const { return fMaxDCAforVC[1]; }
  void SetMaxDCAforVC(double dr = 0.1, double dz = 0.6)
  {
    fMaxDCAforVC[0] = dr;
    fMaxDCAforVC[1] = dz;
  }
  double GetMaxChi2forVC() const { return fMaxChi2forVC; }
  void SetMaxChi2forVC(double chi2 = 10) { fMaxChi2forVC = chi2; }
  //
  bool CheckDetectorPattern(uint32_t patt) const;
  bool CheckDetectorPoints(const int* npsel) const;
  void SetObligatoryDetector(int detID, int tp, bool v = true);
  void SetEventSpeciiSelection(uint32_t sel) { fSelEventSpecii = sel; }
  uint32_t GetEventSpeciiSelection() const { return fSelEventSpecii; }
  //
  //  void SetVertex(const AliESDVertex* v) { fVertex = v; } FIXME(milettri): needs AliESDVertex
  //  const AliESDVertex* GetVertex() const { return fVertex; } FIXME(milettri): needs AliESDVertex
  //
  //----------------------------------------
  bool ReadParameters(const char* parfile = "millepede.res", bool useErrors = true);
  float* GetGloParVal() const { return (float*)fGloParVal; }
  float* GetGloParErr() const { return (float*)fGloParErr; }
  int* GetGloParLab() const { return (int*)fGloParLab; }
  int GetGloParLab(int i) const { return (int)fGloParLab[i]; }
  int ParID2Label(int i) const { return GetGloParLab(i); }
  int Label2ParID(int lab) const;
  AliAlgVol* GetVolOfDOFID(int id) const;
  AliAlgDet* GetDetOfDOFID(int id) const;
  //
  AliAlgPoint* GetRefPoint() const { return (AliAlgPoint*)fRefPoint; }
  //
  AliAlgRes* GetContResid() const { return (AliAlgRes*)fCResid; }
  AliAlgMPRecord* GetMPRecord() const { return (AliAlgMPRecord*)fMPRecord; }
  TTree* GetMPRecTree() const { return fMPRecTree; }
  AliAlgTrack* GetAlgTrack() const { return (AliAlgTrack*)fAlgTrack; }
  //  bool ProcessEvent(const AliESDEvent* esdEv); FIXME(milettri): needs AliESDEvent
  //  bool ProcessTrack(const AliESDtrack* esdTr); FIXME(milettri): needs AliESDtrack
  //  bool ProcessTrack(const AliESDCosmicTrack* esdCTr); FIXME(milettri): needs AliESDCosmicTrack
  //  uint32_t AcceptTrack(const AliESDtrack* esdTr, bool strict = true) const; FIXME(milettri): needs AliESDtrack
  //  uint32_t AcceptTrackCosmic(const AliESDtrack* esdPairCosm[kNCosmLegs]) const; FIXME(milettri): needs AliESDtrack
  //  bool CheckSetVertex(const AliESDVertex* vtx); FIXME(milettri): needs AliESDVertex
  bool AddVertexConstraint();
  int GetNDetectors() const { return fNDet; }
  AliAlgDet* GetDetector(int i) const { return fDetectors[i]; }
  AliAlgDet* GetDetectorByDetID(int i) const { return fDetPos[i] < 0 ? 0 : fDetectors[fDetPos[i]]; }
  AliAlgDet* GetDetectorByVolID(int id) const;
  AliAlgVtx* GetVertexSensor() const { return fVtxSens; }
  //
  void ResetDetectors();
  int GetNDOFs() const { return fNDOFs; }
  //
  const char* GetConfMacroName() const { return fConfMacroName.Data(); }
  //----------------------------------------
  // output related
  void SetMPDatFileName(const char* name = "mpData");
  void SetMPParFileName(const char* name = "mpParams.txt");
  void SetMPConFileName(const char* name = "mpConstraints.txt");
  void SetMPSteerFileName(const char* name = "mpSteer.txt");
  void SetResidFileName(const char* name = "mpControlRes.root");
  void SetOutCDBPath(const char* name = "local://outOCDB");
  void SetOutCDBComment(const char* cm = 0) { fOutCDBComment = cm; }
  void SetOutCDBResponsible(const char* v = 0) { fOutCDBResponsible = v; }
  //  void SetOutCDBRunRange(int rmin = 0, int rmax = 999999999); FIXME(milettri): needs OCDB
  int* GetOutCDBRunRange() const { return (int*)fOutCDBRunRange; }
  int GetOutCDBRunMin() const { return fOutCDBRunRange[0]; }
  int GetOutCDBRunMax() const { return fOutCDBRunRange[1]; }
  float GetControlFrac() const { return fControlFrac; }
  void SetControlFrac(float v = 1.) { fControlFrac = v; }
  //  void WriteCalibrationResults() const; FIXME(milettri): needs OCDB
  void ApplyAlignmentFromMPSol();
  const char* GetOutCDBComment() const { return fOutCDBComment.Data(); }
  const char* GetOutCDBResponsible() const { return fOutCDBResponsible.Data(); }
  const char* GetOutCDBPath() const { return fOutCDBPath.Data(); }
  const char* GetMPDatFileName() const { return fMPDatFileName.Data(); }
  const char* GetResidFileName() const { return fResidFileName.Data(); }
  const char* GetMPParFileName() const { return fMPParFileName.Data(); }
  const char* GetMPConFileName() const { return fMPConFileName.Data(); }
  const char* GetMPSteerFileName() const { return fMPSteerFileName.Data(); }
  //
  bool FillMPRecData();
  bool FillMilleData();
  bool FillControlData();
  void SetDoKalmanResid(bool v = true) { fDoKalmanResid = v; }
  void SetMPOutType(int t) { fMPOutType = t; }
  void ProduceMPData(bool v = true)
  {
    if (v)
      fMPOutType |= kMille;
    else
      fMPOutType &= ~kMille;
  }
  void ProduceMPRecord(bool v = true)
  {
    if (v)
      fMPOutType |= kMPRec;
    else
      fMPOutType &= ~kMPRec;
  }
  void ProduceControlRes(bool v = true)
  {
    if (v)
      fMPOutType |= kContR;
    else
      fMPOutType &= ~kContR;
  }
  int GetMPOutType() const { return fMPOutType; }
  bool GetDoKalmanResid() const { return fDoKalmanResid; }
  bool GetProduceMPData() const { return fMPOutType & kMille; }
  bool GetProduceMPRecord() const { return fMPOutType & kMPRec; }
  bool GetProduceControlRes() const { return fMPOutType & kContR; }
  void CloseMPRecOutput();
  void CloseMilleOutput();
  void CloseResidOutput();
  void InitMPRecOutput();
  void InitMIlleOutput();
  void InitResidOutput();
  bool StoreProcessedTrack(int what);
  void PrintStatistics() const;
  bool GetMilleTXT() const { return !fMilleOutBin; }
  void SetMilleTXT(bool v = true) { fMilleOutBin = !v; }
  //
  void GenPedeSteerFile(const Option_t* opt = "") const;
  void WritePedeConstraints() const;
  void CheckConstraints(const char* params = 0);
  AliAlgDOFStat* GetDOFStat() const { return fDOFStat; }
  void SetDOFStat(AliAlgDOFStat* st) { fDOFStat = st; }
  void DetachDOFStat() { SetDOFStat(0); }
  TH1* GetHistoStat() const { return fHistoStat; }
  void DetachHistoStat() { SetHistoStat(0); }
  void SetHistoStat(TH1F* h) { fHistoStat = h; }
  void FillStatHisto(int type, float w = 1);
  void CreateStatHisto();
  void FixLowStatFromDOFStat(int thresh = 40);
  void LoadStat(const char* flname);
  //
  //----------------------------------------
  //
  int GetRefRunNumber() const { return fRefRunNumber; }
  void SetRefRunNumber(int r = -1) { fRefRunNumber = r; }
  //
  void SetRefOCDBConfigMacro(const char* nm = "configRefOCDB.C") { fRefOCDBConf = nm; }
  const char* GetRefOCDBConfigMacro() const { return fRefOCDBConf.Data(); }
  void SetRecoOCDBConfigMacro(const char* nm = "configRecoOCDB.C") { fRecoOCDBConf = nm; }
  const char* GetRecoOCDBConfigMacro() const { return fRecoOCDBConf.Data(); }
  int GetRefOCDBLoaded() const { return fRefOCDBLoaded; }
  //
  virtual void Print(const Option_t* opt = "") const;
  void PrintLabels() const;
  char* GetDOFLabelTxt(int idf) const;
  //
  static char* GetDetNameByDetID(int id) { return (char*)fgkDetectorName[id]; }
  static void MPRec2Mille(const char* mprecfile, const char* millefile = "mpData.mille", bool bindata = true);
  static void MPRec2Mille(TTree* mprTree, const char* millefile = "mpData.mille", bool bindata = true);
  //
  //  AliSymMatrix* BuildMatrix(TVectorD& vec); FIXME(milettri): needs AliSymMatrix
  bool TestLocalSolution();
  //
  // fast check of solution using derivatives
  void CheckSol(TTree* mpRecTree, bool store = true, bool verbose = false, bool loc = true, const char* outName = "resFast");
  bool CheckSol(AliAlgMPRecord* rec, AliAlgResFast* rLG = 0, AliAlgResFast* rL = 0, bool verbose = true, bool loc = true);
  //
 protected:
  //
  // --------- dummies -----------
  AliAlgSteer(const AliAlgSteer&);
  AliAlgSteer& operator=(const AliAlgSteer&);
  //
 protected:
  //
  int fNDet;                          // number of deectors participating in the alignment
  int fNDOFs;                         // number of degrees of freedom
  int fRunNumber;                     // current run number
  bool fFieldOn;                      // field on flag
  int fTracksType;                    // collision/cosmic event type
  AliAlgTrack* fAlgTrack;             // current alignment track
  AliAlgDet* fDetectors[kNDetectors]; // detectors participating in the alignment
  int fDetPos[kNDetectors];           // entry of detector in the fDetectors array
  AliAlgVtx* fVtxSens;                // fake sensor for the vertex
  TObjArray fConstraints;             // array of constraints
  //
  // Track selection
  uint32_t fSelEventSpecii;                                // consider only these event specii
  uint32_t fObligatoryDetPattern[AliAlgAux::kNTrackTypes]; // pattern of obligatory detectors
  bool fCosmicSelStrict;                                   // if true, each cosmic track leg selected like separate track
  int fMinPoints[AliAlgAux::kNTrackTypes][2];              // require min points per leg (case Boff,Bon)
  int fMinDetAcc[AliAlgAux::kNTrackTypes];                 // min number of detector required in track
  double fDefPtBOff[AliAlgAux::kNTrackTypes];              // nominal pt for tracks in Boff run
  double fPtMin[AliAlgAux::kNTrackTypes];                  // min pT of tracks to consider
  double fEtaMax[AliAlgAux::kNTrackTypes];                 // eta cut on tracks
  int fVtxMinCont;                                         // require min number of contributors in Vtx
  int fVtxMaxCont;                                         // require max number of contributors in Vtx
  int fVtxMinContVC;                                       // min number of contributors to use as constraint
  //
  int fMinITSClforVC;     // use vertex constraint for tracks with enough points
  int fITSPattforVC;      // optional request on ITS hits to allow vertex constraint
  double fMaxDCAforVC[2]; // DCA cut in R,Z to allow vertex constraint
  double fMaxChi2forVC;   // track-vertex chi2 cut to allow vertex constraint
  //
  //
  float* fGloParVal; //[fNDOFs] parameters for DOFs
  float* fGloParErr; //[fNDOFs] errors for DOFs
  int* fGloParLab;   //[fNDOFs] labels for DOFs
  int* fOrderedLbl;  //[fNDOFs] ordered labels
  int* fLbl2ID;      //[fNDOFs] Label order in fOrderedLbl -> parID
  //
  AliAlgPoint* fRefPoint; // reference point for track definition
  //
  const TTree* fESDTree; //! externally set esdTree, needed to access UserInfo list
                         //  const AliESDEvent* fESDEvent;             //! externally set event  FIXME(milettri): needs AliESDEvent
                         //  const AliESDtrack* fESDTrack[kNCosmLegs]; //! externally set ESD tracks  FIXME(milettri): needs AliESDtrack
                         //  const AliESDVertex* fVertex;              //! event vertex FIXME(milettri): needs AliESDVertex
  //
  // statistics
  float fStat[kNStatCl][kMaxStat];            // processing statistics
  static const char* fgkStatClName[kNStatCl]; // stat classes names
  static const char* fgkStatName[kMaxStat];   // stat type names
  //
  // output related
  float fControlFrac;        //  fraction of tracks to process control residuals
  int fMPOutType;            // What to store as an output, see StoreProcessedTrack
  Mille* fMille;             //! Mille interface
  AliAlgMPRecord* fMPRecord; //! MP record
  AliAlgRes* fCResid;        //! control residuals
  TTree* fMPRecTree;         //! tree to store MP record
  TTree* fResidTree;         //! tree to store control residuals
  TFile* fMPRecFile;         //! file to store MP record tree
  TFile* fResidFile;         //! file to store control residuals tree
  TArrayF fMilleDBuffer;     //! buffer for Mille Derivatives output
  TArrayI fMilleIBuffer;     //! buffer for Mille Indecis output
  TString fMPDatFileName;    //  file name for records binary data output
  TString fMPParFileName;    //  file name for MP params
  TString fMPConFileName;    //  file name for MP constraints
  TString fMPSteerFileName;  //  file name for MP steering
  TString fResidFileName;    //  file name for optional control residuals
  bool fMilleOutBin;         //  optionally text output for Mille debugging
  bool fDoKalmanResid;       //  calculate residuals with smoothed kalman in the ControlRes
  //
  TString fOutCDBPath;        // output OCDB path
  TString fOutCDBComment;     // optional comment to add to output cdb objects
  TString fOutCDBResponsible; // optional responsible for output metadata
  int fOutCDBRunRange[2];     // run range for output storage
  //
  AliAlgDOFStat* fDOFStat; // stat of entries per dof
  TH1F* fHistoStat;        // histo with general statistics
  //
  // input related
  TString fConfMacroName; // optional configuration macro
  TString fRecoOCDBConf;  // optional macro name for reco-time OCDB setup: void fun(int run)
  TString fRefOCDBConf;   // optional macro name for prealignment OCDB setup: void fun()
  int fRefRunNumber;      // optional run number used for reference
  int fRefOCDBLoaded;     // flag/counter for ref.OCDB loading
  bool fUseRecoOCDB;      // flag to preload reco-time calib objects
  //
  static const int fgkSkipLayers[kNLrSkip];        // detector layers for which we don't need module matrices
  static const char* fgkDetectorName[kNDetectors]; // names of detectors
  static const char* fgkHStatName[kNHVars];        // names for stat.bins in the stat histo
  static const char* fgkMPDataExt;                 // extension for MP2 binary data
  //
  ClassDef(AliAlgSteer, 2)
};

//__________________________________________________________
inline void AliAlgSteer::SetMinPointsColl(int vbOff, int vbOn)
{
  // ask min number of points per track
  SetMinPoints(AliAlgAux::kColl, false, vbOff);
  SetMinPoints(AliAlgAux::kColl, true, vbOn);
}

//__________________________________________________________
inline void AliAlgSteer::SetMinPointsCosm(int vbOff, int vbOn)
{
  // ask min number of points per track
  SetMinPoints(AliAlgAux::kCosm, false, vbOff);
  SetMinPoints(AliAlgAux::kCosm, true, vbOn);
}
} // namespace align
} // namespace o2
#endif
