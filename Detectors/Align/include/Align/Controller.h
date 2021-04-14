// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Controller.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Steering class for the global alignment

/**
 * Steering class for the global alignment. Responsible for feeding the track data
 * to participating detectors and preparation of the millepede input.
 */

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "DetectorsBase/GeometryManager.h"
#include "Align/AlignmentTrack.h"
// #include "AliSymMatrix.h" FIXME(milettri): needs AliSymMatrix

#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <TString.h>
#include <TArrayF.h>
#include <TArrayI.h>
#include <TH1F.h>
#include "Align/utils.h"

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

class AlignableDetector;
class AlignableVolume;
class EventVertex;
class AlignmentPoint;
class Millepede2Record;
class ResidualsController;
class ResidualsControllerFast;
class GeometricalConstraint;
class DOFStatistics;

class Controller : public TObject
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
  Controller(const char* configMacro = 0, int refRun = -1);
  virtual ~Controller();
  //  bool LoadRefOCDB(); FIXME(milettri): needs OCDB
  //  bool LoadRecoTimeOCDB(); FIXME(milettri): needs OCDB
  bool getUseRecoOCDB() const { return mUseRecoOCDB; }
  void setUseRecoOCDB(bool v = true) { mUseRecoOCDB = v; }

  void initDetectors();
  void initDOFs();
  void terminate(bool dostat = true);
  void setStatHistoLabels(TH1* h) const;
  //
  void setInitGeomDone() { SetBit(kInitGeomDone); }
  bool getInitGeomDone() const { return TestBit(kInitGeomDone); }
  //
  void setInitDOFsDone() { SetBit(kInitDOFsDone); }
  bool getInitDOFsDone() const { return TestBit(kInitDOFsDone); }
  //
  void setMPAlignDone() { SetBit(kMPAlignDone); }
  bool getMPAlignDone() const { return TestBit(kMPAlignDone); }

  void assignDOFs();
  //
  void addDetector(uint32_t id, AlignableDetector* det = 0);
  void addDetector(AlignableDetector* det);
  //
  void addConstraint(const GeometricalConstraint* cs) { mConstraints.AddLast((TObject*)cs); }
  int getNConstraints() const { return mConstraints.GetEntriesFast(); }
  const TObjArray* getConstraints() const { return &mConstraints; }
  const GeometricalConstraint* getConstraint(int i) const { return (GeometricalConstraint*)mConstraints[i]; }
  void addAutoConstraints();
  //
  void acknowledgeNewRun(int run);
  void setRunNumber(int run);
  int getRunNumber() const { return mRunNumber; }
  bool getFieldOn() const { return mFieldOn; }
  void setFieldOn(bool v = true) { mFieldOn = v; }
  int getTracksType() const { return mTracksType; }
  void setTracksType(int t = utils::Coll) { mTracksType = t; }
  bool isCosmic() const { return mTracksType == utils::Cosm; }
  bool isCollision() const { return mTracksType == utils::Coll; }
  void setCosmic(bool v = true) { mTracksType = v ? utils::Cosm : utils::Coll; }
  float getStat(int cls, int tp) const { return mStat[cls][tp]; }
  //
  void setESDTree(const TTree* tr) { mESDTree = tr; }
  const TTree* getESDTree() const { return mESDTree; }
  //  void SetESDEvent(const AliESDEvent* ev); FIXME(milettri): needs AliESDEvent
  //  const AliESDEvent* GetESDEvent() const { return fESDEvent; } FIXME(milettri): needs AliESDEvent
  //  void SetESDtrack(const AliESDtrack* tr, int i = 0) { fESDTrack[i] = tr; } FIXME(milettri): needs AliESDtrack
  //  const AliESDtrack* GetESDtrack(int i = 0) const { return fESDTrack[i]; } FIXME(milettri): needs AliESDtrack
  //
  // Track selection
  void setCosmicSelStrict(bool v = true) { mCosmicSelStrict = v; }
  bool getCosmicSelStrict() const { return mCosmicSelStrict; }
  //
  int getMinPoints() const { return mMinPoints[mTracksType][getFieldOn()]; }
  int getMinPoints(bool tp, bool bON) const { return mMinPoints[tp][bON]; }
  void setMinPoints(bool tp, bool bON, int n)
  {
    int mn = bON ? 4 : 3;
    mMinPoints[tp][bON] = n > mn ? n : mn;
  }
  void setMinPointsColl(int vbOff = 3, int vbOn = 4);
  void setMinPointsCosm(int vbOff = 3, int vbOn = 4);
  //
  double getPtMin(bool tp) const { return mPtMin[tp]; }
  void setPtMin(bool tp, double pt) { mPtMin[tp] = pt; }
  void setPtMinColl(double pt = 0.7) { setPtMin(utils::Coll, pt); }
  void setPtMinCosm(double pt = 1.0) { setPtMin(utils::Cosm, pt); }
  //
  double getEtaMax(bool tp) const { return mEtaMax[tp]; }
  void setEtaMax(bool tp, double eta) { mEtaMax[tp] = eta; }
  void setEtaMaxColl(double eta = 1.5) { setEtaMax(utils::Coll, eta); }
  void setEtaMaxCosm(double eta = 1.5) { setEtaMax(utils::Cosm, eta); }
  //
  void setDefPtBOffCosm(double pt = 5.0) { mDefPtBOff[utils::Cosm] = pt > 0.3 ? pt : 0.3; }
  void setDefPtBOffColl(double pt = 0.6) { mDefPtBOff[utils::Coll] = pt > 0.3 ? pt : 0.3; }
  double getDefPtBOff(bool tp) { return mDefPtBOff[tp]; }
  //
  int getMinDetAcc(bool tp) const { return mMinDetAcc[tp]; }
  void setMinDetAcc(bool tp, int n) { mMinDetAcc[tp] = n; }
  void setMinDetAccColl(int n = 1) { setMinDetAcc(utils::Coll, n); }
  void setMinDetAccCosm(int n = 1) { setMinDetAcc(utils::Cosm, n); }
  //
  int getVtxMinCont() const { return mVtxMinCont; }
  void setVtxMinCont(int n) { mVtxMinCont = n; }
  int getVtxMaxCont() const { return mVtxMaxCont; }
  void setVtxMaxCont(int n) { mVtxMaxCont = n; }
  int getVtxMinContVC() const { return mVtxMinContVC; }
  void setVtxMinContVC(int n) { mVtxMinContVC = n; }
  //
  int getMinITSClforVC() const { return mMinITSClforVC; }
  void setMinITSClforVC(int n) { mMinITSClforVC = n; }
  int getITSPattforVC() const { return mITSPattforVC; }
  void setITSPattforVC(int p) { mITSPattforVC = p; }
  double getMaxDCARforVC() const { return mMaxDCAforVC[0]; }
  double getMaxDCAZforVC() const { return mMaxDCAforVC[1]; }
  void setMaxDCAforVC(double dr = 0.1, double dz = 0.6)
  {
    mMaxDCAforVC[0] = dr;
    mMaxDCAforVC[1] = dz;
  }
  double getMaxChi2forVC() const { return mMaxChi2forVC; }
  void setMaxChi2forVC(double chi2 = 10) { mMaxChi2forVC = chi2; }
  //
  bool checkDetectorPattern(uint32_t patt) const;
  bool checkDetectorPoints(const int* npsel) const;
  void setObligatoryDetector(int detID, int tp, bool v = true);
  void setEventSpeciiSelection(uint32_t sel) { mSelEventSpecii = sel; }
  uint32_t getEventSpeciiSelection() const { return mSelEventSpecii; }
  //
  //  void SetVertex(const AliESDVertex* v) { fVertex = v; } FIXME(milettri): needs AliESDVertex
  //  const AliESDVertex* GetVertex() const { return fVertex; } FIXME(milettri): needs AliESDVertex
  //
  //----------------------------------------
  bool readParameters(const char* parfile = "millepede.res", bool useErrors = true);
  float* getGloParVal() const { return (float*)mGloParVal; }
  float* getGloParErr() const { return (float*)mGloParErr; }
  int* getGloParLab() const { return (int*)mGloParLab; }
  int getGloParLab(int i) const { return (int)mGloParLab[i]; }
  int parID2Label(int i) const { return getGloParLab(i); }
  int label2ParID(int lab) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  AlignableDetector* getDetOfDOFID(int id) const;
  //
  AlignmentPoint* getRefPoint() const { return (AlignmentPoint*)mRefPoint; }
  //
  ResidualsController* getContResid() const { return (ResidualsController*)mCResid; }
  Millepede2Record* getMPRecord() const { return (Millepede2Record*)mMPRecord; }
  TTree* getMPRecTree() const { return mMPRecTree; }
  AlignmentTrack* getAlgTrack() const { return (AlignmentTrack*)mAlgTrack; }
  //  bool ProcessEvent(const AliESDEvent* esdEv); FIXME(milettri): needs AliESDEvent
  //  bool ProcessTrack(const AliESDtrack* esdTr); FIXME(milettri): needs AliESDtrack
  //  bool ProcessTrack(const AliESDCosmicTrack* esdCTr); FIXME(milettri): needs AliESDCosmicTrack
  //  uint32_t AcceptTrack(const AliESDtrack* esdTr, bool strict = true) const; FIXME(milettri): needs AliESDtrack
  //  uint32_t AcceptTrackCosmic(const AliESDtrack* esdPairCosm[kNCosmLegs]) const; FIXME(milettri): needs AliESDtrack
  //  bool CheckSetVertex(const AliESDVertex* vtx); FIXME(milettri): needs AliESDVertex
  bool addVertexConstraint();
  int getNDetectors() const { return mNDet; }
  AlignableDetector* getDetector(int i) const { return mDetectors[i]; }
  AlignableDetector* getDetectorByDetID(int i) const { return mDetPos[i] < 0 ? 0 : mDetectors[mDetPos[i]]; }
  AlignableDetector* getDetectorByVolID(int id) const;
  EventVertex* getVertexSensor() const { return mVtxSens; }
  //
  void resetDetectors();
  int getNDOFs() const { return mNDOFs; }
  //
  const char* getConfMacroName() const { return mConfMacroName.Data(); }
  //----------------------------------------
  // output related
  void setMPDatFileName(const char* name = "mpData");
  void setMPParFileName(const char* name = "mpParams.txt");
  void setMPConFileName(const char* name = "mpConstraints.txt");
  void setMPSteerFileName(const char* name = "mpSteer.txt");
  void setResidFileName(const char* name = "mpControlRes.root");
  void setOutCDBPath(const char* name = "local://outOCDB");
  void setOutCDBComment(const char* cm = 0) { mOutCDBComment = cm; }
  void setOutCDBResponsible(const char* v = 0) { mOutCDBResponsible = v; }
  //  void SetOutCDBRunRange(int rmin = 0, int rmax = 999999999); FIXME(milettri): needs OCDB
  int* getOutCDBRunRange() const { return (int*)mOutCDBRunRange; }
  int getOutCDBRunMin() const { return mOutCDBRunRange[0]; }
  int getOutCDBRunMax() const { return mOutCDBRunRange[1]; }
  float getControlFrac() const { return mControlFrac; }
  void setControlFrac(float v = 1.) { mControlFrac = v; }
  //  void writeCalibrationResults() const; FIXME(milettri): needs OCDB
  void applyAlignmentFromMPSol();
  const char* getOutCDBComment() const { return mOutCDBComment.Data(); }
  const char* getOutCDBResponsible() const { return mOutCDBResponsible.Data(); }
  const char* getOutCDBPath() const { return mOutCDBPath.Data(); }
  const char* getMPDatFileName() const { return mMPDatFileName.Data(); }
  const char* getResidFileName() const { return mResidFileName.Data(); }
  const char* getMPParFileName() const { return mMPParFileName.Data(); }
  const char* getMPConFileName() const { return mMPConFileName.Data(); }
  const char* getMPSteerFileName() const { return mMPSteerFileName.Data(); }
  //
  bool fillMPRecData();
  bool fillMilleData();
  bool fillControlData();
  void setDoKalmanResid(bool v = true) { mDoKalmanResid = v; }
  void setMPOutType(int t) { mMPOutType = t; }
  void produceMPData(bool v = true)
  {
    if (v)
      mMPOutType |= kMille;
    else
      mMPOutType &= ~kMille;
  }
  void produceMPRecord(bool v = true)
  {
    if (v)
      mMPOutType |= kMPRec;
    else
      mMPOutType &= ~kMPRec;
  }
  void produceControlRes(bool v = true)
  {
    if (v)
      mMPOutType |= kContR;
    else
      mMPOutType &= ~kContR;
  }
  int getMPOutType() const { return mMPOutType; }
  bool getDoKalmanResid() const { return mDoKalmanResid; }
  bool getProduceMPData() const { return mMPOutType & kMille; }
  bool getProduceMPRecord() const { return mMPOutType & kMPRec; }
  bool getProduceControlRes() const { return mMPOutType & kContR; }
  void closeMPRecOutput();
  void closeMilleOutput();
  void closeResidOutput();
  void initMPRecOutput();
  void initMIlleOutput();
  void initResidOutput();
  bool storeProcessedTrack(int what);
  void printStatistics() const;
  bool getMilleTXT() const { return !mMilleOutBin; }
  void setMilleTXT(bool v = true) { mMilleOutBin = !v; }
  //
  void genPedeSteerFile(const Option_t* opt = "") const;
  void writePedeConstraints() const;
  void checkConstraints(const char* params = 0);
  DOFStatistics* GetDOFStat() const { return mDOFStat; }
  void setDOFStat(DOFStatistics* st) { mDOFStat = st; }
  void detachDOFStat() { setDOFStat(0); }
  TH1* getHistoStat() const { return mHistoStat; }
  void detachHistoStat() { setHistoStat(0); }
  void setHistoStat(TH1F* h) { mHistoStat = h; }
  void fillStatHisto(int type, float w = 1);
  void createStatHisto();
  void fixLowStatFromDOFStat(int thresh = 40);
  void loadStat(const char* flname);
  //
  //----------------------------------------
  //
  int getRefRunNumber() const { return mRefRunNumber; }
  void setRefRunNumber(int r = -1) { mRefRunNumber = r; }
  //
  void setRefOCDBConfigMacro(const char* nm = "configRefOCDB.C") { mRefOCDBConf = nm; }
  const char* getRefOCDBConfigMacro() const { return mRefOCDBConf.Data(); }
  void setRecoOCDBConfigMacro(const char* nm = "configRecoOCDB.C") { mRecoOCDBConf = nm; }
  const char* getRecoOCDBConfigMacro() const { return mRecoOCDBConf.Data(); }
  int getRefOCDBLoaded() const { return mRefOCDBLoaded; }
  //
  virtual void Print(const Option_t* opt = "") const;
  void printLabels() const;
  Char_t* getDOFLabelTxt(int idf) const;
  //
  static Char_t* getDetNameByDetID(int id) { return (Char_t*)sDetectorName[id]; }
  static void mPRec2Mille(const char* mprecfile, const char* millefile = "mpData.mille", bool bindata = true);
  static void mPRec2Mille(TTree* mprTree, const char* millefile = "mpData.mille", bool bindata = true);
  //
  //  AliSymMatrix* BuildMatrix(TVectorD& vec); FIXME(milettri): needs AliSymMatrix
  bool testLocalSolution();
  //
  // fast check of solution using derivatives
  void checkSol(TTree* mpRecTree, bool store = true, bool verbose = false, bool loc = true, const char* outName = "resFast");
  bool checkSol(Millepede2Record* rec, ResidualsControllerFast* rLG = 0, ResidualsControllerFast* rL = 0, bool verbose = true, bool loc = true);
  //
 protected:
  //
  // --------- dummies -----------
  Controller(const Controller&);
  Controller& operator=(const Controller&);
  //
 protected:
  //
  int mNDet;                                  // number of deectors participating in the alignment
  int mNDOFs;                                 // number of degrees of freedom
  int mRunNumber;                             // current run number
  bool mFieldOn;                              // field on flag
  int mTracksType;                            // collision/cosmic event type
  AlignmentTrack* mAlgTrack;                  // current alignment track
  AlignableDetector* mDetectors[kNDetectors]; // detectors participating in the alignment
  int mDetPos[kNDetectors];                   // entry of detector in the mDetectors array
  EventVertex* mVtxSens;                      // fake sensor for the vertex
  TObjArray mConstraints;                     // array of constraints
  //
  // Track selection
  uint32_t mSelEventSpecii;                           // consider only these event specii
  uint32_t mObligatoryDetPattern[utils::NTrackTypes]; // pattern of obligatory detectors
  bool mCosmicSelStrict;                              // if true, each cosmic track leg selected like separate track
  int mMinPoints[utils::NTrackTypes][2];              // require min points per leg (case Boff,Bon)
  int mMinDetAcc[utils::NTrackTypes];                 // min number of detector required in track
  double mDefPtBOff[utils::NTrackTypes];              // nominal pt for tracks in Boff run
  double mPtMin[utils::NTrackTypes];                  // min pT of tracks to consider
  double mEtaMax[utils::NTrackTypes];                 // eta cut on tracks
  int mVtxMinCont;                                    // require min number of contributors in Vtx
  int mVtxMaxCont;                                    // require max number of contributors in Vtx
  int mVtxMinContVC;                                  // min number of contributors to use as constraint
  //
  int mMinITSClforVC;     // use vertex constraint for tracks with enough points
  int mITSPattforVC;      // optional request on ITS hits to allow vertex constraint
  double mMaxDCAforVC[2]; // DCA cut in R,Z to allow vertex constraint
  double mMaxChi2forVC;   // track-vertex chi2 cut to allow vertex constraint
  //
  //
  float* mGloParVal; //[mNDOFs] parameters for DOFs
  float* mGloParErr; //[mNDOFs] errors for DOFs
  int* mGloParLab;   //[mNDOFs] labels for DOFs
  int* mOrderedLbl;  //[mNDOFs] ordered labels
  int* mLbl2ID;      //[mNDOFs] Label order in mOrderedLbl -> parID
  //
  AlignmentPoint* mRefPoint; // reference point for track definition
  //
  const TTree* mESDTree; //! externally set esdTree, needed to access UserInfo list
                         //  const AliESDEvent* fESDEvent;             //! externally set event  FIXME(milettri): needs AliESDEvent
                         //  const AliESDtrack* fESDTrack[kNCosmLegs]; //! externally set ESD tracks  FIXME(milettri): needs AliESDtrack
                         //  const AliESDVertex* fVertex;              //! event vertex FIXME(milettri): needs AliESDVertex
  //
  // statistics
  float mStat[kNStatCl][kMaxStat];            // processing statistics
  static const Char_t* sStatClName[kNStatCl]; // stat classes names
  static const Char_t* sStatName[kMaxStat];   // stat type names
  //
  // output related
  float mControlFrac;           //  fraction of tracks to process control residuals
  int mMPOutType;               // What to store as an output, see storeProcessedTrack
  Mille* mMille;                //! Mille interface
  Millepede2Record* mMPRecord;  //! MP record
  ResidualsController* mCResid; //! control residuals
  TTree* mMPRecTree;            //! tree to store MP record
  TTree* mResidTree;            //! tree to store control residuals
  TFile* mMPRecFile;            //! file to store MP record tree
  TFile* mResidFile;            //! file to store control residuals tree
  TArrayF mMilleDBuffer;        //! buffer for Mille Derivatives output
  TArrayI mMilleIBuffer;        //! buffer for Mille Indecis output
  TString mMPDatFileName;       //  file name for records binary data output
  TString mMPParFileName;       //  file name for MP params
  TString mMPConFileName;       //  file name for MP constraints
  TString mMPSteerFileName;     //  file name for MP steering
  TString mResidFileName;       //  file name for optional control residuals
  bool mMilleOutBin;            //  optionally text output for Mille debugging
  bool mDoKalmanResid;          //  calculate residuals with smoothed kalman in the ControlRes
  //
  TString mOutCDBPath;        // output OCDB path
  TString mOutCDBComment;     // optional comment to add to output cdb objects
  TString mOutCDBResponsible; // optional responsible for output metadata
  int mOutCDBRunRange[2];     // run range for output storage
  //
  DOFStatistics* mDOFStat; // stat of entries per dof
  TH1F* mHistoStat;        // histo with general statistics
  //
  // input related
  TString mConfMacroName; // optional configuration macro
  TString mRecoOCDBConf;  // optional macro name for reco-time OCDB setup: void fun(int run)
  TString mRefOCDBConf;   // optional macro name for prealignment OCDB setup: void fun()
  int mRefRunNumber;      // optional run number used for reference
  int mRefOCDBLoaded;     // flag/counter for ref.OCDB loading
  bool mUseRecoOCDB;      // flag to preload reco-time calib objects
  //
  static const int sSkipLayers[kNLrSkip];          // detector layers for which we don't need module matrices
  static const Char_t* sDetectorName[kNDetectors]; // names of detectors
  static const Char_t* sHStatName[kNHVars];        // names for stat.bins in the stat histo
  static const Char_t* sMPDataExt;                 // extension for MP2 binary data
  //
  ClassDef(Controller, 3)
};

//__________________________________________________________
inline void Controller::setMinPointsColl(int vbOff, int vbOn)
{
  // ask min number of points per track
  setMinPoints(utils::Coll, false, vbOff);
  setMinPoints(utils::Coll, true, vbOn);
}

//__________________________________________________________
inline void Controller::setMinPointsCosm(int vbOff, int vbOn)
{
  // ask min number of points per track
  setMinPoints(utils::Cosm, false, vbOff);
  setMinPoints(utils::Cosm, true, vbOn);
}
} // namespace align
} // namespace o2
#endif
