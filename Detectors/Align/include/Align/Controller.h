// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "Align/AlignmentTrack.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
// #include "AliSymMatrix.h" FIXME(milettri): needs AliSymMatrix

#include "Align/Millepede2Record.h"
#include "Align/ResidualsController.h"

#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <string>
#include <TArrayF.h>
#include <TArrayI.h>
#include <TH1F.h>
#include "Align/utils.h"

// can be fwd declared if we don't require root dict.
//class TTree;
//class TFile;

#include <TTree.h>
#include <TFile.h>
#include "Align/Mille.h"

namespace o2
{
namespace globaltracking
{
class RecoContainer;
}

namespace align
{

//class Mille;

class EventVertex;
class AlignableDetector;
class AlignableVolume;
class AlignmentPoint;
class ResidualsControllerFast;
class GeometricalConstraint;
class DOFStatistics;

class Controller : public TObject
{
 public:
  struct ProcStat {
    enum { kInput,
           kAccepted,
           kNStatCl };
    enum { kRun,
           kEventColl,
           kEventCosm,
           kTrackColl,
           kTrackCosm,
           kMaxStat };
    std::array<std::array<int, kMaxStat>, kNStatCl> data{};
    void print() const;
  };

  using DetID = o2::detectors::DetID;

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

  Controller() = default;
  Controller(DetID::mask_t detmask);
  ~Controller() final;

  void expandGlobalsBy(int n);
  void process();

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
  float getStat(int cls, int tp) const { return mStat.data[cls][tp]; }
  //
  bool checkDetectorPattern(DetID::mask_t patt) const;
  bool checkDetectorPoints(const int* npsel) const;
  void setObligatoryDetector(DetID id, int tp, bool v = true);
  //
  //  void SetVertex(const AliESDVertex* v) { fVertex = v; } FIXME(milettri): needs AliESDVertex
  //  const AliESDVertex* GetVertex() const { return fVertex; } FIXME(milettri): needs AliESDVertex
  //
  //----------------------------------------
  bool readParameters(const char* parfile = "millepede.res", bool useErrors = true);
  auto& getGloParVal() { return mGloParVal; }
  auto& getGloParErr() { return mGloParErr; }
  auto& getGloParLab() { return mGloParLab; }
  int getGloParLab(int i) const { return mGloParLab[i]; }
  int parID2Label(int i) const { return getGloParLab(i); }
  int label2ParID(int lab) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  AlignableDetector* getDetOfDOFID(int id) const;
  //
  AlignmentPoint* getRefPoint() const { return mRefPoint.get(); }
  //
  const ResidualsController& getContResid() const { return mCResid; }
  const Millepede2Record& getMPRecord() const { return mMPRecord; }
  TTree* getMPRecTree() const { return mMPRecTree.get(); }
  AlignmentTrack* getAlgTrack() const { return mAlgTrack.get(); }

  const o2::globaltracking::RecoContainer* getRecoContainer() const { return mRecoData; }
  void setRecoContainer(const o2::globaltracking::RecoContainer* cont) { mRecoData = cont; }

  //  bool ProcessEvent(const AliESDEvent* esdEv); FIXME(milettri): needs AliESDEvent
  //  bool ProcessTrack(const AliESDtrack* esdTr); FIXME(milettri): needs AliESDtrack
  //  bool ProcessTrack(const AliESDCosmicTrack* esdCTr); FIXME(milettri): needs AliESDCosmicTrack
  //  uint32_t AcceptTrack(const AliESDtrack* esdTr, bool strict = true) const; FIXME(milettri): needs AliESDtrack
  //  uint32_t AcceptTrackCosmic(const AliESDtrack* esdPairCosm[kNCosmLegs]) const; FIXME(milettri): needs AliESDtrack
  //  bool CheckSetVertex(const AliESDVertex* vtx); FIXME(milettri): needs AliESDVertex
  bool addVertexConstraint(const o2::dataformats::PrimaryVertex& vtx);
  int getNDetectors() const { return mNDet; }
  AlignableDetector* getDetector(DetID id) const { return mDetectors[id]; }

  EventVertex* getVertexSensor() const { return mVtxSens.get(); }
  //
  void resetForNextTrack();
  int getNDOFs() const { return mGloParVal.size(); }
  //----------------------------------------
  // output related
  void setMPDatFileName(const char* name = "mpData");
  void setMPParFileName(const char* name = "mpParams.txt");
  void setMPConFileName(const char* name = "mpConstraints.txt");
  void setMPSteerFileName(const char* name = "mpSteer.txt");
  void setResidFileName(const char* name = "mpControlRes.root");
  void setOutCDBPath(const char* name = "local://outOCDB");
  void setOutCDBComment(const char* cm = nullptr) { mOutCDBComment = cm; }
  void setOutCDBResponsible(const char* v = nullptr) { mOutCDBResponsible = v; }
  //  void SetOutCDBRunRange(int rmin = 0, int rmax = 999999999); FIXME(milettri): needs OCDB
  int* getOutCDBRunRange() const { return (int*)mOutCDBRunRange; }
  int getOutCDBRunMin() const { return mOutCDBRunRange[0]; }
  int getOutCDBRunMax() const { return mOutCDBRunRange[1]; }
  float getControlFrac() const { return mControlFrac; }
  void setControlFrac(float v = 1.) { mControlFrac = v; }
  //  void writeCalibrationResults() const; FIXME(milettri): needs OCDB
  void applyAlignmentFromMPSol();
  const char* getOutCDBComment() const { return mOutCDBComment.c_str(); }
  const char* getOutCDBResponsible() const { return mOutCDBResponsible.c_str(); }
  const char* getOutCDBPath() const { return mOutCDBPath.c_str(); }
  const char* getMPDatFileName() const { return mMPDatFileName.c_str(); }
  const char* getResidFileName() const { return mResidFileName.c_str(); }
  const char* getMPParFileName() const { return mMPParFileName.c_str(); }
  const char* getMPConFileName() const { return mMPConFileName.c_str(); }
  const char* getMPSteerFileName() const { return mMPSteerFileName.c_str(); }
  //
  bool fillMPRecData();
  bool fillMilleData();
  bool fillControlData();
  void setDoKalmanResid(bool v = true) { mDoKalmanResid = v; }
  void setMPOutType(int t) { mMPOutType = t; }
  void produceMPData(bool v = true)
  {
    if (v) {
      mMPOutType |= kMille;
    } else {
      mMPOutType &= ~kMille;
    }
  }
  void produceMPRecord(bool v = true)
  {
    if (v) {
      mMPOutType |= kMPRec;
    } else {
      mMPOutType &= ~kMPRec;
    }
  }
  void produceControlRes(bool v = true)
  {
    if (v) {
      mMPOutType |= kContR;
    } else {
      mMPOutType &= ~kContR;
    }
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
  void checkConstraints(const char* params = nullptr);
  DOFStatistics& GetDOFStat() { return mDOFStat; }
  void setDOFStat(const DOFStatistics& st) { mDOFStat = st; }
  TH1* getHistoStat() const { return mHistoStat; }
  void detachHistoStat() { setHistoStat(nullptr); }
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
  int getRefOCDBLoaded() const { return mRefOCDBLoaded; }
  //
  void Print(const Option_t* opt = "") const final;
  void printLabels() const;
  Char_t* getDOFLabelTxt(int idf) const;
  //
  static Char_t* getDetNameByDetID(int id) { return (Char_t*)sDetectorName[id]; } //RSREM
  static void mPRec2Mille(const char* mprecfile, const char* millefile = "mpData.mille", bool bindata = true);
  static void mPRec2Mille(TTree* mprTree, const char* millefile = "mpData.mille", bool bindata = true);
  //
  //  AliSymMatrix* BuildMatrix(TVectorD& vec); FIXME(milettri): needs AliSymMatrix
  bool testLocalSolution();
  //
  // fast check of solution using derivatives
  void checkSol(TTree* mpRecTree, bool store = true, bool verbose = false, bool loc = true, const char* outName = "resFast");
  bool checkSol(Millepede2Record* rec, ResidualsControllerFast* rLG = nullptr, ResidualsControllerFast* rL = nullptr, bool verbose = true, bool loc = true);
  //
  // RSTMP new code
  void init();

  void setDetectorsMask(DetID::mask_t m) { mDetMask = m; }
  DetID::mask_t getDetectorsMask() const { return mDetMask; }

 protected:
  //
  // --------- dummies -----------
  Controller(const Controller&);
  Controller& operator=(const Controller&);
  //
 protected:
  //
  DetID::mask_t mDetMask{};

  int mNDet = 0;                             // number of deectors participating in the alignment
  int mNDOFs = 0;                            // number of degrees of freedom
  int mRunNumber = -1;                       // current run number
  bool mFieldOn = false;                     // field on flag
  int mTracksType = utils::Coll;             // collision/cosmic event type
  std::unique_ptr<AlignmentTrack> mAlgTrack; // current alignment track
  const o2::globaltracking::RecoContainer* mRecoData = nullptr; // externally set RecoContainer

  std::array<AlignableDetector*, DetID::nDetectors> mDetectors{}; // detectors participating in the alignment

  std::unique_ptr<EventVertex> mVtxSens; // fake sensor for the vertex
  TObjArray mConstraints{};              // array of constraints
  //
  // Track selection
  std::array<DetID::mask_t, utils::NTrackTypes> mObligatoryDetPattern{}; // pattern of obligatory detectors
  //
  std::vector<float> mGloParVal; // parameters for DOFs
  std::vector<float> mGloParErr; // errors for DOFs
  std::vector<int> mGloParLab;   // labels for DOFs
  std::vector<int> mOrderedLbl;  //ordered labels
  std::vector<int> mLbl2ID;      //Label order in mOrderedLbl -> parID
  //
  std::unique_ptr<AlignmentPoint> mRefPoint; //! reference point for track definition
  //
  // statistics
  ProcStat mStat{}; // processing statistics
  //
  // output related
  float mControlFrac = 1.0;                    //  fraction of tracks to process control residuals
  int mMPOutType = kMille | kMPRec | kContR;   // What to store as an output, see storeProcessedTrack
  std::unique_ptr<Mille> mMille;               //! Mille interface
  Millepede2Record mMPRecord;                  //! MP record
  Millepede2Record* mMPRecordPtr = &mMPRecord; //! MP record
  ResidualsController mCResid;                 //! control residuals
  ResidualsController* mCResidPtr = &mCResid;  //! control residuals

  std::unique_ptr<TTree> mMPRecTree; //! tree to store MP record
  std::unique_ptr<TTree> mResidTree; //! tree to store control residuals
  std::unique_ptr<TFile> mMPRecFile; //! file to store MP record tree
  std::unique_ptr<TFile> mResidFile; //! file to store control residuals tree
  TArrayF mMilleDBuffer;        //! buffer for Mille Derivatives output
  TArrayI mMilleIBuffer;        //! buffer for Mille Indecis output
  std::string mMPDatFileName{"mpData"};            //  file name for records binary data output
  std::string mMPParFileName{"mpParams.txt"};      //  file name for MP params
  std::string mMPConFileName{"mpConstraints.txt"}; //  file name for MP constraints
  std::string mMPSteerFileName{"mpSteer.txt"};     //  file name for MP steering
  std::string mResidFileName{"mpContolRes.root"};  //  file name for optional control residuals
  bool mMilleOutBin = true;                        //  optionally text output for Mille debugging
  bool mDoKalmanResid = true;                      //  calculate residuals with smoothed kalman in the ControlRes
  //
  std::string mOutCDBPath{};        // output OCDB path
  std::string mOutCDBComment{};     // optional comment to add to output cdb objects
  std::string mOutCDBResponsible{}; // optional responsible for output metadata
  int mOutCDBRunRange[2] = {};      // run range for output storage
  //
  DOFStatistics mDOFStat;     // stat of entries per dof
  TH1F* mHistoStat = nullptr; // histo with general statistics
  //
  // input related
  int mRefRunNumber = 0;    // optional run number used for reference
  int mRefOCDBLoaded = 0;   // flag/counter for ref.OCDB loading
  bool mUseRecoOCDB = true; // flag to preload reco-time calib objects
  //
  static const int sSkipLayers[kNLrSkip];          // detector layers for which we don't need module matrices
  static const Char_t* sDetectorName[kNDetectors]; // names of detectors //RSREM
  static const Char_t* sHStatName[kNHVars];        // names for stat.bins in the stat histo
  static const Char_t* sMPDataExt;                 // extension for MP2 binary data
  //
  ClassDefOverride(Controller, 1)
};

} // namespace align
} // namespace o2
#endif
