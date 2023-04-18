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
#include "ReconstructionDataFormats/TrackCosmics.h"

#include "Align/Millepede2Record.h"
#include "Align/ResidualsController.h"
#include "Align/GeometricalConstraint.h"

#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <string>
#include <TArrayF.h>
#include <TArrayI.h>
#include <TH1F.h>
#include "Align/utils.h"
#include "Framework/TimingInfo.h"
#include "Align/AlignableDetector.h"

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
namespace trd
{
class TrackletTransformer;
}
namespace utils
{
class TreeStreamRedirector;
}

namespace align
{

//class Mille;

class EventVertex;
class AlignableVolume;
class AlignmentPoint;
class ResidualsControllerFast;

class Controller : public TObject
{
 public:
  struct ProcStat {
    enum {
      kInput,
      kAccepted,
      kNStatCl
    };
    enum {
      kVertices,
      kTracks,
      kTracksWithVertex,
      kCosmic,
      kMaxStat
    };
    std::array<std::array<size_t, kMaxStat>, kNStatCl> data{};
    void print() const;
  };

  using DetID = o2::detectors::DetID;
  using GTrackID = o2::dataformats::GlobalTrackID;

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
  enum { kInitGeomDone = BIT(14),
         kInitDOFsDone = BIT(15),
         kMPAlignDone = BIT(16) };

  Controller() = default;
  Controller(DetID::mask_t detmask, GTrackID::mask_t trcmask, bool cosmic = false, bool useMC = false, int instID = 0);
  ~Controller() final;

  void expandGlobalsBy(int n);
  void process();
  void processCosmic();

  bool getUseRecoOCDB() const { return mUseRecoOCDB; }
  void setUseRecoOCDB(bool v = true) { mUseRecoOCDB = v; }

  void initDetectors();
  void initDOFs();
  void terminate();
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
  int getNConstraints() const { return mConstraints.size(); }
  const std::vector<GeometricalConstraint>& getConstraints() const { return mConstraints; }
  std::vector<GeometricalConstraint>& getConstraints() { return mConstraints; }
  const GeometricalConstraint& getConstraint(int i) const { return mConstraints[i]; }

  void addAutoConstraints();
  //
  void setTimingInfo(const o2::framework::TimingInfo& ti);
  bool getFieldOn() const { return mFieldOn; }
  void setFieldOn(bool v = true) { mFieldOn = v; }
  int getTracksType() const { return mTracksType; }
  void setTracksType(int t = utils::Coll) { mTracksType = t; }
  bool isCosmic() const { return mTracksType == utils::Cosm; }
  bool isCollision() const { return mTracksType == utils::Coll; }
  void setCosmic(bool v = true) { mTracksType = v ? utils::Cosm : utils::Coll; }
  auto getStat(int cls, int tp) const { return mStat.data[cls][tp]; }
  auto& getStat() const { return mStat; }
  //
  bool checkDetectorPattern(DetID::mask_t patt) const;
  bool checkDetectorPoints(const int* npsel) const;
  void setObligatoryDetector(DetID id, int tp, bool v = true);
  //
  //  void SetVertex(const AliESDVertex* v) { fVertex = v; } FIXME(milettri): needs AliESDVertex
  //  const AliESDVertex* GetVertex() const { return fVertex; } FIXME(milettri): needs AliESDVertex
  //
  //----------------------------------------
  bool readParameters(const std::string& parfile = "millepede.res", bool useErrors = true);
  auto& getGloParVal() { return mGloParVal; }
  auto& getGloParErr() { return mGloParErr; }
  auto& getGloParLab() { return mGloParLab; }
  int getGloParLab(int i) const { return mGloParLab[i]; }
  int parID2Label(int i) const { return getGloParLab(i); }
  int label2ParID(int lab) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  AlignableVolume* getVolOfLabel(int label) const;
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

  bool addVertexConstraint(const o2::dataformats::PrimaryVertex& vtx);
  int getNDetectors() const { return mNDet; }
  AlignableDetector* getDetector(DetID id) const { return mDetectors[id].get(); }

  EventVertex* getVertexSensor() const { return mVtxSens.get(); }
  //
  void resetForNextTrack();
  int getNDOFs() const { return mGloParVal.size(); }
  //----------------------------------------
  float getControlFrac() const { return mControlFrac; }
  void setControlFrac(float v = 1.) { mControlFrac = v; }
  void writeCalibrationResults() const;
  void applyAlignmentFromMPSol();
  //
  bool fillMPRecData(o2::dataformats::GlobalTrackID tid);
  bool fillControlData(o2::dataformats::GlobalTrackID tid);
  bool fillMilleData();

  void closeMPRecOutput();
  void closeMilleOutput();
  void closeResidOutput();
  void initMPRecOutput();
  void initMIlleOutput();
  void initResidOutput();
  bool storeProcessedTrack(o2::dataformats::GlobalTrackID tid);
  void printStatistics() const;
  //
  void genPedeSteerFile(const Option_t* opt = "") const;
  void writeLabeledPedeResults() const;
  void writePedeConstraints() const;
  void checkConstraints(const char* params = nullptr);
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
  static void MPRec2Mille(const std::string& mprecfile, const std::string& millefile = "mpData.mille", bool bindata = true);
  static void MPRec2Mille(TTree* mprTree, const std::string& millefile = "mpData.mille", bool bindata = true);
  //
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

  void setTrackSourceMask(GTrackID::mask_t m) { mMPsrc = m; }
  GTrackID::mask_t getTrackSourceMask() const { return mMPsrc; }

  void setTRDTransformer(const o2::trd::TrackletTransformer* trans) { mTRDTransformer = trans; }
  void setTRDTrigRecFilterActive(bool v) { mTRDTrigRecFilterActive = v; }
  void setAllowAfterburnerTracks(bool v) { mAllowAfterburnerTracks = v; }

  const o2::trd::TrackletTransformer* getTRDTransformer() const { return mTRDTransformer; }
  bool getTRDTrigRecFilterActive() const { return mTRDTrigRecFilterActive; }
  bool getAllowAfterburnerTracks() const { return mAllowAfterburnerTracks; }

  int getInstanceID() const { return mInstanceID; }
  void setInstanceID(int i) { mInstanceID = i; }

  int getDebugOutputLevel() const { return mDebugOutputLevel; }
  void setDebugOutputLevel(int i) { mDebugOutputLevel = i; }
  void setDebugStream(o2::utils::TreeStreamRedirector* d) { mDBGOut = d; }

 protected:
  //
  // --------- dummies -----------
  Controller(const Controller&);
  Controller& operator=(const Controller&);
  //
 protected:
  //
  DetID::mask_t mDetMask{};
  GTrackID::mask_t mMPsrc{};
  std::vector<int> mTrackSources;
  o2::framework::TimingInfo mTimingInfo{};
  int mInstanceID = 0; // instance in case of pipelining
  int mRunNumber = 0;
  int mNDet = 0;                             // number of deectors participating in the alignment
  int mNDOFs = 0;                            // number of degrees of freedom
  bool mUseMC = false;
  bool mFieldOn = false;                     // field on flag
  int mTracksType = utils::Coll;             // collision/cosmic event type
  float mMPRecOutFraction = 0.;
  float mControlFraction = 0.;
  std::unique_ptr<AlignmentTrack> mAlgTrack; // current alignment track
  const o2::globaltracking::RecoContainer* mRecoData = nullptr; // externally set RecoContainer
  const o2::trd::TrackletTransformer* mTRDTransformer = nullptr;  // TRD tracket transformer
  bool mTRDTrigRecFilterActive = false;                           // select TRD triggers processed with ITS
  bool mAllowAfterburnerTracks = false;                           // allow using ITS-TPC afterburner tracks
  std::array<std::unique_ptr<AlignableDetector>, DetID::nDetectors> mDetectors{}; // detectors participating in the alignment

  std::unique_ptr<EventVertex> mVtxSens; // fake sensor for the vertex
  std::vector<GeometricalConstraint> mConstraints{}; // array of constraints
  //
  // Track selection
  std::array<DetID::mask_t, utils::NTrackTypes> mObligatoryDetPattern{}; // pattern of obligatory detectors
  //
  std::vector<float> mGloParVal; // parameters for DOFs
  std::vector<float> mGloParErr; // errors for DOFs
  std::vector<int> mGloParLab;   // labels for DOFs
  std::unordered_map<int, int> mLbl2ID; // Labels mapping to parameter ID
  //
  std::unique_ptr<AlignmentPoint> mRefPoint; //! reference point for track definition
  //
  int mDebugOutputLevel = 0;
  o2::utils::TreeStreamRedirector* mDBGOut = nullptr;

  // statistics
  ProcStat mStat{}; // processing statistics
  int mNTF = 0;
  //
  // output related
  float mControlFrac = 1.0;                    //  fraction of tracks to process control residuals
  std::unique_ptr<Mille> mMille;               //! Mille interface
  Millepede2Record mMPRecord;                  //! MP record
  Millepede2Record* mMPRecordPtr = &mMPRecord; //! MP record
  ResidualsController mCResid;                 //! control residuals
  ResidualsController* mCResidPtr = &mCResid;  //! control residuals

  std::unique_ptr<TTree> mMPRecTree; //! tree to store MP record
  std::unique_ptr<TTree> mResidTree; //! tree to store control residuals
  std::unique_ptr<TFile> mMPRecFile; //! file to store MP record tree
  std::unique_ptr<TFile> mResidFile; //! file to store control residuals tree
  std::string mMilleFileName{};      //!
  //
  // input related
  int mRefRunNumber = 0;    // optional run number used for reference
  int mRefOCDBLoaded = 0;   // flag/counter for ref.OCDB loading
  bool mUseRecoOCDB = true; // flag to preload reco-time calib objects
  //
  static const int sSkipLayers[kNLrSkip];          // detector layers for which we don't need module matrices
  static const Char_t* sDetectorName[kNDetectors]; // names of detectors //RSREM
  static const Char_t* sMPDataExt;                 // extension for MP2 binary data
  static const Char_t* sMPDataTxtExt;              // extension for MP2 txt data
  //
  ClassDefOverride(Controller, 1)
};

} // namespace align
} // namespace o2
#endif
