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

/// @file   AlignableDetector.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class for detector: wrapper for set of volumes

#include "Align/Controller.h"
#include "Align/AlignableDetector.h"
#include "Align/AlignableSensor.h"
#include "Align/Controller.h"
#include "Align/AlignmentTrack.h"
#include "Align/GeometricalConstraint.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/NameConf.h"
#include "Framework/Logger.h"
#include <TString.h>
#include <TH1.h>
#include <TTree.h>
#include <TFile.h>
#include <cstdio>
#include <regex>

ClassImp(o2::align::AlignableDetector);

using namespace o2::align::utils;
using GIndex = o2::dataformats::VtxTrackIndex;

namespace o2
{
namespace align
{
//____________________________________________
AlignableDetector::AlignableDetector(DetID id, Controller* ctr) : DOFSet(id.getName(), ctr), mDetID(id)
{
  mVolumes.SetOwner(true);
  mSensors.SetOwner(false); // sensors are just pointers on particular volumes
}

//____________________________________________
AlignableDetector::~AlignableDetector()
{
  // d-tor
  mSensors.Clear();  // sensors are also attached as volumes, don't delete them here
  mVolumes.Delete(); // here all is deleted
}

//____________________________________________
int AlignableDetector::processPoints(GIndex gid, int npntCut, bool inv)
{
  // Create alignment points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  /*
    auto algTrack = mController->getAlgTrack();
    for (clus: clusters_of_track_gid) {
      auto& pnt = mPoints.emplace_back();
      // realign as needed the cluster data
      auto* sensor = getSensor(clus.getSensorID());
      pnt.setXYZTracking(clus.getX(), clus.getY(), clus.getZ());
      pnt.setAlphaSens(sensor->getAlpTracking());
      pnt.setXSens(sensor->getXTracking());
      pnt.setDetID(mDetID);
      pnt.setSID(sensor->getSID());
      //
      pnt.setContainsMeasurement();
      pnt.init();
      algTrack->AddPoint(&pnt);
    }
  */
  LOGP(error, "Detector {} must implement its own ProcessPoints method", getName());
  return 0;
}

//_________________________________________________________
void AlignableDetector::acknowledgeNewRun(int run)
{
  // update parameters needed to process this run

  // detector should be able to undo alignment/calibration used during the reco
  updateL2GRecoMatrices();
}

//_________________________________________________________
void AlignableDetector::updateL2GRecoMatrices()
{
  LOG(fatal) << __PRETTY_FUNCTION__ << " is disabled";
  //FIXME(milettri): needs OCDB
  //  // Update L2G matrices used for data reconstruction
  //  //
  //  AliCDBManager* man = AliCDBManager::Instance();
  //  AliCDBEntry* ent = man->Get(Form("%s/Align/Data", mDetID.getName()));
  //  const TClonesArray* algArr = (const TClonesArray*)ent->GetObject();
  //  //
  //  int nvol = getNVolumes();
  //  for (int iv = 0; iv < nvol; iv++) {
  //    AlignableVolume* vol = getVolume(iv);
  //    // call init for root level volumes, they will take care of their children
  //    if (!vol->getParent()){
  //      vol->updateL2GRecoMatrices(algArr, 0);}
  //  }
  //  //
}

//_________________________________________________________
void AlignableDetector::reset()
{
  // prepare for the next track processing
  mNPoints = 0;
}

//_________________________________________________________
void AlignableDetector::applyAlignmentFromMPSol()
{
  // apply alignment from millepede solution array to reference alignment level
  LOG(info) << "Applying alignment from Millepede solution";
  for (int isn = getNSensors(); isn--;) {
    getSensor(isn)->applyAlignmentFromMPSol();
  }
}

//_________________________________________________________
void AlignableDetector::cacheReferenceCCDB()
{
  LOGP(info, "caching reference CCDB for {}", getName());
  const auto& ggHelper = o2::base::GRPGeomHelper::instance();
  const auto* algVec = ggHelper.getAlignment(mDetID);
  for (const auto& alg : *algVec) {
    AlignableVolume* vol = getVolume(alg.getSymName().c_str());
    if (!vol) {
      LOGP(fatal, "Volume {} not found", alg.getSymName());
    }
    auto mat = alg.createMatrix();
    vol->setGlobalDeltaRef(mat);
  }
}

//_________________________________________________________
void AlignableDetector::addVolume(AlignableVolume* vol)
{
  // add volume
  if (getVolume(vol->getSymName())) {
    LOG(fatal) << "Volume " << vol->GetName() << " was already added to " << mDetID.getName();
  }
  mVolumes.AddLast(vol);
  if (vol->isSensor()) {
    mSensors.AddLast(vol);
    ((AlignableSensor*)vol)->setDetector(this);
    int vid = ((AlignableSensor*)vol)->getVolID();
    if (mVolIDMin < 0 || vid < mVolIDMin) {
      mVolIDMin = vid;
    }
    if (mVolIDMax < 0 || vid > mVolIDMax) {
      mVolIDMax = vid;
    }
  }
  //
}

//_________________________________________________________
void AlignableDetector::defineMatrices()
{
  // define transformation matrices. Detectors may override this method
  //
  TGeoHMatrix mtmp;
  //
  TIter next(&mVolumes);
  AlignableVolume* vol(nullptr);
  while ((vol = (AlignableVolume*)next())) {
    if (vol->isDummy() || vol->isDummyEnvelope()) {
      continue;
    }
    vol->prepareMatrixL2G();      // modified global-local matrix
    vol->prepareMatrixL2GIdeal(); // ideal global-local matrix
  }
  // Now set tracking-local matrix (MUST be done after ALL L2G matrices are done!)
  // Attention: for sensor it is a real tracking matrix extracted from
  // the geometry but for container alignable volumes the tracking frame
  // is used for as the reference for the alignment parameters only,
  // see its definition in the AlignableVolume::PrepateMatrixT2L
  next.Reset();
  while ((vol = (AlignableVolume*)next())) {
    if (vol->isDummy()) {
      continue;
    }
    vol->prepareMatrixT2L();
    if (vol->isSensor()) {
      ((AlignableSensor*)vol)->prepareMatrixClAlg();
    } // alignment matrix
  }
  //
}

//_________________________________________________________
void AlignableDetector::sortSensors()
{
  // build local tables for internal numbering
  mNSensors = mSensors.GetEntriesFast();
  if (!mNSensors) {
    LOG(warning) << "No sensors defined";
    return;
  }
  mSensors.Sort();
  mSID2VolID = new int[mNSensors]; // cash id's for fast binary search RS FIXME DO WE NEED THIS?
  for (int i = 0; i < mNSensors; i++) {
    mSID2VolID[i] = getSensor(i)->getVolID();
    getSensor(i)->setSID(i);
  }
  //
}

//_________________________________________________________
int AlignableDetector::initGeom()
{
  // define hiearchy, initialize matrices, return number of global parameters
  if (getInitGeomDone()) {
    return 0;
  }
  //
  defineVolumes();
  sortSensors(); // VolID's must be in increasing order
  defineMatrices();
  //
  // calculate number of global parameters
  int nvol = getNVolumes();
  mNDOFs = 0;
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    mNDOFs += vol->getNDOFs();
  }
  //
  mNDOFs += mNCalibDOFs;
  setInitGeomDone();
  return mNDOFs;
}

//_________________________________________________________
int AlignableDetector::assignDOFs()
{
  // assign DOFs IDs, parameters
  //
  setFirstParGloID(mController->getNDOFs());
  if (mFirstParGloID == (int)mController->getGloParVal().size() && mNCalibDOFs) { // new detector is being added
    mController->expandGlobalsBy(mNCalibDOFs);
  }
  for (int icl = 0; icl < mNCalibDOFs; icl++) {
    setParLab(icl, icl); // TODO RS FIXME
  }
  //
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    if (!vol->getParent()) { // call init for root level volumes, they will take care of their children
      vol->assignDOFs();
    }
  }
  return mNDOFs;
}

//_________________________________________________________
void AlignableDetector::initDOFs()
{
  // initialize free parameters
  if (getInitDOFsDone()) {
    LOG(fatal) << "DOFs are already initialized for " << mDetID.getName();
  }
  //
  auto pars = getParVals();
  auto errs = getParErrs();
  // process calibration DOFs
  for (int i = 0; i < mNCalibDOFs; i++) {
    if (errs[i] < -9999 && isZeroAbs(pars[i])) {
      fixDOF(i);
    }
  }
  //
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) {
    getVolume(iv)->initDOFs();
  }
  //
  calcFree(true);
  //
  setInitDOFsDone();
  return;
}

//_________________________________________________________
int AlignableDetector::volID2SID(int vid) const
{
  // find SID corresponding to VolID
  int mn(0), mx(mNSensors - 1);
  while (mx >= mn) {
    int md((mx + mn) >> 1), vids(getSensor(md)->getVolID());
    if (vid < vids) {
      mx = md - 1;
    } else if (vid > vids) {
      mn = md + 1;
    } else {
      return md;
    }
  }
  return -1;
}

//____________________________________________
void AlignableDetector::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("\nDetector:%5s %5d volumes %5d sensors {VolID: %5d-%5d} Def.Sys.Err: %.4e %.4e | Stat:%d\n",
         mDetID.getName(), getNVolumes(), getNSensors(), getVolIDMin(),
         getVolIDMax(), mAddError[0], mAddError[1], mNProcPoints);
  //
  printf("Errors assignment: ");
  if (mUseErrorParam) {
    printf("param %d\n", mUseErrorParam);
  } else {
    printf("from TrackPoints\n");
  }
  //
  printf("Allowed    in Collisions: %7s | Cosmic: %7s\n",
         isDisabled(Coll) ? "  NO " : " YES ", isDisabled(Cosm) ? "  NO " : " YES ");
  //
  printf("Obligatory in Collisions: %7s | Cosmic: %7s\n",
         isObligatory(Coll) ? " YES " : "  NO ", isObligatory(Cosm) ? " YES " : "  NO ");
  //
  printf("Min.points in Collisions: %7d | Cosmic: %7d\n", mNPointsSel[Coll], mNPointsSel[Cosm]);
  //
  if (!(IsDisabledColl() && IsDisabledCosm()) && opts.Contains("long")) {
    for (int iv = 0; iv < getNVolumes(); iv++) {
      getVolume(iv)->Print(opt);
    }
  }
  //
  for (int i = 0; i < getNCalibDOFs(); i++) {
    printf("CalibDOF%2d: %-20s\t%e\n", i, getCalibDOFName(i), getCalibDOFValWithCal(i));
  }
}

//____________________________________________
void AlignableDetector::setAddError(double sigy, double sigz)
{
  // add syst error to all sensors
  LOG(info) << "Adding sys.error " << std::fixed << std::setprecision(4) << sigy << " " << sigz << " to all sensors";
  mAddError[0] = sigy;
  mAddError[1] = sigz;
  for (int isn = getNSensors(); isn--;) {
    getSensor(isn)->setAddError(sigy, sigz);
  }
  //
}

//____________________________________________
void AlignableDetector::setUseErrorParam(int v)
{
  // set type of points error parameterization
  LOG(fatal) << "setUseErrorParam is not implemented for this detector";
  //
}

//____________________________________________
void AlignableDetector::updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const
{
  // update point using specific error parameterization
  LOG(fatal) << "If needed, this method has to be implemented for specific detector";
}

//____________________________________________
void AlignableDetector::defineVolumes()
{
  // define alignment volumes
  LOG(fatal) << "defineVolumes method has to be implemented for specific detector";
}

//____________________________________________
void AlignableDetector::setObligatory(int tp, bool v)
{
  // mark detector presence obligatory in the track
  mObligatory[tp] = v;
  mController->setObligatoryDetector(getDetID(), tp, v);
}

//______________________________________________________
void AlignableDetector::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  fprintf(parOut, "\n!!\t\tDetector:\t%s\tNDOFs: %d\n", mDetID.getName(), getNDOFs());
  //
  // parameters
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) { // call for root level volumes, they will take care of their children
    AlignableVolume* vol = getVolume(iv);
    if (!vol->getParent()) {
      vol->writePedeInfo(parOut, opt);
    }
  }
  //
}

//______________________________________________________
void AlignableDetector::writeLabeledPedeResults(FILE* parOut) const
{
  // contribute to params and constraints template files for PEDE
  fprintf(parOut, "\n!!\t\tDetector:\t%s\tNDOFs: %d\n", mDetID.getName(), getNDOFs());
  //
  // parameters
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) { // call for root level volumes, they will take care of their children
    AlignableVolume* vol = getVolume(iv);
    if (!vol->getParent()) {
      vol->writeLabeledPedeResults(parOut);
    }
  }
  //
}

//______________________________________________________
void AlignableDetector::writeCalibrationResults() const
{
  // store calibration results
  writeAlignmentResults();
  //
  // eventually we may write other calibrations
}

//______________________________________________________
void AlignableDetector::writeAlignmentResults() const
{
  std::vector<o2::detectors::AlignParam> arr;
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    // call only for top level objects, they will take care of children
    if (!vol->getParent()) {
      vol->createAlignmentObjects(arr);
    }
  }
  TFile outalg(fmt::format("alignment{}.root", getName()).c_str(), "recreate");
  outalg.WriteObjectAny(&arr, "std::vector<o2::detectors::AlignParam>", o2::base::NameConf::CCDBOBJECT.data());
  outalg.Close();
  LOGP(info, "storing {} alignment in {}", getName(), outalg.GetName());
}

//______________________________________________________
bool AlignableDetector::ownsDOFID(int id) const
{
  // check if DOF ID belongs to this detector
  for (int iv = getNVolumes(); iv--;) {
    AlignableVolume* vol = getVolume(iv); // check only top level volumes
    if (!vol->getParent() && vol->ownsDOFID(id)) {
      return true;
    }
  }
  // calibration DOF?
  if (id >= mFirstParGloID && id < mFirstParGloID + mNCalibDOFs) {
    return true;
  }
  //
  return false;
}

//______________________________________________________
AlignableVolume* AlignableDetector::getVolOfDOFID(int id) const
{
  // gets volume owning this DOF ID
  for (int iv = getNVolumes(); iv--;) {
    AlignableVolume* vol = getVolume(iv);
    if (vol->getParent()) {
      continue;
    } // check only top level volumes
    if ((vol = vol->getVolOfDOFID(id))) {
      return vol;
    }
  }
  return nullptr;
}

//______________________________________________________
void AlignableDetector::terminate()
{
  // called at the end of processing
  //  if (isDisabled()) return;
  int nvol = getNVolumes();
  mNProcPoints = 0;
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->getParent()) {
      mNProcPoints += vol->finalizeStat();
    }
  }
}

//________________________________________
void AlignableDetector::addAutoConstraints() const
{
  // adds automatic constraints
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) { // call for root level volumes, they will take care of their children
    AlignableVolume* vol = getVolume(iv);
    if (!vol->getParent()) {
      vol->addAutoConstraints();
    }
  }
}

//________________________________________
void AlignableDetector::fixNonSensors()
{
  // fix all non-sensor volumes
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (vol->isSensor()) {
      continue;
    }
    vol->setFreeDOFPattern(0);
    vol->setChildrenConstrainPattern(0);
  }
}

//________________________________________
int AlignableDetector::selectVolumes(std::vector<AlignableVolume*> cont, int lev, const std::string& regexStr)
{
  // select volumes matching to pattern and/or hierarchy level
  //
  std::regex selRegEx(regexStr);
  int nadd = 0;
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!regexStr.empty() && !std::regex_match(vol->getSymName(), selRegEx)) {
      continue;
    }
    cont.push_back(vol);
    nadd++;
  }
  //
  return nadd;
}

//________________________________________
void AlignableDetector::setFreeDOFPattern(uint32_t pat, int lev, const std::string& regexStr)
{
  // set free DOFs to volumes matching either to hierarchy level or whose name contains match
  //
  std::regex selRegEx(regexStr);
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!regexStr.empty() && !std::regex_match(vol->getSymName(), selRegEx)) {
      continue;
    } // wrong name
    vol->setFreeDOFPattern(pat);
  }
  //
}

//________________________________________
void AlignableDetector::setDOFCondition(int dof, float condErr, int lev, const std::string& regexStr)
{
  // set condition for DOF of volumes matching either to hierarchy level or
  // whose name contains match
  //
  std::regex selRegEx(regexStr);
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!regexStr.empty() && !std::regex_match(vol->getSymName(), selRegEx)) {
      continue;
    } // wrong name
    if (dof >= vol->getNDOFs()) {
      continue;
    }
    vol->setParErr(dof, condErr);
    if (condErr >= 0 && !vol->isFreeDOF(dof)) {
      vol->setFreeDOF(dof);
    }
    //if (condErr<0  && vol->isFreeDOF(dof)) vol->fixDOF(dof);
  }
  //
}

//________________________________________
void AlignableDetector::constrainOrphans(const double* sigma, const char* match)
{
  // additional constraint on volumes w/o parents (optionally containing "match" in symname)
  // sigma<0 : dof is not contrained
  // sigma=0 : dof constrained exactly (Lagrange multiplier)
  // sigma>0 : dof constrained by gaussian constraint
  //
  TString mts = match, syms;
  auto cstr = getController()->getConstraints().emplace_back();
  for (int i = 0; i < AlignableVolume::kNDOFGeom; i++) {
    if (sigma[i] >= 0) {
      cstr.constrainDOF(i);
    } else {
      cstr.unConstrainDOF(i);
    }
    cstr.setSigma(i, sigma[i]);
  }
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (vol->getParent()) {
      continue;
    } // wrong level
    if (!mts.IsNull() && !(syms = vol->getSymName()).Contains(mts)) {
      continue;
    } //wrong name
    cstr.addChild(vol);
  }
  //
  if (!cstr.getNChildren()) {
    LOG(info) << "No volume passed filter " << match;
    getController()->getConstraints().pop_back();
  }
}

//________________________________________
void AlignableDetector::setFreeDOF(int dof)
{
  // set detector free dof
  if (dof >= kNMaxKalibDOF) {
    LOG(fatal) << "Detector CalibDOFs limited to " << kNMaxKalibDOF << ", requested " << dof;
  }
  mCalibDOF |= 0x1 << dof;
  calcFree();
}

//________________________________________
void AlignableDetector::fixDOF(int dof)
{
  // fix detector dof
  if (dof >= kNMaxKalibDOF) {
    LOG(fatal) << "Detector CalibDOFs limited to " << kNMaxKalibDOF << ", requested " << dof;
  }
  mCalibDOF &= ~(0x1 << dof);
  calcFree();
}

//__________________________________________________________________
bool AlignableDetector::isCondDOF(int i) const
{
  // is DOF free and conditioned?
  return (!isZeroAbs(getParVal(i)) || !isZeroAbs(getParErr(i)));
}

//__________________________________________________________________
void AlignableDetector::calcFree(bool condFix)
{
  // calculate free calib dofs. If condFix==true, condition parameter a la pede, i.e. error < 0
  mNCalibDOFsFree = 0;
  for (int i = 0; i < mNCalibDOFs; i++) {
    if (!isFreeDOF(i)) {
      if (condFix && varsSet()) {
        setParErr(i, -999);
      }
      continue;
    }
    mNCalibDOFsFree++;
  }
  //
}

//______________________________________________________
void AlignableDetector::writeSensorPositions(const char* outFName)
{
  // create tree with sensors ideal, ref and reco positions
  int ns = getNSensors();
  double loc[3] = {0};
  // ------- local container type for dumping sensor positions ------
  typedef struct {
    int volID;     // volume id
    double pId[3]; // ideal
    double pRf[3]; // reference
    double pRc[3]; // reco-time
  } snpos_t;
  snpos_t spos; //
  TFile* fl = TFile::Open(outFName, "recreate");
  TTree* tr = new TTree("snpos", Form("sensor poisitions for %s", mDetID.getName()));
  tr->Branch("volID", &spos.volID, "volID/I");
  tr->Branch("pId", &spos.pId, "pId[3]/D");
  tr->Branch("pRf", &spos.pRf, "pRf[3]/D");
  tr->Branch("pRc", &spos.pRc, "pRc[3]/D");
  //
  for (int isn = 0; isn < ns; isn++) {
    AlignableSensor* sens = getSensor(isn);
    spos.volID = sens->getVolID();
    sens->getMatrixL2GIdeal().LocalToMaster(loc, spos.pId);
    sens->getMatrixL2G().LocalToMaster(loc, spos.pRf);
    sens->getMatrixL2GReco().LocalToMaster(loc, spos.pRc);
    tr->Fill();
  }
  tr->Write();
  delete tr;
  fl->Close();
  delete fl;
}

} // namespace align
} // namespace o2
