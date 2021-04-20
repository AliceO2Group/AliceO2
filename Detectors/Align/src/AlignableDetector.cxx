// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetector.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class for detector: wrapper for set of volumes

#include "Align/AlignableDetector.h"
#include "Align/AlignableSensor.h"
#include "Align/Controller.h"
#include "Align/AlignmentTrack.h"
#include "Align/DOFStatistics.h"
#include "Align/GeometricalConstraint.h"
#include "Framework/Logger.h"
//#include "AliGeomManager.h"
//#include "AliCDBManager.h"
//#include "AliCDBMetaData.h"
//#include "AliCDBEntry.h"
//#include "AliAlignObj.h"
//#include "AliCDBId.h"
//#include "AliExternalTrackParam.h"
//#include "AliAlignObjParams.h"
#include <TString.h>
#include <TH1.h>
#include <TTree.h>
#include <TFile.h>
#include <cstdio>

ClassImp(o2::align::AlignableDetector);

using namespace o2::align::utils;

namespace o2
{
namespace align
{

//____________________________________________
AlignableDetector::AlignableDetector()
  : mNDOFs(0), mVolIDMin(-1), mVolIDMax(-1), mNSensors(0), mSID2VolID(nullptr), mNProcPoints(0)
    //
    ,
    mNCalibDOF(0),
    mNCalibDOFFree(0),
    mCalibDOF(0),
    mFirstParGloID(-1),
    mParVals(nullptr),
    mParErrs(nullptr),
    mParLabs(nullptr)
    //
    ,
    mUseErrorParam(0),
    mSensors(),
    mVolumes()
    //
    ,
    mNPoints(0),
    mPoolNPoints(0),
    mPoolFreePointID(0),
    mPointsPool(),
    mAlgSteer(nullptr)
{
  // def c-tor
  SetUniqueID(Controller::kUndefined); // derived detectors must override this
  SetUniqueID(6);
  mAddError[0] = mAddError[1] = 0;
  //
  for (int i = 0; i < NTrackTypes; i++) {
    mDisabled[i] = false;
    mObligatory[i] = false;
    mTrackFlagSel[i] = 0;
    mNPointsSel[i] = 0;
  }
  //
}

//____________________________________________
AlignableDetector::~AlignableDetector()
{
  // d-tor
  mSensors.Clear();  // sensors are also attached as volumes, don't delete them here
  mVolumes.Delete(); // here all is deleted
  mPointsPool.Delete();
}

//FIXME(milettri): needs AliESDtrack
////____________________________________________
//int AlignableDetector::ProcessPoints(const AliESDtrack* esdTr, AlignmentTrack* algTrack, bool inv)
//{
//  // Extract the points corresponding to this detector, recalibrate/realign them to the
//  // level of the "starting point" for the alignment/calibration session.
//  // If inv==true, the track propagates in direction of decreasing tracking X
//  // (i.e. upper leg of cosmic track)
//  //
//  const AliESDfriendTrack* trF(esdTr->GetFriendTrack());
//  const AliTrackPointArray* trP(trF->GetTrackPointArray());
//  //
//  int np(trP->getNPoints());
//  int npSel(0);
//  AlignmentPoint* apnt(0);
//  for (int ip = 0; ip < np; ip++) {
//    int vid = trP->GetVolumeID()[ip];
//    if (!sensorOfDetector(vid)){
//      continue;}
//    apnt = getSensorByVolId(vid)->TrackPoint2AlgPoint(ip, trP, esdTr);
//    if (!apnt){
//      continue;}
//    algTrack->addPoint(apnt);
//    if (inv){
//      apnt->setInvDir();}
//    npSel++;
//    mNPoints++;
//  }
//  //
//  return npSel;
//}

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
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  //FIXME(milettri): needs OCDB
  //  // Update L2G matrices used for data reconstruction
  //  //
  //  AliCDBManager* man = AliCDBManager::Instance();
  //  AliCDBEntry* ent = man->Get(Form("%s/Align/Data", GetName()));
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
void AlignableDetector::applyAlignmentFromMPSol()
{
  // apply alignment from millepede solution array to reference alignment level
  LOG(INFO) << "Applying alignment from Millepede solution";
  for (int isn = getNSensors(); isn--;) {
    getSensor(isn)->applyAlignmentFromMPSol();
  }
}

//_________________________________________________________
void AlignableDetector::cacheReferenceOCDB()
{
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  //FIXME(milettri): needs OCDB
  //  // if necessary, detector may fetch here some reference OCDB data
  //  //
  //  // cache global deltas to avoid preicision problem
  //  AliCDBManager* man = AliCDBManager::Instance();
  //  AliCDBEntry* ent = man->Get(Form("%s/Align/Data", GetName()));
  //  TObjArray* arr = (TObjArray*)ent->GetObject();
  //  for (int i = arr->GetEntriesFast(); i--;) {
  //    const AliAlignObjParams* par = (const AliAlignObjParams*)arr->At(i);
  //    AlignableVolume* vol = getVolume(par->GetSymName());
  //    if (!vol) {
  //      AliErrorF("Volume %s not found", par->GetSymName());
  //      continue;
  //    }
  //    TGeoHMatrix delta;
  //    par->GetMatrix(delta);
  //    vol->setGlobalDeltaRef(delta);
  //  }
}

//_________________________________________________________
AlignmentPoint* AlignableDetector::getPointFromPool()
{
  // fetch or create new free point from the pool.
  // detector may override this method to create its own points derived from AlignmentPoint
  //
  if (mPoolFreePointID >= mPoolNPoints) { // expand pool
    mPointsPool.AddAtAndExpand(new AlignmentPoint(), mPoolNPoints++);
  }
  //
  AlignmentPoint* pnt = (AlignmentPoint*)mPointsPool.UncheckedAt(mPoolFreePointID++);
  pnt->Clear();
  return pnt;
  //
}

//_________________________________________________________
void AlignableDetector::resetPool()
{
  // declare pool free
  mPoolFreePointID = 0;
  mNPoints = 0;
}

//_________________________________________________________
void AlignableDetector::defineVolumes()
{
  // dummy method
  LOG(ERROR) << "This method must be implemented by specific detector";
}

//_________________________________________________________
void AlignableDetector::addVolume(AlignableVolume* vol)
{
  // add volume
  if (getVolume(vol->getSymName())) {
    LOG(FATAL) << "Volume " << vol->GetName() << " was already added to " << GetName();
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
    // modified global-local matrix
    vol->prepareMatrixL2G();
    // ideal global-local matrix
    vol->prepareMatrixL2GIdeal();
    //
  }
  // Now set tracking-local matrix (MUST be done after ALL L2G matrices are done!)
  // Attention: for sensor it is a real tracking matrix extracted from
  // the geometry but for container alignable volumes the tracking frame
  // is used for as the reference for the alignment parameters only,
  // see its definition in the AlignableVolume::PrepateMatrixT2L
  next.Reset();
  while ((vol = (AlignableVolume*)next())) {
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
    LOG(WARNING) << "No sensors defined";
    return;
  }
  mSensors.Sort();
  mSID2VolID = new int[mNSensors]; // cash id's for fast binary search
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
  mNDOFs += mNCalibDOF;
  setInitGeomDone();
  return mNDOFs;
}

//_________________________________________________________
int AlignableDetector::assignDOFs()
{
  // assign DOFs IDs, parameters
  //
  int gloCount0(mAlgSteer->getNDOFs()), gloCount(mAlgSteer->getNDOFs());
  float* pars = mAlgSteer->getGloParVal();
  float* errs = mAlgSteer->getGloParErr();
  int* labs = mAlgSteer->getGloParLab();
  //
  // assign calibration DOFs
  mFirstParGloID = gloCount;
  mParVals = pars + gloCount;
  mParErrs = errs + gloCount;
  mParLabs = labs + gloCount;
  for (int icl = 0; icl < mNCalibDOF; icl++) {
    mParLabs[icl] = (getDetLabel() + 10000) * 100 + icl;
    gloCount++;
  }
  //
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->getParent()) {
      vol->assignDOFs(gloCount, pars, errs, labs);
    }
  }
  if (mNDOFs != gloCount - gloCount0) {
    LOG(FATAL) << "Mismatch between declared " << mNDOFs << " and initialized " << (gloCount - gloCount0) << " DOFs for " << GetName();
  }
  return mNDOFs;
}

//_________________________________________________________
void AlignableDetector::initDOFs()
{
  // initialize free parameters
  if (getInitDOFsDone()) {
    LOG(FATAL) << "DOFs are already initialized for " << GetName();
  }
  //
  // process calibration DOFs
  for (int i = 0; i < mNCalibDOF; i++) {
    if (mParErrs[i] < -9999 && isZeroAbs(mParVals[i])) {
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
         GetName(), getNVolumes(), getNSensors(), getVolIDMin(),
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
  fmt::printf("Sel. flags in Collisions: {:05#x}%05 | Cosmic: 0x{:05#x}%05\n", mTrackFlagSel[Coll], mTrackFlagSel[Cosm]);
  //
  printf("Min.points in Collisions: %7d | Cosmic: %7d\n",
         mNPointsSel[Coll], mNPointsSel[Cosm]);
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
void AlignableDetector::setDetID(uint32_t tp)
{
  o2::detectors::DetID detID(tp);
  SetUniqueID(detID);
  LOG(WARNING) << __PRETTY_FUNCTION__ << "Possible discrepancies with o2 detector ID";
}

//____________________________________________
void AlignableDetector::setAddError(double sigy, double sigz)
{
  // add syst error to all sensors
  LOG(INFO) << "Adding sys.error " << std::fixed << std::setprecision(4) << sigy << " " << sigz << " to all sensors";
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
  LOG(FATAL) << "UpdatePointByTrackInfo is not implemented for this detector";
  //
}

//____________________________________________
void AlignableDetector::updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const
{
  // update point using specific error parameterization
  LOG(FATAL) << "If needed, this method has to be implemented for specific detector";
}

//____________________________________________
void AlignableDetector::setObligatory(int tp, bool v)
{
  // mark detector presence obligatory in the track
  mObligatory[tp] = v;
  mAlgSteer->setObligatoryDetector(getDetID(), tp, v);
}

//______________________________________________________
void AlignableDetector::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  fprintf(parOut, "\n!!\t\tDetector:\t%s\tNDOFs: %d\n", GetName(), getNDOFs());
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
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  //FIXME(lettrich): needs OCDB
  //  // store updated alignment
  //  TClonesArray* arr = new TClonesArray("AliAlignObjParams", 10);
  //  //
  //  int nvol = getNVolumes();
  //  for (int iv = 0; iv < nvol; iv++) {
  //    AlignableVolume* vol = getVolume(iv);
  //    // call only for top level objects, they will take care of children
  //    if (!vol->getParent()){
  //      vol->createAlignmentObjects(arr);}
  //  }
  //  //
  //  AliCDBManager* man = AliCDBManager::Instance();
  //  AliCDBMetaData* md = new AliCDBMetaData();
  //  md->SetResponsible(mAlgSteer->getOutCDBResponsible());
  //  md->SetComment(mAlgSteer->getOutCDBResponsible());
  //  //
  //  AliCDBId id(Form("%s/Align/Data", GetName()), mAlgSteer->getOutCDBRunMin(), mAlgSteer->getOutCDBRunMax());
  //  man->Put(arr, id, md);
  //  //
  //  delete arr;
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
  if (id >= mFirstParGloID && id < mFirstParGloID + mNCalibDOF) {
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
  DOFStatistics* st = mAlgSteer->GetDOFStat();
  for (int iv = 0; iv < nvol; iv++) {
    AlignableVolume* vol = getVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->getParent()) {
      mNProcPoints += vol->finalizeStat(st);
    }
  }
  fillDOFStat(st); // fill stat for calib dofs
}

//________________________________________
void AlignableDetector::addAutoConstraints() const
{
  // adds automatic constraints
  int nvol = getNVolumes();
  for (int iv = 0; iv < nvol; iv++) { // call for root level volumes, they will take care of their children
    AlignableVolume* vol = getVolume(iv);
    if (!vol->getParent()) {
      vol->addAutoConstraints((TObjArray*)mAlgSteer->getConstraints());
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
int AlignableDetector::selectVolumes(TObjArray* arr, int lev, const char* match)
{
  // select volumes matching to pattern and/or hierarchy level
  //
  if (!arr) {
    return 0;
  }
  int nadd = 0;
  TString mts = match, syms;
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!mts.IsNull() && !(syms = vol->getSymName()).Contains(mts)) {
      continue;
    } //wrong name
    arr->AddLast(vol);
    nadd++;
  }
  //
  return nadd;
}

//________________________________________
void AlignableDetector::setFreeDOFPattern(uint32_t pat, int lev, const char* match)
{
  // set free DOFs to volumes matching either to hierarchy level or
  // whose name contains match
  //
  TString mts = match, syms;
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!mts.IsNull() && !(syms = vol->getSymName()).Contains(mts)) {
      continue;
    } //wrong name
    vol->setFreeDOFPattern(pat);
  }
  //
}

//________________________________________
void AlignableDetector::setDOFCondition(int dof, float condErr, int lev, const char* match)
{
  // set condition for DOF of volumes matching either to hierarchy level or
  // whose name contains match
  //
  TString mts = match, syms;
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (lev >= 0 && vol->countParents() != lev) {
      continue;
    } // wrong level
    if (!mts.IsNull() && !(syms = vol->getSymName()).Contains(mts)) {
      continue;
    } //wrong name
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
  GeometricalConstraint* constr = new GeometricalConstraint();
  for (int i = 0; i < AlignableVolume::kNDOFGeom; i++) {
    if (sigma[i] >= 0) {
      constr->constrainDOF(i);
    } else {
      constr->unConstrainDOF(i);
    }
    constr->setSigma(i, sigma[i]);
  }
  for (int i = getNVolumes(); i--;) {
    AlignableVolume* vol = getVolume(i);
    if (vol->getParent()) {
      continue;
    } // wrong level
    if (!mts.IsNull() && !(syms = vol->getSymName()).Contains(mts)) {
      continue;
    } //wrong name
    constr->addChild(vol);
  }
  //
  if (!constr->getNChildren()) {
    LOG(INFO) << "No volume passed filter " << match;
    delete constr;
  } else {
    ((TObjArray*)mAlgSteer->getConstraints())->Add(constr);
  }
}

//________________________________________
void AlignableDetector::setFreeDOF(int dof)
{
  // set detector free dof
  if (dof >= kNMaxKalibDOF) {
    LOG(FATAL) << "Detector CalibDOFs limited to " << kNMaxKalibDOF << ", requested " << dof;
  }
  mCalibDOF |= 0x1 << dof;
  calcFree();
}

//________________________________________
void AlignableDetector::fixDOF(int dof)
{
  // fix detector dof
  if (dof >= kNMaxKalibDOF) {
    LOG(FATAL) << "Detector CalibDOFs limited to " << kNMaxKalibDOF << ", requested " << dof;
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
  mNCalibDOFFree = 0;
  for (int i = 0; i < mNCalibDOF; i++) {
    if (!isFreeDOF(i)) {
      if (condFix) {
        setParErr(i, -999);
      }
      continue;
    }
    mNCalibDOFFree++;
  }
  //
}

//______________________________________________________
void AlignableDetector::fillDOFStat(DOFStatistics* st) const
{
  // fill statistics info hist
  if (!st) {
    return;
  }
  int ndf = getNCalibDOFs();
  int dof0 = getFirstParGloID();
  int stat = getNProcessedPoints();
  for (int idf = 0; idf < ndf; idf++) {
    int dof = idf + dof0;
    st->addStat(dof, stat);
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
  TTree* tr = new TTree("snpos", Form("sensor poisitions for %s", GetName()));
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
