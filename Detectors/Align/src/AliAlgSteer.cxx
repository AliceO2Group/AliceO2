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

#include "AliAlgSteer.h"
#include "AliLog.h"
#include "AliAlgAux.h"
#include "AliAlgPoint.h"
#include "AliAlgDet.h"
#include "AliAlgVol.h"
#include "AliAlgDetITS.h"
#include "AliAlgDetTPC.h"
#include "AliAlgDetTRD.h"
#include "AliAlgDetTOF.h"
#include "AliAlgVtx.h"
#include "AliAlgMPRecord.h"
#include "AliAlgRes.h"
#include "AliAlgResFast.h"
#include "AliAlgConstraint.h"
#include "AliAlgDOFStat.h"
#include "AliTrackerBase.h"
#include "AliESDCosmicTrack.h"
#include "AliESDtrack.h"
#include "AliESDEvent.h"
#include "AliESDVertex.h"
#include "AliRecoParam.h"
#include "AliCDBRunRange.h"
#include "AliCDBManager.h"
#include "AliCDBEntry.h"
#include "Mille.h"
#include <TMath.h>
#include <TString.h>
#include <TTree.h>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TRandom.h>
#include <TH1F.h>
#include <TList.h>
#include <stdio.h>
#include <TGeoGlobalMagField.h>

using namespace TMath;
using namespace o2::align::AliAlgAux;
using std::ifstream;

ClassImp(o2::align::AliAlgSteer);

namespace o2
{
namespace align
{

const Char_t* AliAlgSteer::fgkMPDataExt = ".mille";
const Char_t* AliAlgSteer::fgkDetectorName[AliAlgSteer::kNDetectors] = {"ITS", "TPC", "TRD", "TOF", "HMPID"};
const Int_t AliAlgSteer::fgkSkipLayers[AliAlgSteer::kNLrSkip] = {AliGeomManager::kPHOS1, AliGeomManager::kPHOS2,
                                                                 AliGeomManager::kMUON, AliGeomManager::kEMCAL};

const Char_t* AliAlgSteer::fgkStatClName[AliAlgSteer::kNStatCl] = {"Inp: ", "Acc: "};
const Char_t* AliAlgSteer::fgkStatName[AliAlgSteer::kMaxStat] =
  {"runs", "Ev.Coll", "Ev.Cosm", "Trc.Coll", "Trc.Cosm"};

const Char_t* AliAlgSteer::fgkHStatName[AliAlgSteer::kNHVars] = {
  "Runs", "Ev.Inp", "Ev.VtxOK", "Tr.Inp", "Tr.2Fit", "Tr.2FitVC", "Tr.2PrMat", "Tr.2ResDer", "Tr.Stored", "Tr.Acc", "Tr.ContRes"};

//________________________________________________________________
AliAlgSteer::AliAlgSteer(const char* configMacro, int refRun)
  : fNDet(0), fNDOFs(0), fRunNumber(-1), fFieldOn(kFALSE), fTracksType(kColl), fAlgTrack(0), fVtxSens(0), fConstraints(), fSelEventSpecii(AliRecoParam::kCosmic | AliRecoParam::kLowMult | AliRecoParam::kHighMult | AliRecoParam::kDefault), fCosmicSelStrict(kFALSE), fVtxMinCont(-1), fVtxMaxCont(-1), fVtxMinContVC(10), fMinITSClforVC(3), fITSPattforVC(AliAlgDetITS::kSPDAny), fMaxChi2forVC(10)
    //
    ,
    fGloParVal(0),
    fGloParErr(0),
    fGloParLab(0),
    fOrderedLbl(0),
    fLbl2ID(0),
    fRefPoint(0),
    fESDTree(0),
    fESDEvent(0),
    fVertex(0),
    fControlFrac(1.0),
    fMPOutType(kMille | kMPRec | kContR),
    fMille(0),
    fMPRecord(0),
    fCResid(0),
    fMPRecTree(0),
    fResidTree(0),
    fMPRecFile(0),
    fResidFile(0),
    fMilleDBuffer(),
    fMilleIBuffer(),
    fMPDatFileName("mpData"),
    fMPParFileName("mpParams.txt"),
    fMPConFileName("mpConstraints.txt"),
    fMPSteerFileName("mpSteer.txt"),
    fResidFileName("mpControlRes.root"),
    fMilleOutBin(kTRUE),
    fDoKalmanResid(kTRUE)
    //
    ,
    fOutCDBPath("local://outOCDB"),
    fOutCDBComment("AliAlgSteer"),
    fOutCDBResponsible("")
    //
    ,
    fDOFStat(0),
    fHistoStat(0)
    //
    ,
    fConfMacroName(configMacro),
    fRecoOCDBConf("configRecoOCDB.C"),
    fRefOCDBConf("configRefOCDB.C"),
    fRefRunNumber(refRun),
    fRefOCDBLoaded(0),
    fUseRecoOCDB(kTRUE)
{
  // def c-tor
  for (int i = kNDetectors; i--;) {
    fDetectors[i] = 0;
    fDetPos[i] = -1;
  }
  SetPtMinColl();
  SetPtMinCosm();
  SetEtaMaxColl();
  SetEtaMaxCosm();
  SetMinDetAccColl();
  SetMinDetAccCosm();
  for (int i = 0; i < kNTrackTypes; i++) {
    fObligatoryDetPattern[i] = 0;
  }
  //
  SetMinPointsColl();
  SetMinPointsCosm();
  //
  for (int i = kNCosmLegs; i--;)
    fESDTrack[i] = 0;
  memset(fStat, 0, kNStatCl * kMaxStat * sizeof(float));
  SetMaxDCAforVC();
  SetMaxChi2forVC();
  SetOutCDBRunRange();
  SetDefPtBOffCosm();
  SetDefPtBOffColl();
  //
  // run config macro if provided
  if (!fConfMacroName.IsNull()) {
    gROOT->ProcessLine(Form(".x %s+g((AliAlgSteer*)%p)", fConfMacroName.Data(), this));
    if (!GetInitDOFsDone())
      InitDOFs();
    if (!GetNDOFs())
      AliFatalF("No DOFs found, initialization with %s failed",
                fConfMacroName.Data());
  }
}

//________________________________________________________________
AliAlgSteer::~AliAlgSteer()
{
  // d-tor
  if (fMPRecFile)
    CloseMPRecOutput();
  if (fMille)
    CloseMilleOutput();
  if (fResidFile)
    CloseResidOutput();
  //
  delete fAlgTrack;
  delete[] fGloParVal;
  delete[] fGloParErr;
  delete[] fGloParLab;
  for (int i = 0; i < fNDet; i++)
    delete fDetectors[i];
  delete fVtxSens;
  delete fRefPoint;
  delete fDOFStat;
  delete fHistoStat;
  //
}

//________________________________________________________________
void AliAlgSteer::InitDetectors()
{
  // init all detectors geometry
  //
  if (GetInitGeomDone())
    return;
  //

  //
  fAlgTrack = new AliAlgTrack();
  fRefPoint = new AliAlgPoint();
  //
  int dofCnt = 0;
  // special fake sensor for vertex constraint point
  // it has special T2L matrix adjusted for each track, no need to init it here
  fVtxSens = new AliAlgVtx();
  fVtxSens->SetInternalID(1);
  fVtxSens->PrepareMatrixL2G();
  fVtxSens->PrepareMatrixL2GIdeal();
  dofCnt += fVtxSens->GetNDOFs();
  //
  for (int i = 0; i < fNDet; i++)
    dofCnt += fDetectors[i]->InitGeom();
  if (!dofCnt)
    AliFatal("No DOFs found");
  //
  //
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det || det->IsDisabled())
      continue;
    det->CacheReferenceOCDB();
  }
  //
  fGloParVal = new Float_t[dofCnt];
  fGloParErr = new Float_t[dofCnt];
  fGloParLab = new Int_t[dofCnt];
  fOrderedLbl = new Int_t[dofCnt];
  fLbl2ID = new Int_t[dofCnt];
  memset(fGloParVal, 0, dofCnt * sizeof(Float_t));
  memset(fGloParErr, 0, dofCnt * sizeof(Float_t));
  memset(fGloParLab, 0, dofCnt * sizeof(Int_t));
  memset(fOrderedLbl, 0, dofCnt * sizeof(Int_t));
  memset(fLbl2ID, 0, dofCnt * sizeof(Int_t));
  AssignDOFs();
  AliInfoF("Booked %d global parameters", dofCnt);
  //
  SetInitGeomDone();
  //
}

//________________________________________________________________
void AliAlgSteer::InitDOFs()
{
  // scan all free global parameters, link detectors to array of params
  //
  if (GetInitDOFsDone()) {
    AliInfoF("InitDOFs was already done, just reassigning %d DOFs arrays/labels", fNDOFs);
    AssignDOFs();
    return;
  }
  //
  fNDOFs = 0;
  int ndfAct = 0;
  AssignDOFs();
  //
  int nact = 0;
  fVtxSens->InitDOFs();
  for (int i = 0; i < fNDet; i++) {
    AliAlgDet* det = GetDetector(i);
    det->InitDOFs();
    if (det->IsDisabled())
      continue;
    nact++;
    ndfAct += det->GetNDOFs();
  }
  for (int i = 0; i < kNTrackTypes; i++)
    if (nact < fMinDetAcc[i])
      AliFatalF("%d detectors are active, while %d in track are asked", nact, fMinDetAcc[i]);
  //
  AliInfoF("%d global parameters %d detectors, %d in %d active detectors", fNDOFs, fNDet, ndfAct, nact);
  //
  AddAutoConstraints();
  //
  SetInitDOFsDone();
}

//________________________________________________________________
void AliAlgSteer::AssignDOFs()
{
  // add parameters/labels arrays to volumes. If the AliAlgSteer is read from the file, this method need
  // to be called (of InitDOFs should be called)
  //
  int ndfOld = -1;
  if (fNDOFs > 0)
    ndfOld = fNDOFs;
  fNDOFs = 0;
  //
  fVtxSens->AssignDOFs(fNDOFs, fGloParVal, fGloParErr, fGloParLab);
  //
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det)
      continue;
    //if (det->IsDisabled()) continue;
    fNDOFs += det->AssignDOFs();
  }
  AliInfoF("Assigned parameters/labels arrays for %d DOFs", fNDOFs);
  if (ndfOld > -1 && ndfOld != fNDOFs)
    AliErrorF("Recalculated NDOFs=%d not equal to saved NDOFs=%d", fNDOFs, ndfOld);
  //
  // build Lbl <-> parID table
  Sort(fNDOFs, fGloParLab, fLbl2ID, kFALSE); // sort in increasing order
  for (int i = fNDOFs; i--;)
    fOrderedLbl[i] = fGloParLab[fLbl2ID[i]];
  //
}

//________________________________________________________________
void AliAlgSteer::AddDetector(UInt_t id, AliAlgDet* det)
{
  // add detector participating in the alignment, optionally constructed externally
  //
  if (!fRefOCDBLoaded)
    LoadRefOCDB();
  //
  if (id >= kNDetectors)
    AliFatalF("Detector typeID %d exceeds allowed range %d:%d",
              id, 0, kNDetectors - 1);
  //
  if (fDetPos[id] != -1)
    AliFatalF("Detector %d was already added", id);
  if (!det) {
    switch (id) {
      case kITS:
        det = new AliAlgDetITS(GetDetNameByDetID(kITS));
        break;
      case kTPC:
        det = new AliAlgDetTPC(GetDetNameByDetID(kTPC));
        break;
      case kTRD:
        det = new AliAlgDetTRD(GetDetNameByDetID(kTRD));
        break;
      case kTOF:
        det = new AliAlgDetTOF(GetDetNameByDetID(kTOF));
        break;
      default:
        AliFatalF("%d not implemented yet", id);
        break;
    };
  }
  //
  fDetectors[fNDet] = det;
  fDetPos[id] = fNDet;
  det->SetAlgSteer(this);
  for (int i = 0; i < kNTrackTypes; i++)
    SetObligatoryDetector(id, i, det->IsObligatory(i));
  fNDet++;
  //
}

//_________________________________________________________
void AliAlgSteer::AddDetector(AliAlgDet* det)
{
  // add detector constructed externally to alignment framework
  AddDetector(det->GetDetID(), det);
}

//_________________________________________________________
Bool_t AliAlgSteer::CheckDetectorPattern(UInt_t patt) const
{
  //validate detector pattern
  return ((patt & fObligatoryDetPattern[fTracksType]) ==
          fObligatoryDetPattern[fTracksType]) &&
         NumberOfBitsSet(patt) >= fMinDetAcc[fTracksType];
}

//_________________________________________________________
Bool_t AliAlgSteer::CheckDetectorPoints(const int* npsel) const
{
  //validate detectors pattern according to number of selected points
  int ndOK = 0;
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det || det->IsDisabled(fTracksType))
      continue;
    if (npsel[idt] < det->GetNPointsSel(fTracksType)) {
      if (det->IsObligatory(fTracksType))
        return kFALSE;
      continue;
    }
    ndOK++;
  }
  return ndOK >= fMinDetAcc[fTracksType];
}

//_________________________________________________________
UInt_t AliAlgSteer::AcceptTrack(const AliESDtrack* esdTr, Bool_t strict) const
{
  // decide if the track should be processed
  AliAlgDet* det = 0;
  UInt_t detAcc = 0;
  if (fFieldOn && esdTr->Pt() < fPtMin[fTracksType])
    return 0;
  if (Abs(esdTr->Eta()) > fEtaMax[fTracksType])
    return 0;
  //
  for (int idet = 0; idet < kNDetectors; idet++) {
    if (!(det = GetDetectorByDetID(idet)) || det->IsDisabled(fTracksType))
      continue;
    if (!det->AcceptTrack(esdTr, fTracksType)) {
      if (strict && det->IsObligatory(fTracksType))
        return 0;
      else
        continue;
    }
    //
    detAcc |= 0x1 << idet;
  }
  if (NumberOfBitsSet(detAcc) < fMinDetAcc[fTracksType])
    return 0;
  return detAcc;
  //
}

//_________________________________________________________
UInt_t AliAlgSteer::AcceptTrackCosmic(const AliESDtrack* esdPairCosm[kNCosmLegs]) const
{
  // decide if the pair of tracks making cosmic track should be processed
  UInt_t detAcc = 0, detAccLeg;
  for (int i = kNCosmLegs; i--;) {
    detAccLeg = AcceptTrack(esdPairCosm[i], fCosmicSelStrict); // missing obligatory detectors in one leg might be allowed
    if (!detAccLeg)
      return 0;
    detAcc |= detAccLeg;
  }
  if (fCosmicSelStrict)
    return detAcc;
  //
  // for non-stric selection check convolution of detector presence
  if (!CheckDetectorPattern(detAcc))
    return 0;
  return detAcc;
  //
}

//_________________________________________________________
void AliAlgSteer::SetESDEvent(const AliESDEvent* ev)
{
  // attach event to analyse
  fESDEvent = ev;
  // setup magnetic field if needed
  if (fESDEvent &&
      (!TGeoGlobalMagField::Instance()->GetField() ||
       !SmallerAbs(fESDEvent->GetMagneticField() - AliTrackerBase::GetBz(), 5e-4))) {
    fESDEvent->InitMagneticField();
  }
}

//_________________________________________________________
Bool_t AliAlgSteer::ProcessEvent(const AliESDEvent* esdEv)
{
  // process event
  const int kProcStatFreq = 100;
  static int evCount = 0;
  if (!(evCount % kProcStatFreq)) {
    ProcInfo_t procInfo;
    gSystem->GetProcInfo(&procInfo);
    AliInfoF("ProcStat: CPUusr: %6d CPUsys: %6d RMem:%6d VMem:%6d",
             int(procInfo.fCpuUser), int(procInfo.fCpuSys),
             int(procInfo.fMemResident / 1024), int(procInfo.fMemVirtual / 1024));
  }
  evCount++;
  //
  SetESDEvent(esdEv);
  //
  if (esdEv->GetRunNumber() != GetRunNumber())
    SetRunNumber(esdEv->GetRunNumber());
  //
  if (!(esdEv->GetEventSpecie() & fSelEventSpecii)) {
#if DEBUG > 2
    AliInfoF("Reject: specie does not match, allowed 0x%0x", fSelEventSpecii);
#endif
    return kFALSE;
  }
  //
  SetCosmic(esdEv->GetEventSpecie() == AliRecoParam::kCosmic ||
            (esdEv->GetNumberOfCosmicTracks() > 0 && !esdEv->GetPrimaryVertexTracks()->GetStatus()));
  //
  FillStatHisto(kEvInp);
  //
#if DEBUG > 2
  AliInfoF("Processing event %d of ev.specie %d -> Ntr: %4d NtrCosm: %d",
           esdEv->GetEventNumberInFile(), esdEv->GetEventSpecie(),
           esdEv->GetNumberOfTracks(), esdEv->GetNumberOfCosmicTracks());
#endif
  //
  SetFieldOn(Abs(esdEv->GetMagneticField()) > kAlmost0Field);
  if (!IsCosmic() && !CheckSetVertex(esdEv->GetPrimaryVertexTracks()))
    return kFALSE;
  FillStatHisto(kEvVtx);
  //
  int ntr = 0, accTr = 0;
  if (IsCosmic()) {
    fStat[kInpStat][kEventCosm]++;
    ntr = esdEv->GetNumberOfCosmicTracks();
    FillStatHisto(kTrackInp, ntr);
    for (int itr = 0; itr < ntr; itr++) {
      accTr += ProcessTrack(esdEv->GetCosmicTrack(itr));
    }
    if (accTr)
      fStat[kAccStat][kEventCosm]++;
  } else {
    fStat[kInpStat][kEventColl]++;
    ntr = esdEv->GetNumberOfTracks();
    FillStatHisto(kTrackInp, ntr);
    for (int itr = 0; itr < ntr; itr++) {
      //      int accTrOld = accTr;
      accTr += ProcessTrack(esdEv->GetTrack(itr));
      /*
      if (accTr>accTrOld && fCResid) {
	int ndf = fCResid->GetNPoints()*2-5;
	if (fCResid->GetChi2()/ndf>20 || !fCResid->GetKalmanDone()
	    || fCResid->GetChi2K()/ndf>20) {
	  printf("BAD FIT for %d\n",itr);
	}
       	fCResid->Print("er");
      }
      */
    }
    if (accTr)
      fStat[kAccStat][kEventColl]++;
  }
  //
  FillStatHisto(kTrackAcc, accTr);
  //
  if (accTr) {
    AliInfoF("Processed event %d of ev.specie %d -> Accepted: %4d of %4d tracks",
             esdEv->GetEventNumberInFile(), esdEv->GetEventSpecie(), accTr, ntr);
  }
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::ProcessTrack(const AliESDtrack* esdTr)
{
  // process single track
  //
  fStat[kInpStat][kTrackColl]++;
  fESDTrack[0] = esdTr;
  fESDTrack[1] = 0;
  //
  int nPnt = 0;
  const AliESDfriendTrack* trF = esdTr->GetFriendTrack();
  if (!trF)
    return kFALSE;
  const AliTrackPointArray* trPoints = trF->GetTrackPointArray();
  if (!trPoints || (nPnt = trPoints->GetNPoints()) < 1)
    return kFALSE;
  //
  UInt_t detAcc = AcceptTrack(esdTr);
  if (!detAcc)
    return kFALSE;
  //
  ResetDetectors();
  fAlgTrack->Clear();
  //
  // process the track points for each detector,
  AliAlgDet* det = 0;
  for (int idet = 0; idet < kNDetectors; idet++) {
    if (!(detAcc & (0x1 << idet)))
      continue;
    det = GetDetectorByDetID(idet);
    if (det->ProcessPoints(esdTr, fAlgTrack) < det->GetNPointsSel(kColl)) {
      detAcc &= ~(0x1 << idet); // did not survive, suppress detector in the track
      if (det->IsObligatory(kColl))
        return kFALSE;
    }
    if (NumberOfBitsSet(detAcc) < fMinDetAcc[kColl])
      return kFALSE; // abandon track
  }
  //
  if (fAlgTrack->GetNPoints() < GetMinPoints())
    return kFALSE;
  // fill needed points (tracking frame) in the fAlgTrack
  fRefPoint->SetContainsMeasurement(kFALSE);
  fRefPoint->SetContainsMaterial(kFALSE);
  fAlgTrack->AddPoint(fRefPoint); // reference point which the track will refer to
  //
  fAlgTrack->CopyFrom(esdTr);
  if (!GetFieldOn())
    fAlgTrack->ImposePtBOff(fDefPtBOff[AliAlgAux::kColl]);
  fAlgTrack->SetFieldON(GetFieldOn());
  fAlgTrack->SortPoints();
  //
  // at this stage the points are sorted from maxX to minX, the latter corresponding to
  // reference point (e.g. vertex) with X~0. The fAlgTrack->GetInnerPointID() points on it,
  // hence fAlgTrack->GetInnerPointID() is the 1st really measured point. We will set the
  // alpha of the reference point to alpha of the barrel sector corresponding to this
  // 1st measured point
  int pntMeas = fAlgTrack->GetInnerPointID() - 1;
  if (pntMeas < 0) { // this should not happen
    fAlgTrack->Print("p meas");
    AliFatal("AliAlgTrack->GetInnerPointID() cannot be 0");
  }
  // do we want to add the vertex as a measured point ?
  if (!AddVertexConstraint()) { // no constrain, just reference point w/o measurement
    fRefPoint->SetXYZTracking(0, 0, 0);
    fRefPoint->SetAlphaSens(Sector2Alpha(fAlgTrack->GetPoint(pntMeas)->GetAliceSector()));
  } else
    FillStatHisto(kTrackFitInpVC);
  //
  FillStatHisto(kTrackFitInp);
  if (!fAlgTrack->IniFit())
    return kFALSE;
  FillStatHisto(kTrackProcMatInp);
  if (!fAlgTrack->ProcessMaterials())
    return kFALSE;
  fAlgTrack->DefineDOFs();
  //
  FillStatHisto(kTrackResDerInp);
  if (!fAlgTrack->CalcResidDeriv())
    return kFALSE;
  //
  if (!StoreProcessedTrack(fMPOutType & ~kContR))
    return kFALSE; // store derivatives for MP
  //
  if (GetProduceControlRes() &&                                   // need control residuals, ignore selection fraction if this is the
      (fMPOutType == kContR || gRandom->Rndm() < fControlFrac)) { // output requested
    if (!TestLocalSolution() || !StoreProcessedTrack(kContR))
      return kFALSE;
  }
  //
  FillStatHisto(kTrackStore);
  //
  fStat[kAccStat][kTrackColl]++;
  //
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::CheckSetVertex(const AliESDVertex* vtx)
{
  // vertex selection/constraint check
  if (!vtx) {
    fVertex = 0;
    return kTRUE;
  }
  int ncont = vtx->GetNContributors();
  if (fVtxMinCont > 0 && fVtxMinCont > ncont) {
#if DEBUG > 2
    AliInfoF("Rejecting event with %d vertex contributors (min %d asked)", ncont, fVtxMinCont);
#endif
    return kFALSE;
  }
  if (fVtxMaxCont > 0 && ncont > fVtxMaxCont) {
#if DEBUG > 2
    AliInfoF("Rejecting event with %d vertex contributors (max %d asked)", ncont, fVtxMaxCont);
#endif
    return kFALSE;
  }
  fVertex = (ncont >= fVtxMinContVC) ? vtx : 0; // use vertex as a constraint
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::ProcessTrack(const AliESDCosmicTrack* cosmTr)
{
  // process single cosmic track
  //
  fStat[kInpStat][kTrackCosm]++;
  int nPnt = 0;
  fESDTrack[0] = 0;
  fESDTrack[1] = 0;
  //
  for (int leg = kNCosmLegs; leg--;) {
    const AliESDtrack* esdTr =
      fESDEvent->GetTrack(leg == kCosmLow ? cosmTr->GetESDLowerTrackIndex() : cosmTr->GetESDUpperTrackIndex());
    const AliESDfriendTrack* trF = esdTr->GetFriendTrack();
    if (!trF)
      return kFALSE;
    const AliTrackPointArray* trPoints = trF->GetTrackPointArray();
    if (!trPoints || (nPnt += trPoints->GetNPoints()) < 1)
      return kFALSE;
    //
    fESDTrack[leg] = esdTr;
  }
  //
  UInt_t detAcc = AcceptTrackCosmic(fESDTrack);
  if (!detAcc)
    return kFALSE;
  //
  ResetDetectors();
  fAlgTrack->Clear();
  fAlgTrack->SetCosmic(kTRUE);
  //
  // process the track points for each detector,
  // fill needed points (tracking frame) in the fAlgTrack
  fRefPoint->SetContainsMeasurement(kFALSE);
  fRefPoint->SetContainsMaterial(kFALSE);
  fAlgTrack->AddPoint(fRefPoint); // reference point which the track will refer to
  //
  AliAlgDet* det = 0;
  Int_t npsel[kNDetectors] = {0};
  for (int nPleg = 0, leg = kNCosmLegs; leg--;) {
    for (int idet = 0; idet < kNDetectors; idet++) {
      if (!(detAcc & (0x1 << idet)))
        continue;
      det = GetDetectorByDetID(idet);
      //
      // upper leg points marked as the track going in inverse direction
      int np = det->ProcessPoints(fESDTrack[leg], fAlgTrack, leg == kCosmUp);
      if (np < det->GetNPointsSel(kCosm) && fCosmicSelStrict &&
          det->IsObligatory(kCosm))
        return kFALSE;
      npsel[idet] += np;
      nPleg += np;
    }
    if (nPleg < GetMinPoints())
      return kFALSE;
  }
  // last check on legs-combined patter
  if (!CheckDetectorPoints(npsel))
    return kFALSE;
  //
  fAlgTrack->CopyFrom(cosmTr);
  if (!GetFieldOn())
    fAlgTrack->ImposePtBOff(fDefPtBOff[AliAlgAux::kCosm]);
  fAlgTrack->SetFieldON(GetFieldOn());
  fAlgTrack->SortPoints();
  //
  // at this stage the points are sorted from maxX to minX, the latter corresponding to
  // reference point (e.g. vertex) with X~0. The fAlgTrack->GetInnerPointID() points on it,
  // hence fAlgTrack->GetInnerPointID() is the 1st really measured point. We will set the
  // alpha of the reference point to alpha of the barrel sector corresponding to this
  // 1st measured point
  int pntMeas = fAlgTrack->GetInnerPointID() - 1;
  if (pntMeas < 0) { // this should not happen
    fAlgTrack->Print("p meas");
    AliFatal("AliAlgTrack->GetInnerPointID() cannot be 0");
  }
  fRefPoint->SetAlphaSens(Sector2Alpha(fAlgTrack->GetPoint(pntMeas)->GetAliceSector()));
  //
  FillStatHisto(kTrackFitInp);
  if (!fAlgTrack->IniFit())
    return kFALSE;
  //
  FillStatHisto(kTrackProcMatInp);
  if (!fAlgTrack->ProcessMaterials())
    return kFALSE;
  fAlgTrack->DefineDOFs();
  //
  FillStatHisto(kTrackResDerInp);
  if (!fAlgTrack->CalcResidDeriv())
    return kFALSE;
  //
  if (!StoreProcessedTrack(fMPOutType & ~kContR))
    return kFALSE; // store derivatives for MP
  //
  if (GetProduceControlRes() &&                                   // need control residuals, ignore selection fraction if this is the
      (fMPOutType == kContR || gRandom->Rndm() < fControlFrac)) { // output requested
    if (!TestLocalSolution() || !StoreProcessedTrack(kContR))
      return kFALSE;
  }
  //
  FillStatHisto(kTrackStore);
  fStat[kAccStat][kTrackCosm]++;
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::StoreProcessedTrack(Int_t what)
{
  // write alignment track
  Bool_t res = kTRUE;
  if ((what & kMille))
    res &= FillMilleData();
  if ((what & kMPRec))
    res &= FillMPRecData();
  if ((what & kContR))
    res &= FillControlData();
  //
  return res;
}

//_________________________________________________________
Bool_t AliAlgSteer::FillMilleData()
{
  // store MP2 data in Mille format
  if (!fMille) {
    TString mo = Form("%s%s", fMPDatFileName.Data(), fgkMPDataExt);
    fMille = new Mille(mo.Data(), fMilleOutBin);
    if (!fMille)
      AliFatalF("Failed to create output file %s", mo.Data());
  }
  //
  if (!fAlgTrack->GetDerivDone()) {
    AliError("Track derivatives are not yet evaluated");
    return kFALSE;
  }
  int np(fAlgTrack->GetNPoints()), nDGloTot(0); // total number global derivatives stored
  int nParETP(fAlgTrack->GetNLocExtPar());      // numnber of local parameters for reference track param
  int nVarLoc(fAlgTrack->GetNLocPar());         // number of local degrees of freedom in the track
  float *buffDL(0), *buffDG(0);                 // faster acces arrays
  int* buffI(0);
  //
  const int* gloParID(fAlgTrack->GetGloParID()); // IDs of global DOFs this track depends on
  for (int ip = 0; ip < np; ip++) {
    AliAlgPoint* pnt = fAlgTrack->GetPoint(ip);
    if (pnt->ContainsMeasurement()) {
      int gloOffs = pnt->GetDGloOffs(); // 1st entry of global derivatives for this point
      int nDGlo = pnt->GetNGloDOFs();   // number of global derivatives (number of DOFs it depends on)
      if (!pnt->IsStatOK())
        pnt->IncrementStat();
      // check buffer sizes
      {
        if (fMilleDBuffer.GetSize() < nVarLoc + nDGlo)
          fMilleDBuffer.Set(100 + nVarLoc + nDGlo);
        if (fMilleIBuffer.GetSize() < nDGlo)
          fMilleIBuffer.Set(100 + nDGlo);
        buffDL = fMilleDBuffer.GetArray(); // faster acces
        buffDG = buffDL + nVarLoc;         // faster acces
        buffI = fMilleIBuffer.GetArray();  // faster acces
      }
      // local der. array cannot be 0-suppressed by Mille construction, need to reset all to 0
      //
      for (int idim = 0; idim < 2; idim++) { // 2 dimensional orthogonal measurement
        memset(buffDL, 0, nVarLoc * sizeof(float));
        const double* deriv = fAlgTrack->GetDResDLoc(idim, ip); // array of Dresidual/Dparams_loc
        // derivatives over reference track parameters
        for (int j = 0; j < nParETP; j++)
          buffDL[j] = (IsZeroAbs(deriv[j])) ? 0 : deriv[j];
        //
        // point may depend on material variables within these limits
        int lp0 = pnt->GetMinLocVarID(), lp1 = pnt->GetMaxLocVarID();
        for (int j = lp0; j < lp1; j++)
          buffDL[j] = (IsZeroAbs(deriv[j])) ? 0 : deriv[j];
        //
        // derivatives over global params: this array can be 0-suppressed, no need to reset
        int nGlo(0);
        deriv = fAlgTrack->GetDResDGlo(idim, gloOffs);
        const int* gloIDP(gloParID + gloOffs);
        for (int j = 0; j < nDGlo; j++) {
          if (!IsZeroAbs(deriv[j])) {
            buffDG[nGlo] = deriv[j];                 // value of derivative
            buffI[nGlo++] = GetGloParLab(gloIDP[j]); // global DOF ID + 1 (Millepede needs positive labels)
          }
        }
        fMille->mille(nVarLoc, buffDL, nGlo, buffDG, buffI,
                      fAlgTrack->GetResidual(idim, ip), Sqrt(pnt->GetErrDiag(idim)));
        nDGloTot += nGlo;
        //
      }
    }
    if (pnt->ContainsMaterial()) { // material point can add 4 or 5 otrhogonal pseudo-measurements
      memset(buffDL, 0, nVarLoc * sizeof(float));
      int nmatpar = pnt->GetNMatPar(); // residuals (correction expectation value)
      //      const float* expMatCorr = pnt->GetMatCorrExp(); // expected corrections (diagonalized)
      const float* expMatCov = pnt->GetMatCorrCov(); // their diagonalized error matrix
      int offs = pnt->GetMaxLocVarID() - nmatpar;    // start of material variables
      // here all derivatives are 1 = dx/dx
      for (int j = 0; j < nmatpar; j++) { // mat. "measurements" don't depend on global params
        int j1 = j + offs;
        buffDL[j1] = 1.0; // only 1 non-0 derivative
        //fMille->mille(nVarLoc,buffDL,0,buffDG,buffI,expMatCorr[j],Sqrt(expMatCov[j]));
        // expectation for MS effect is 0
        fMille->mille(nVarLoc, buffDL, 0, buffDG, buffI, 0, Sqrt(expMatCov[j]));
        buffDL[j1] = 0.0; // reset buffer
      }
    } // material "measurement"
  }   // loop over points
  //
  if (!nDGloTot) {
    AliInfo("Track does not depend on free global parameters, discard");
    fMille->kill();
    return kFALSE;
  }
  fMille->end(); // store the record
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::FillMPRecData()
{
  // store MP2 in MPRecord format
  if (!fMPRecord)
    InitMPRecOutput();
  //
  fMPRecord->Clear();
  if (!fMPRecord->FillTrack(fAlgTrack, fGloParLab))
    return kFALSE;
  fMPRecord->SetRun(fRunNumber);
  fMPRecord->SetTimeStamp(fESDEvent->GetTimeStamp());
  UInt_t tID = 0xffff & UInt_t(fESDTrack[0]->GetID());
  if (IsCosmic())
    tID |= (0xffff & UInt_t(fESDTrack[1]->GetID())) << 16;
  fMPRecord->SetTrackID(tID);
  fMPRecTree->Fill();
  return kTRUE;
}

//_________________________________________________________
Bool_t AliAlgSteer::FillControlData()
{
  // store control residuals
  if (!fCResid)
    InitResidOutput();
  //
  int nps, np = fAlgTrack->GetNPoints();
  nps = (!fRefPoint->ContainsMeasurement()) ? np - 1 : np; // ref point is dummy?
  if (nps < 0)
    return kTRUE;
  //
  fCResid->Clear();
  if (!fCResid->FillTrack(fAlgTrack, fDoKalmanResid))
    return kFALSE;
  fCResid->SetRun(fRunNumber);
  fCResid->SetTimeStamp(fESDEvent->GetTimeStamp());
  fCResid->SetBz(fESDEvent->GetMagneticField());
  UInt_t tID = 0xffff & UInt_t(fESDTrack[0]->GetID());
  if (IsCosmic())
    tID |= (0xffff & UInt_t(fESDTrack[1]->GetID())) << 16;
  fCResid->SetTrackID(tID);
  //
  fResidTree->Fill();
  FillStatHisto(kTrackControl);
  //
  return kTRUE;
}

//_________________________________________________________
void AliAlgSteer::SetRunNumber(Int_t run)
{
  if (run == fRunNumber)
    return; // nothing to do
  //
  AcknowledgeNewRun(run);
}

//_________________________________________________________
void AliAlgSteer::AcknowledgeNewRun(Int_t run)
{
  // load needed info for new run
  if (run == fRunNumber)
    return; // nothing to do
  if (run > 0) {
    fStat[kAccStat][kRun]++;
  }
  if (fRunNumber > 0)
    FillStatHisto(kRunDone);
  fRunNumber = run;
  AliInfoF("Processing new run %d", fRunNumber);
  //
  // setup magnetic field
  if (fESDEvent &&
      (!TGeoGlobalMagField::Instance()->GetField() ||
       !SmallerAbs(fESDEvent->GetMagneticField() - AliTrackerBase::GetBz(), 5e-4))) {
    fESDEvent->InitMagneticField();
  }
  //
  if (!fUseRecoOCDB) {
    AliWarning("Reco-time OCDB will NOT be preloaded");
    return;
  }
  LoadRecoTimeOCDB();
  //
  for (int idet = 0; idet < fNDet; idet++) {
    AliAlgDet* det = GetDetector(idet);
    if (!det->IsDisabled())
      det->AcknowledgeNewRun(run);
  }
  //
  // bring to virgin state
  // CleanOCDB();
  //
  // LoadRefOCDB(); //??? we need to get back reference OCDB ???
  //
  fStat[kInpStat][kRun]++;
  //
}

//_________________________________________________________
Bool_t AliAlgSteer::LoadRecoTimeOCDB()
{
  // Load OCDB paths used for the reconstruction of data being processed
  // In order to avoid unnecessary uploads, the objects are not actually
  // loaded/cached but just added as specific paths with version
  AliInfoF("Preloading Reco-Time OCDB for run %d from ESD UserInfo list", fRunNumber);
  //
  CleanOCDB();
  //
  if (!fRecoOCDBConf.IsNull() && !gSystem->AccessPathName(fRecoOCDBConf.Data(), kFileExists)) {
    AliInfoF("Executing reco-time OCDB setup macro %s", fRecoOCDBConf.Data());
    gROOT->ProcessLine(Form(".x %s(%d)", fRecoOCDBConf.Data(), fRunNumber));
    if (AliCDBManager::Instance()->IsDefaultStorageSet())
      return kTRUE;
    AliFatalF("macro %s failed to configure reco-time OCDB", fRecoOCDBConf.Data());
  } else
    AliWarningF("No reco-time OCDB config macro %s is found, will use ESD:UserInfo",
                fRecoOCDBConf.Data());
  //
  if (!fESDTree)
    AliFatal("Cannot preload Reco-Time OCDB since the ESD tree is not set");
  const TTree* tr = fESDTree; // go the the real ESD tree
  while (tr->GetTree() && tr->GetTree() != tr)
    tr = tr->GetTree();
  //
  const TList* userInfo = const_cast<TTree*>(tr)->GetUserInfo();
  TMap* cdbMap = (TMap*)userInfo->FindObject("cdbMap");
  TList* cdbList = (TList*)userInfo->FindObject("cdbList");
  //
  if (!cdbMap || !cdbList) {
    userInfo->Print();
    AliFatal("Failed to extract cdbMap and cdbList from UserInfo list");
  }
  //
  return PreloadOCDB(fRunNumber, cdbMap, cdbList);
}

//_________________________________________________________
AliAlgDet* AliAlgSteer::GetDetectorByVolID(Int_t vid) const
{
  // get detector by sensor volid
  for (int i = fNDet; i--;)
    if (fDetectors[i]->SensorOfDetector(vid))
      return fDetectors[i];
  return 0;
}

//____________________________________________
void AliAlgSteer::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("%5d DOFs in %d detectors", fNDOFs, fNDet);
  if (!fConfMacroName.IsNull())
    printf("(config: %s)", fConfMacroName.Data());
  printf("\n");
  if (GetMPAlignDone())
    printf("ALIGNMENT FROM MILLEPEDE SOLUTION IS APPLIED\n");
  //
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det)
      continue;
    det->Print(opt);
  }
  if (!opts.IsNull()) {
    printf("\nSpecial sensor for Vertex Constraint\n");
    fVtxSens->Print(opt);
  }
  //
  // event selection
  printf("\n");
  printf("%-40s:\t", "Alowed event specii mask");
  PrintBits((ULong64_t)fSelEventSpecii, 5);
  printf("\n");
  printf("%-40s:\t%d/%d\n", "Min points per collisions track (BOff/ON)",
         fMinPoints[kColl][0], fMinPoints[kColl][1]);
  printf("%-40s:\t%d/%d\n", "Min points per cosmic track leg (BOff/ON)",
         fMinPoints[kCosm][0], fMinPoints[kCosm][1]);
  printf("%-40s:\t%d\n", "Min detectots per collision track", fMinDetAcc[kColl]);
  printf("%-40s:\t%d (%s)\n", "Min detectots per cosmic track/leg", fMinDetAcc[kCosm],
         fCosmicSelStrict ? "STRICT" : "SOFT");
  printf("%-40s:\t%d/%d\n", "Min/Max vertex contrib. to accept event", fVtxMinCont, fVtxMaxCont);
  printf("%-40s:\t%d\n", "Min vertex contrib. for constraint", fVtxMinContVC);
  printf("%-40s:\t%d\n", "Min Ncl ITS for vertex constraint", fMinITSClforVC);
  printf("%-40s:\t%s\n", "SPD request for vertex constraint",
         AliAlgDetITS::GetITSPattName(fITSPattforVC));
  printf("%-40s:\t%.4f/%.4f/%.2f\n", "DCAr/DCAz/Chi2 cut for vertex constraint",
         fMaxDCAforVC[0], fMaxDCAforVC[1], fMaxChi2forVC);
  printf("Collision tracks: Min pT: %5.2f |etaMax|: %5.2f\n", fPtMin[kColl], fEtaMax[kColl]);
  printf("Cosmic    tracks: Min pT: %5.2f |etaMax|: %5.2f\n", fPtMin[kCosm], fEtaMax[kCosm]);
  //
  printf("%-40s:\t%s", "Config. for reference OCDB", fRefOCDBConf.Data());
  if (fRefRunNumber >= 0)
    printf("(%d)", fRefRunNumber);
  printf("\n");
  printf("%-40s:\t%s\n", "Config. for reco-time OCDB", fRecoOCDBConf.Data());
  //
  printf("%-40s:\t%s\n", "Output OCDB path", fOutCDBPath.Data());
  printf("%-40s:\t%s/%s\n", "Output OCDB comment/responsible",
         fOutCDBComment.Data(), fOutCDBResponsible.Data());
  printf("%-40s:\t%6d:%6d\n", "Output OCDB run range", fOutCDBRunRange[0], fOutCDBRunRange[1]);
  //
  printf("%-40s:\t%s\n", "Filename for MillePede steering", fMPSteerFileName.Data());
  printf("%-40s:\t%s\n", "Filename for MillePede parameters", fMPParFileName.Data());
  printf("%-40s:\t%s\n", "Filename for MillePede constraints", fMPConFileName.Data());
  printf("%-40s:\t%s\n", "Filename for control residuals:", fResidFileName.Data());
  printf("%-40s:\t%.3f\n", "Fraction of control tracks", fControlFrac);
  printf("MPData output :\t");
  if (GetProduceMPData())
    printf("%s%s ", fMPDatFileName.Data(), fgkMPDataExt);
  if (GetProduceMPRecord())
    printf("%s%s ", fMPDatFileName.Data(), ".root");
  printf("\n");
  //
  if (opts.Contains("stat"))
    PrintStatistics();
}

//________________________________________________________
void AliAlgSteer::PrintStatistics() const
{
  // print processing stat
  printf("\nProcessing Statistics\n");
  printf("Type: ");
  for (int i = 0; i < kMaxStat; i++)
    printf("%s ", fgkStatName[i]);
  printf("\n");
  for (int icl = 0; icl < kNStatCl; icl++) {
    printf("%s ", fgkStatClName[icl]);
    for (int i = 0; i < kMaxStat; i++)
      printf(Form("%%%dd ", (int)strlen(fgkStatName[i])), (int)fStat[icl][i]);
    printf("\n");
  }
}

//________________________________________________________
void AliAlgSteer::ResetDetectors()
{
  // reset detectors for next track
  fRefPoint->Clear();
  for (int idet = fNDet; idet--;) {
    AliAlgDet* det = GetDetector(idet);
    det->ResetPool(); // reset used alignment points
  }
}

//____________________________________________
Bool_t AliAlgSteer::TestLocalSolution()
{
  // test track local solution
  TVectorD rhs;
  AliSymMatrix* mat = BuildMatrix(rhs);
  if (!mat)
    return kFALSE;
  //  mat->Print("long data");
  //  rhs.Print();
  TVectorD vsl(rhs);
  if (!mat->SolveChol(rhs, vsl, kTRUE)) {
    delete mat;
    return kFALSE;
  }
  //
  /*
  // print solution vector
  int nlocpar = fAlgTrack->GetNLocPar();
  int nlocparETP = fAlgTrack->GetNLocExtPar(); // parameters of external track param
  printf("ETP Update: ");
  for (int i=0;i<nlocparETP;i++) printf("%+.2e(%+.2e) ",vsl[i],Sqrt((*mat)(i,i))); printf("\n");
  //
  if (nlocpar>nlocparETP) printf("Mat.Corr. update:\n");
  for (int ip=fAlgTrack->GetNPoints();ip--;) {
    AliAlgPoint* pnt = fAlgTrack->GetPoint(ip);
    int npm = pnt->GetNMatPar();
    const float* expMatCov  = pnt->GetMatCorrCov(); // its error
    int offs  = pnt->GetMaxLocVarID() - npm;
    for (int ipar=0;ipar<npm;ipar++) {
      int parI = offs + ipar;
      double err = Sqrt(expMatCov[ipar]);
      printf("Pnt:%3d MatVar:%d DOF %3d | %+.3e(%+.3e) -> sig:%+.3e -> pull: %+.2e\n",
      	     ip,ipar,parI,vsl[parI],Sqrt((*mat)(parI,parI)), err,vsl[parI]/err);
    }
  }
  */
  //
  // increment current params by new solution
  rhs.SetElements(fAlgTrack->GetLocPars());
  vsl += rhs;
  fAlgTrack->SetLocPars(vsl.GetMatrixArray());
  fAlgTrack->CalcResiduals();
  delete mat;
  //
  return kTRUE;
}

//____________________________________________
AliSymMatrix* AliAlgSteer::BuildMatrix(TVectorD& vec)
{
  // build matrix/vector for local track
  int npnt = fAlgTrack->GetNPoints();
  int nlocpar = fAlgTrack->GetNLocPar();
  //
  vec.ResizeTo(nlocpar);
  memset(vec.GetMatrixArray(), 0, nlocpar * sizeof(double));
  AliSymMatrix* matp = new AliSymMatrix(nlocpar);
  AliSymMatrix& mat = *matp;
  //
  for (int ip = npnt; ip--;) {
    AliAlgPoint* pnt = fAlgTrack->GetPoint(ip);
    //
    if (pnt->ContainsMeasurement()) {
      //      pnt->Print("meas");
      for (int idim = 2; idim--;) {                       // each point has 2 position residuals
        double sigma2 = pnt->GetErrDiag(idim);            // residual error
        double resid = fAlgTrack->GetResidual(idim, ip);  // residual
        double* deriv = fAlgTrack->GetDResDLoc(idim, ip); // array of Dresidual/Dparams
        //
        double sg2inv = 1. / sigma2;
        for (int parI = nlocpar; parI--;) {
          vec[parI] -= deriv[parI] * resid * sg2inv;
          //  printf("%d %d %d %+e %+e %+e -> %+e\n",ip,idim,parI,sg2inv,deriv[parI],resid,vec[parI]);
          //	  for (int parJ=nlocpar;parJ--;) {
          for (int parJ = parI + 1; parJ--;) {
            mat(parI, parJ) += deriv[parI] * deriv[parJ] * sg2inv;
          }
        }
      } // loop over 2 orthogonal measurements at the point
    }   // derivarives at measured points
    //
    // if the point contains material, consider its expected kinks, eloss
    // as measurements
    if (pnt->ContainsMaterial()) {
      // at least 4 parameters: 2 spatial + 2 angular kinks with 0 expectaction
      int npm = pnt->GetNMatPar();
      // const float* expMatCorr = pnt->GetMatCorrExp(); // expected correction (diagonalized)
      const float* expMatCov = pnt->GetMatCorrCov(); // its error
      int offs = pnt->GetMaxLocVarID() - npm;
      for (int ipar = 0; ipar < npm; ipar++) {
        int parI = offs + ipar;
        // expected
        //	vec[parI] -= expMatCorr[ipar]/expMatCov[ipar]; // consider expectation as measurement
        mat(parI, parI) += 1. / expMatCov[ipar]; // this measurement is orthogonal to all others
                                                 //printf("Pnt:%3d MatVar:%d DOF %3d | ExpVal: %+e Cov: %+e\n",ip,ipar,parI, expMatCorr[ipar], expMatCov[ipar]);
      }
    } // material effect descripotion params
    //
  } // loop over track points
  //
  return matp;
}

//____________________________________________
void AliAlgSteer::InitMPRecOutput()
{
  // prepare MP record output
  if (!fMPRecord)
    fMPRecord = new AliAlgMPRecord();
  //
  TString mo = Form("%s%s", fMPDatFileName.Data(), ".root");
  fMPRecFile = TFile::Open(mo.Data(), "recreate");
  if (!fMPRecFile)
    AliFatalF("Failed to create output file %s", mo.Data());
  //
  fMPRecTree = new TTree("mpTree", "MPrecord Tree");
  fMPRecTree->Branch("mprec", "AliAlgMPRecord", &fMPRecord);
  //
}

//____________________________________________
void AliAlgSteer::InitResidOutput()
{
  // prepare residual output
  if (!fCResid)
    fCResid = new AliAlgRes();
  //
  fResidFile = TFile::Open(fResidFileName.Data(), "recreate");
  if (!fResidFile)
    AliFatalF("Failed to create output file %s", fResidFileName.Data());
  //
  fResidTree = new TTree("res", "Control Residuals");
  fResidTree->Branch("t", "AliAlgRes", &fCResid);
  //
}

//____________________________________________
void AliAlgSteer::CloseMPRecOutput()
{
  // close output
  if (!fMPRecFile)
    return;
  AliInfoF("Closing %s", fMPRecFile->GetName());
  fMPRecFile->cd();
  fMPRecTree->Write();
  delete fMPRecTree;
  fMPRecTree = 0;
  fMPRecFile->Close();
  delete fMPRecFile;
  fMPRecFile = 0;
  delete fMPRecord;
  fMPRecord = 0;
}

//____________________________________________
void AliAlgSteer::CloseResidOutput()
{
  // close output
  if (!fResidFile)
    return;
  AliInfoF("Closing %s", fResidFile->GetName());
  fResidFile->cd();
  fResidTree->Write();
  delete fResidTree;
  fResidTree = 0;
  fResidFile->Close();
  delete fResidFile;
  fResidFile = 0;
  delete fCResid;
  fCResid = 0;
}

//____________________________________________
void AliAlgSteer::CloseMilleOutput()
{
  // close output
  if (fMille)
    AliInfoF("Closing %s%s", fMPDatFileName.Data(), fgkMPDataExt);
  delete fMille;
  fMille = 0;
}

//____________________________________________
void AliAlgSteer::SetMPDatFileName(const char* name)
{
  // set output file name
  fMPDatFileName = name;
  // strip root or mille extensions, they will be added automatically later
  if (fMPDatFileName.EndsWith(fgkMPDataExt))
    fMPDatFileName.Remove(fMPDatFileName.Length() - strlen(fgkMPDataExt));
  else if (fMPDatFileName.EndsWith(".root"))
    fMPDatFileName.Remove(fMPDatFileName.Length() - strlen(".root"));
  //
  if (fMPDatFileName.IsNull())
    fMPDatFileName = "mpData";
  //
}

//____________________________________________
void AliAlgSteer::SetMPParFileName(const char* name)
{
  // set MP params output file name
  fMPParFileName = name;
  if (fMPParFileName.IsNull())
    fMPParFileName = "mpParams.txt";
  //
}

//____________________________________________
void AliAlgSteer::SetMPConFileName(const char* name)
{
  // set MP constraints output file name
  fMPConFileName = name;
  if (fMPConFileName.IsNull())
    fMPConFileName = "mpConstraints.txt";
  //
}

//____________________________________________
void AliAlgSteer::SetMPSteerFileName(const char* name)
{
  // set MP constraints output file name
  fMPSteerFileName = name;
  if (fMPSteerFileName.IsNull())
    fMPSteerFileName = "mpConstraints.txt";
  //
}

//____________________________________________
void AliAlgSteer::SetResidFileName(const char* name)
{
  // set output file name
  fResidFileName = name;
  if (fResidFileName.IsNull())
    fResidFileName = "mpControlRes.root";
  //
}

//____________________________________________
void AliAlgSteer::SetOutCDBPath(const char* name)
{
  // set output storage name
  fOutCDBPath = name;
  if (fOutCDBPath.IsNull())
    fOutCDBPath = "local://outOCDB";
  //
}

//____________________________________________
void AliAlgSteer::SetObligatoryDetector(Int_t detID, Int_t trtype, Bool_t v)
{
  // mark detector presence obligatory in the track of given type
  AliAlgDet* det = GetDetectorByDetID(detID);
  if (!det) {
    AliErrorF("Detector %d is not defined", detID);
  }
  if (v)
    fObligatoryDetPattern[trtype] |= 0x1 << detID;
  else
    fObligatoryDetPattern[trtype] &= ~(0x1 << detID);
  if (det->IsObligatory(trtype) != v)
    det->SetObligatory(trtype, v);
  //
}

//____________________________________________
Bool_t AliAlgSteer::AddVertexConstraint()
{
  // if vertex is set and if particle is primary, add vertex as a meared point
  //
  const AliESDtrack* esdTr = fESDTrack[0];
  if (!fVertex || !esdTr)
    return kFALSE;
  //
  if (esdTr->GetNcls(0) < fMinITSClforVC)
    return kFALSE; // not enough ITS clusters
  if (!AliAlgDetITS::CheckHitPattern(esdTr, fITSPattforVC))
    return kFALSE;
  //
  AliExternalTrackParam trc = *esdTr;
  Double_t dz[2], dzCov[3];
  if (!trc.PropagateToDCA(fVertex, AliTrackerBase::GetBz(), 2 * fMaxDCAforVC[0], dz, dzCov))
    return kFALSE;
  //
  // check if primary candidate
  if (Abs(dz[0]) > fMaxDCAforVC[0] || Abs(dz[1]) > fMaxDCAforVC[1])
    return kFALSE;
  Double_t covar[6];
  fVertex->GetCovMatrix(covar);
  Double_t p[2] = {trc.GetParameter()[0] - dz[0], trc.GetParameter()[1] - dz[1]};
  Double_t c[3] = {0.5 * (covar[0] + covar[2]), 0., covar[5]};
  Double_t chi2 = trc.GetPredictedChi2(p, c);
  if (chi2 > fMaxChi2forVC)
    return kFALSE;
  //
  // assing measured vertex rotated to VtxSens frame as reference point
  double xyz[3], xyzT[3];
  fVertex->GetXYZ(xyz);
  fVtxSens->SetAlpha(trc.GetAlpha());
  // usually translation from GLO to TRA frame should go via matrix T2G
  // but for the VertexSensor Local and Global are the same frames
  fVtxSens->ApplyCorrection(xyz);
  fVtxSens->GetMatrixT2L().MasterToLocal(xyz, xyzT);
  fRefPoint->SetSensor(fVtxSens);
  fRefPoint->SetAlphaSens(fVtxSens->GetAlpTracking());
  fRefPoint->SetXYZTracking(xyzT);
  fRefPoint->SetYZErrTracking(c);
  fRefPoint->SetContainsMeasurement(kTRUE);
  fRefPoint->Init();
  //
  return kTRUE;
}

//______________________________________________________
void AliAlgSteer::WriteCalibrationResults() const
{
  // writes output calibration
  CleanOCDB();
  AliCDBManager::Instance()->SetDefaultStorage(fOutCDBPath.Data());
  //
  AliAlgDet* det;
  for (int idet = 0; idet < kNDetectors; idet++) {
    if (!(det = GetDetectorByDetID(idet)) || det->IsDisabled())
      continue;
    det->WriteCalibrationResults();
  }
  //
}

//______________________________________________________
void AliAlgSteer::SetOutCDBRunRange(int rmin, int rmax)
{
  // set output run range
  fOutCDBRunRange[0] = rmin >= 0 ? rmin : 0;
  fOutCDBRunRange[1] = rmax > fOutCDBRunRange[0] ? rmax : AliCDBRunRange::Infinity();
}

//______________________________________________________
Bool_t AliAlgSteer::LoadRefOCDB()
{
  // setup OCDB whose objects will be used as a reference with respect to which the
  // alignment/calibration will prodice its corrections.
  // Detectors which need some reference calibration data must use this one
  //
  //
  AliInfo("Loading reference OCDB");
  CleanOCDB();
  AliCDBManager* man = AliCDBManager::Instance();
  //
  if (!fRefOCDBConf.IsNull() && !gSystem->AccessPathName(fRefOCDBConf.Data(), kFileExists)) {
    AliInfoF("Executing reference OCDB setup macro %s", fRefOCDBConf.Data());
    if (fRefRunNumber > 0)
      gROOT->ProcessLine(Form(".x %s(%d)", fRefOCDBConf.Data(), fRefRunNumber));
    else
      gROOT->ProcessLine(Form(".x %s", fRefOCDBConf.Data()));
  } else {
    AliWarningF("No reference OCDB config macro %s is found, assume raw:// with run %d",
                fRefOCDBConf.Data(), AliCDBRunRange::Infinity());
    man->SetRaw(kTRUE);
    man->SetRun(AliCDBRunRange::Infinity());
  }
  //
  if (AliGeomManager::GetGeometry()) {
    AliInfo("Destroying current geometry before loading reference one");
    AliGeomManager::Destroy();
  }
  AliGeomManager::LoadGeometry("geometry.root");
  if (!AliGeomManager::GetGeometry())
    AliFatal("Failed to load geometry, cannot run");
  //
  TString detList = "";
  for (int i = 0; i < kNDetectors; i++) {
    detList += GetDetNameByDetID(i);
    detList += " ";
  }
  AliGeomManager::ApplyAlignObjsFromCDB(detList.Data());
  //
  fRefOCDBLoaded++;
  //
  return kTRUE;
}

//________________________________________________________
AliAlgDet* AliAlgSteer::GetDetOfDOFID(int id) const
{
  // return detector owning DOF with this ID
  for (int i = fNDet; i--;) {
    AliAlgDet* det = GetDetector(i);
    if (det->OwnsDOFID(id))
      return det;
  }
  return 0;
}

//________________________________________________________
AliAlgVol* AliAlgSteer::GetVolOfDOFID(int id) const
{
  // return volume owning DOF with this ID
  for (int i = fNDet; i--;) {
    AliAlgDet* det = GetDetector(i);
    if (det->OwnsDOFID(id))
      return det->GetVolOfDOFID(id);
  }
  if (fVtxSens && fVtxSens->OwnsDOFID(id))
    return fVtxSens;
  return 0;
}

//________________________________________________________
void AliAlgSteer::Terminate(Bool_t doStat)
{
  // finalize processing
  if (fRunNumber > 0)
    FillStatHisto(kRunDone);
  if (doStat) {
    if (fDOFStat)
      delete fDOFStat;
    fDOFStat = new AliAlgDOFStat(fNDOFs);
  }
  if (fVtxSens)
    fVtxSens->FillDOFStat(fDOFStat);
  //
  for (int i = fNDet; i--;)
    GetDetector(i)->Terminate();
  CloseMPRecOutput();
  CloseMilleOutput();
  CloseResidOutput();
  Print("stat");
  //
}

//________________________________________________________
Char_t* AliAlgSteer::GetDOFLabelTxt(int idf) const
{
  // get DOF full label
  AliAlgVol* vol = GetVolOfDOFID(idf);
  if (vol)
    return Form("%d_%s_%s", GetGloParLab(idf), vol->GetSymName(),
                vol->GetDOFName(idf - vol->GetFirstParGloID()));
  //
  // this might be detector-specific calibration dof
  AliAlgDet* det = GetDetOfDOFID(idf);
  if (det)
    return Form("%d_%s_%s", GetGloParLab(idf), det->GetName(),
                det->GetCalibDOFName(idf - det->GetFirstParGloID()));
  return 0;
}

//********************* interaction with PEDE **********************

//______________________________________________________
void AliAlgSteer::GenPedeSteerFile(const Option_t* opt) const
{
  // produce steering file template for PEDE + params and constraints
  //
  enum { kOff,
         kOn,
         kOnOn };
  const char* cmt[3] = {"  ", "! ", "!!"};
  const char* kSolMeth[] = {"inversion", "diagonalization", "fullGMRES", "sparseGMRES", "cholesky", "HIP"};
  const int kDefNIter = 3;     // default number of iterations to ask
  const float kDefDelta = 0.1; // def. delta to exit
  TString opts = opt;
  opts.ToLower();
  AliInfoF(
    "Generating MP2 templates:\n"
    "Steering   :\t%s\n"
    "Parameters :\t%s\n"
    "Constraints:\t%s\n",
    fMPSteerFileName.Data(), fMPParFileName.Data(), fMPConFileName.Data());
  //
  FILE* parFl = fopen(fMPParFileName.Data(), "w+");
  FILE* strFl = fopen(fMPSteerFileName.Data(), "w+");
  //
  // --- template of steering file
  fprintf(strFl, "%-20s%s %s\n", fMPParFileName.Data(), cmt[kOnOn], "parameters template");
  fprintf(strFl, "%-20s%s %s\n", fMPConFileName.Data(), cmt[kOnOn], "constraints template");
  //
  fprintf(strFl, "\n\n%s %s\n", cmt[kOnOn], "MUST uncomment 1 solving methods and tune it");
  //
  int nm = sizeof(kSolMeth) / sizeof(char*);
  for (int i = 0; i < nm; i++) {
    fprintf(strFl, "%s%s %-20s %2d %.2f %s\n", cmt[kOn], "method", kSolMeth[i], kDefNIter, kDefDelta, cmt[kOnOn]);
  }
  //
  const float kDefChi2F0 = 20., kDefChi2F = 3.; // chi2 factors for 1st and following iterations
  const float kDefDWFrac = 0.2;                 // cut outliers with downweighting above this factor
  const int kDefOutlierDW = 4;                  // start Cauchy function downweighting from iteration
  const int kDefEntries = 25;                   // min entries per DOF to allow its variation
  //
  fprintf(strFl, "\n\n%s %s\n", cmt[kOnOn], "Optional settings");
  fprintf(strFl, "\n%s%-20s %.2f %.2f %s %s\n", cmt[kOn], "chisqcut", kDefChi2F0, kDefChi2F,
          cmt[kOnOn], "chi2 cut factors for 1st and next iterations");
  fprintf(strFl, "%s%-20s %2d %s %s\n", cmt[kOn], "outlierdownweighting", kDefOutlierDW,
          cmt[kOnOn], "iteration for outliers downweighting with Cauchi factor");
  fprintf(strFl, "%s%-20s %.3f %s %s\n", cmt[kOn], "dwfractioncut", kDefDWFrac,
          cmt[kOnOn], "cut outliers with downweighting above this factor");
  fprintf(strFl, "%s%-20s %2d %s %s\n", cmt[kOn], "entries", kDefEntries,
          cmt[kOnOn], "min entries per DOF to allow its variation");
  //
  fprintf(strFl, "\n\n\n%s%-20s %s %s\n\n\n", cmt[kOff], "CFiles", cmt[kOnOn], "put below *.mille files list");
  //
  if (fVtxSens)
    fVtxSens->WritePedeInfo(parFl, opt);
  //
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det || det->IsDisabled())
      continue;
    det->WritePedeInfo(parFl, opt);
    //
  }
  //
  WritePedeConstraints();
  //
  fclose(strFl);
  fclose(parFl);
  //
}

//___________________________________________________________
Bool_t AliAlgSteer::ReadParameters(const char* parfile, Bool_t useErrors)
{
  // read parameters file (millepede output)
  if (fNDOFs < 1 || !fGloParVal || !fGloParErr) {
    AliErrorF("Something is wrong in init: fNDOFs=%d fGloParVal=%p fGloParErr=%p",
              fNDOFs, fGloParVal, fGloParErr);
  }
  ifstream inpf(parfile);
  if (!inpf.good()) {
    printf("Failed on input filename %s\n", parfile);
    return kFALSE;
  }
  memset(fGloParVal, 0, fNDOFs * sizeof(float));
  if (useErrors)
    memset(fGloParErr, 0, fNDOFs * sizeof(float));
  int cnt = 0;
  TString fline;
  fline.ReadLine(inpf);
  fline = fline.Strip(TString::kBoth, ' ');
  fline.ToLower();
  if (!fline.BeginsWith("parameter")) {
    AliErrorF("First line is not parameter keyword:\n%s", fline.Data());
    return kFALSE;
  }
  double v0, v1, v2;
  int lab, asg = 0, asg0 = 0;
  while (fline.ReadLine(inpf)) {
    cnt++;
    fline = fline.Strip(TString::kBoth, ' ');
    if (fline.BeginsWith("!") || fline.BeginsWith("*"))
      continue; // ignore comment
    int nr = sscanf(fline.Data(), "%d%lf%lf%lf", &lab, &v0, &v1, &v2);
    if (nr < 3) {
      AliErrorF("Expected to read at least 3 numbers, got %d, this is NOT milleped output", nr);
      AliErrorF("line (%d) was:\n%s", cnt, fline.Data());
      return kFALSE;
    }
    if (nr == 3)
      asg0++;
    int parID = Label2ParID(lab);
    if (parID < 0 || parID >= fNDOFs) {
      AliErrorF("Invalid label %d at line %d -> ParID=%d", lab, cnt, parID);
      return kFALSE;
    }
    fGloParVal[parID] = -v0;
    if (useErrors)
      fGloParErr[parID] = v1;
    asg++;
    //
  };
  AliInfoF("Read %d lines, assigned %d values, %d dummy", cnt, asg, asg0);
  //
  return kTRUE;
}

//______________________________________________________
void AliAlgSteer::CheckConstraints(const char* params)
{
  // check how the constraints are satisfied with already uploaded or provided params
  //
  if (params && !ReadParameters(params)) {
    AliErrorF("Failed to load parameters from %s", params);
    return;
  }
  //
  int ncon = GetNConstraints();
  for (int icon = 0; icon < ncon; icon++) {
    const AliAlgConstraint* con = GetConstraint(icon);
    con->CheckConstraint();
  }
  //
}

//___________________________________________________________
void AliAlgSteer::MPRec2Mille(const char* mprecfile, const char* millefile, Bool_t bindata)
{
  // converts MPRecord tree to millepede binary format
  TFile* flmpr = TFile::Open(mprecfile);
  if (!flmpr) {
    AliErrorClassF("Failed to open MPRecord file %s", mprecfile);
    return;
  }
  TTree* mprTree = (TTree*)flmpr->Get("mpTree");
  if (!mprTree) {
    AliErrorClassF("No mpTree in xMPRecord file %s", mprecfile);
    return;
  }
  MPRec2Mille(mprTree, millefile, bindata);
  delete mprTree;
  flmpr->Close();
  delete flmpr;
}

//___________________________________________________________
void AliAlgSteer::MPRec2Mille(TTree* mprTree, const char* millefile, Bool_t bindata)
{
  // converts MPRecord tree to millepede binary format
  //
  TBranch* br = mprTree->GetBranch("mprec");
  if (!br) {
    AliErrorClass("provided tree does not contain branch mprec");
    return;
  }
  AliAlgMPRecord* rec = new AliAlgMPRecord();
  br->SetAddress(&rec);
  int nent = mprTree->GetEntries();
  TString mlname = millefile;
  if (mlname.IsNull())
    mlname = "mpRec2mpData";
  if (!mlname.EndsWith(fgkMPDataExt))
    mlname += fgkMPDataExt;
  Mille* mille = new Mille(mlname, bindata);
  TArrayF buffDLoc;
  for (int i = 0; i < nent; i++) {
    br->GetEntry(i);
    int nr = rec->GetNResid(); // number of residual records
    int nloc = rec->GetNVarLoc();
    if (buffDLoc.GetSize() < nloc)
      buffDLoc.Set(nloc + 100);
    float* buffLocV = buffDLoc.GetArray();
    const float* recDGlo = rec->GetArrGlo();
    const float* recDLoc = rec->GetArrLoc();
    const short* recLabLoc = rec->GetArrLabLoc();
    const int* recLabGlo = rec->GetArrLabGlo();
    //
    for (int ir = 0; ir < nr; ir++) {
      memset(buffLocV, 0, nloc * sizeof(float));
      int ndglo = rec->GetNDGlo(ir);
      int ndloc = rec->GetNDLoc(ir);
      // fill 0-suppressed array from MPRecord to non-0-suppressed array of Mille
      for (int l = ndloc; l--;)
        buffLocV[recLabLoc[l]] = recDLoc[l];
      //
      mille->mille(nloc, buffLocV, ndglo, recDGlo, recLabGlo, rec->GetResid(ir), rec->GetResErr(ir));
      //
      recLabGlo += ndglo; // next record
      recDGlo += ndglo;
      recLabLoc += ndloc;
      recDLoc += ndloc;
    }
    mille->end();
  }
  delete mille;
  br->SetAddress(0);
  delete rec;
}

//____________________________________________________________
void AliAlgSteer::FillStatHisto(int type, float w)
{
  if (!fHistoStat)
    CreateStatHisto();
  fHistoStat->Fill((IsCosmic() ? kNHVars : 0) + type, w);
}

//____________________________________________________________
void AliAlgSteer::CreateStatHisto()
{
  fHistoStat = new TH1F("stat", "stat", 2 * kNHVars, -0.5, 2 * kNHVars - 0.5);
  fHistoStat->SetDirectory(0);
  TAxis* xax = fHistoStat->GetXaxis();
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < kNHVars; i++) {
      xax->SetBinLabel(j * kNHVars + i + 1, Form("%s.%s", j ? "CSM" : "COL", fgkHStatName[i]));
    }
  }
}

//____________________________________________________________
void AliAlgSteer::PrintLabels() const
{
  // print global IDs and Labels
  for (int i = 0; i < fNDOFs; i++)
    printf("%5d %s\n", i, GetDOFLabelTxt(i));
}

//____________________________________________________________
Int_t AliAlgSteer::Label2ParID(int lab) const
{
  // convert Mille label to ParID (slow)
  int ind = FindKeyIndex(lab, fOrderedLbl, fNDOFs);
  if (ind < 0)
    return -1;
  return fLbl2ID[ind];
}

//____________________________________________________________
void AliAlgSteer::AddAutoConstraints()
{
  // add default constraints on children cumulative corrections within the volumes
  for (int idet = 0; idet < fNDet; idet++) {
    AliAlgDet* det = GetDetector(idet);
    if (det->IsDisabled())
      continue;
    det->AddAutoConstraints();
  }
  AliInfoF("Added %d automatic constraints", GetNConstraints());
}

//____________________________________________________________
void AliAlgSteer::WritePedeConstraints() const
{
  // write constraints file
  FILE* conFl = fopen(fMPConFileName.Data(), "w+");
  //
  int nconstr = GetNConstraints();
  for (int icon = 0; icon < nconstr; icon++)
    GetConstraint(icon)->WriteChildrenConstraints(conFl);
  //
  fclose(conFl);
}

//____________________________________________________________
void AliAlgSteer::FixLowStatFromDOFStat(Int_t thresh)
{
  // fix DOFs having stat below threshold
  //
  if (!fDOFStat) {
    AliError("No object with DOFs statistics");
    return;
  }
  if (fNDOFs != fDOFStat->GetNDOFs()) {
    AliErrorF("Discrepancy between NDOFs=%d of and statistics object: %d", fNDOFs, fDOFStat->GetNDOFs());
    return;
  }
  for (int parID = 0; parID < fNDOFs; parID++) {
    if (fDOFStat->GetStat(parID) >= thresh)
      continue;
    fGloParErr[parID] = -999.;
  }
  //
}

//____________________________________________________________
void AliAlgSteer::LoadStat(const char* flname)
{
  // load statistics histos from external file produced by alignment task
  TFile* fl = TFile::Open(flname);
  //
  TObject *hdfO = 0, *hstO = 0;
  TList* lst = (TList*)fl->Get("clist");
  if (lst) {
    hdfO = lst->FindObject("DOFstat");
    if (hdfO)
      lst->Remove(hdfO);
    hstO = lst->FindObject("stat");
    if (hstO)
      lst->Remove(hstO);
    delete lst;
  } else {
    hdfO = fl->Get("DOFstat");
    hstO = fl->Get("stat");
  }
  TH1F* hst = 0;
  if (hstO && (hst = dynamic_cast<TH1F*>(hstO)))
    hst->SetDirectory(0);
  else
    AliWarning("did not find stat histo");
  //
  AliAlgDOFStat* dofSt = 0;
  if (!hdfO || !(dofSt = dynamic_cast<AliAlgDOFStat*>(hdfO)))
    AliWarning("did not find DOFstat object");
  //
  SetHistoStat(hst);
  SetDOFStat(dofSt);
  //
  fl->Close();
  delete fl;
}

//______________________________________________
void AliAlgSteer::CheckSol(TTree* mpRecTree, Bool_t store,
                           Bool_t verbose, Bool_t loc, const char* outName)
{
  // do fast check of pede solution with MPRecord tree
  AliAlgResFast* rLG = store ? new AliAlgResFast() : 0;
  AliAlgResFast* rL = store && loc ? new AliAlgResFast() : 0;
  TTree *trLG = 0, *trL = 0;
  TFile* outFile = 0;
  if (store) {
    TString outNS = outName;
    if (outNS.IsNull())
      outNS = "resFast";
    if (!outNS.EndsWith(".root"))
      outNS += ".root";
    outFile = TFile::Open(outNS.Data(), "recreate");
    trLG = new TTree("resFLG", "Fast residuals with LG correction");
    trLG->Branch("rLG", "AliAlgResFast", &rLG);
    //
    if (rL) {
      trL = new TTree("resFL", "Fast residuals with L correction");
      trL->Branch("rL", "AliAlgResFast", &rL);
    }
  }
  //
  AliAlgMPRecord* rec = new AliAlgMPRecord();
  mpRecTree->SetBranchAddress("mprec", &rec);
  int nrec = mpRecTree->GetEntriesFast();
  for (int irec = 0; irec < nrec; irec++) {
    mpRecTree->GetEntry(irec);
    CheckSol(rec, rLG, rL, verbose, loc);
    // store even in case of failure, to have the trees aligned with controlRes
    if (trLG)
      trLG->Fill();
    if (trL)
      trL->Fill();
  }
  //
  // save
  if (trLG) {
    outFile->cd();
    trLG->Write();
    delete trLG;
    if (trL) {
      trL->Write();
      delete trL;
    }
    outFile->Close();
    delete outFile;
  }
  //
}

//______________________________________________
Bool_t AliAlgSteer::CheckSol(AliAlgMPRecord* rec,
                             AliAlgResFast* rLG, AliAlgResFast* rL,
                             Bool_t verbose, Bool_t loc)
{
  // Check pede solution using derivates, rather than updated geometry
  // If loc==true, also produces residuals for current geometry,
  // neglecting global corrections
  //
  if (rL)
    loc = kTRUE; // if local sol. tree asked, always evaluate it
  //
  int nres = rec->GetNResid();
  //
  const float* recDGlo = rec->GetArrGlo();
  const float* recDLoc = rec->GetArrLoc();
  const short* recLabLoc = rec->GetArrLabLoc();
  const int* recLabGlo = rec->GetArrLabGlo();
  int nvloc = rec->GetNVarLoc();
  //
  // count number of real measurement duplets and material correction fake 4-plets
  int nPoints = 0;
  int nMatCorr = 0;
  for (int irs = 0; irs < nres; irs++) {
    if (rec->GetNDGlo(irs) > 0) {
      if (irs == nres - 1 || rec->GetNDGlo(irs + 1) == 0)
        AliFatal("Real coordinate measurements must come in pairs");
      nPoints++;
      irs++; // skip 2nd
      continue;
    } else if (rec->GetResid(irs) == 0 && rec->GetVolID(irs) == -1) { // material corrections have 0 residual
      nMatCorr++;
    } else { // might be fixed parameter, global derivs are skept
      nPoints++;
      irs++; // skip 2nd
      continue;
    }
  }
  //
  if (nMatCorr % 4)
    AliWarningF("Error? NMatCorr=%d is not multiple of 4", nMatCorr);
  //
  if (rLG) {
    rLG->Clear();
    rLG->SetNPoints(nPoints);
    rLG->SetNMatSol(nMatCorr);
    rLG->SetCosmic(rec->IsCosmic());
  }
  if (rL) {
    rL->Clear();
    rL->SetNPoints(nPoints);
    rL->SetNMatSol(nMatCorr);
    rL->SetCosmic(rec->IsCosmic());
  }
  //
  AliSymMatrix* matpG = new AliSymMatrix(nvloc);
  TVectorD *vecp = 0, *vecpG = new TVectorD(nvloc);
  //
  if (loc)
    vecp = new TVectorD(nvloc);
  //
  float chi2Ini = 0, chi2L = 0, chi2LG = 0;
  //
  // residuals, accounting for global solution
  double* resid = new Double_t[nres];
  int* volID = new Int_t[nres];
  for (int irs = 0; irs < nres; irs++) {
    double resOr = rec->GetResid(irs);
    resid[irs] = resOr;
    //
    int ndglo = rec->GetNDGlo(irs);
    int ndloc = rec->GetNDLoc(irs);
    volID[irs] = 0;
    for (int ig = 0; ig < ndglo; ig++) {
      int lbI = recLabGlo[ig];
      int idP = Label2ParID(lbI);
      if (idP < 0)
        AliFatalF("Did not find parameted for label %d\n", lbI);
      double parVal = GetGloParVal()[idP];
      //      resid[irs] -= parVal*recDGlo[ig];
      resid[irs] += parVal * recDGlo[ig];
      if (!ig) {
        AliAlgVol* vol = GetVolOfDOFID(idP);
        if (vol)
          volID[irs] = vol->GetVolID();
        else
          volID[irs] = -2; // calibration DOF !!! TODO
      }
    }
    //
    double sg2inv = rec->GetResErr(irs);
    sg2inv = 1. / (sg2inv * sg2inv);
    //
    chi2Ini += resid[irs] * resid[irs] * sg2inv; // chi accounting for global solution only
    //
    // Build matrix to solve local parameters
    for (int il = 0; il < ndloc; il++) {
      int lbLI = recLabLoc[il]; // id of local variable
      (*vecpG)[lbLI] -= recDLoc[il] * resid[irs] * sg2inv;
      if (loc)
        (*vecp)[lbLI] -= recDLoc[il] * resOr * sg2inv;
      for (int jl = il + 1; jl--;) {
        int lbLJ = recLabLoc[jl]; // id of local variable
        (*matpG)(lbLI, lbLJ) += recDLoc[il] * recDLoc[jl] * sg2inv;
      }
    }
    //
    recLabGlo += ndglo; // prepare for next record
    recDGlo += ndglo;
    recLabLoc += ndloc;
    recDLoc += ndloc;
    //
  }
  //
  if (rL)
    rL->SetChi2Ini(chi2Ini);
  if (rLG)
    rLG->SetChi2Ini(chi2Ini);
  //
  TVectorD vecSol(nvloc);
  TVectorD vecSolG(nvloc);
  //
  if (!matpG->SolveChol(*vecpG, vecSolG, kFALSE)) {
    AliInfo("Failed to solve track corrected for globals\n");
    delete matpG;
    matpG = 0;
  } else if (loc) { // solution with local correction only
    if (!matpG->SolveChol(*vecp, vecSol, kFALSE)) {
      AliInfo("Failed to solve track corrected for globals\n");
      delete matpG;
      matpG = 0;
    }
  }
  delete vecpG;
  delete vecp;
  if (!matpG) { // failed
    delete[] resid;
    delete[] volID;
    if (rLG)
      rLG->Clear();
    if (rL)
      rL->Clear();
    return kFALSE;
  }
  // check
  recDGlo = rec->GetArrGlo();
  recDLoc = rec->GetArrLoc();
  recLabLoc = rec->GetArrLabLoc();
  recLabGlo = rec->GetArrLabGlo();
  //
  if (verbose) {
    printf(loc ? "Sol L/LG:\n" : "Sol LG:\n");
    int nExtP = (nvloc % 4) ? 5 : 4;
    for (int i = 0; i < nExtP; i++)
      loc ? printf("%+.3e/%+.3e ", vecSol[i], vecSolG[i]) : printf("%+.3e ", vecSolG[i]);
    printf("\n");
    Bool_t nln = kTRUE;
    int cntL = 0;
    for (int i = nExtP; i < nvloc; i++) {
      nln = kTRUE;
      loc ? printf("%+.3e/%+.3e ", vecSol[i], vecSolG[i]) : printf("%+.3e ", vecSolG[i]);
      if (((++cntL) % 4) == 0) {
        printf("\n");
        nln = kFALSE;
      }
    }
    if (!nln)
      printf("\n");
    if (loc)
      printf("%3s (%9s) %6s | [ %7s:%7s ] [ %7s:%7s ]\n", "Pnt", "Label",
             "Sigma", "resid", "pull/L ", "resid", "pull/LG");
    else
      printf("%3s (%9s) %6s | [ %7s:%7s ]\n", "Pnt", "Label",
             "Sigma", "resid", "pull/LG");
  }
  int idMeas = -1, pntID = -1, matID = -1;
  for (int irs = 0; irs < nres; irs++) {
    double resOr = rec->GetResid(irs);
    double resL = resOr;
    double resLG = resid[irs];
    double sg = rec->GetResErr(irs);
    double sg2Inv = 1 / (sg * sg);
    //
    int ndglo = rec->GetNDGlo(irs);
    int ndloc = rec->GetNDLoc(irs);
    //
    for (int il = 0; il < ndloc; il++) {
      int lbLI = recLabLoc[il]; // id of local variable
      resL += recDLoc[il] * vecSol[lbLI];
      resLG += recDLoc[il] * vecSolG[lbLI];
    }
    //
    chi2L += resL * resL * sg2Inv;    // chi accounting for global solution only
    chi2LG += resLG * resLG * sg2Inv; // chi accounting for global solution only
    //
    if (ndglo || resOr != 0) { // real measurement
      idMeas++;
      if (idMeas > 1)
        idMeas = 0;
      if (idMeas == 0)
        pntID++; // measurements come in pairs
      int lbl = rec->GetVolID(irs);
      lbl = ndglo ? recLabGlo[0] : 0; // TMP, until VolID is filled // RS!!!!
      if (rLG) {
        rLG->SetResSigMeas(pntID, idMeas, resLG, sg);
        if (idMeas == 0)
          rLG->SetLabel(pntID, lbl, volID[irs]);
      }
      if (rL) {
        rL->SetResSigMeas(pntID, idMeas, resL, sg);
        if (idMeas == 0)
          rL->SetLabel(pntID, lbl, volID[irs]);
      }
    } else {
      matID++; // mat.correcitons come in 4-plets, but we fill each separately
      //
      if (rLG)
        rLG->SetMatCorr(matID, resLG, sg);
      if (rL)
        rL->SetMatCorr(matID, resL, sg);
    }
    //
    if (verbose) {
      int lbl = rec->GetVolID(irs);
      lbl = ndglo ? recLabGlo[0] : (resOr == 0 ? -1 : 0); // TMP, until VolID is filled // RS!!!!
      if (loc)
        printf("%3d (%9d) %6.4f | [%+.2e:%+7.2f] [%+.2e:%+7.2f]\n",
               irs, lbl, sg, resL, resL / sg, resLG, resLG / sg);
      else
        printf("%3d (%9d) %6.4f | [%+.2e:%+7.2f]\n",
               irs, lbl, sg, resLG, resLG / sg);
    }
    //
    recLabGlo += ndglo; // prepare for next record
    recDGlo += ndglo;
    recLabLoc += ndloc;
    recDLoc += ndloc;
  }
  if (rL)
    rL->SetChi2(chi2L);
  if (rLG)
    rLG->SetChi2(chi2LG);
  //
  if (verbose) {
    printf("Chi: G = %e | LG = %e", chi2Ini, chi2LG);
    if (loc)
      printf(" | L = %e", chi2L);
    printf("\n");
  }
  // store track corrections
  int nTrCor = nvloc - matID - 1;
  for (int i = 0; i < nTrCor; i++) {
    if (rLG)
      rLG->GetTrCor()[i] = vecSolG[i];
    if (rL)
      rL->GetTrCor()[i] = vecSol[i];
  }
  //
  delete[] resid;
  delete[] volID;
  return kTRUE;
}

//______________________________________________
void AliAlgSteer::ApplyAlignmentFromMPSol()
{
  // apply alignment from millepede solution array to reference alignment level
  AliInfo("Applying alignment from Millepede solution");
  for (int idt = 0; idt < kNDetectors; idt++) {
    AliAlgDet* det = GetDetectorByDetID(idt);
    if (!det || det->IsDisabled())
      continue;
    det->ApplyAlignmentFromMPSol();
  }
  SetMPAlignDone();
  //
}

} // namespace align
} // namespace o2
