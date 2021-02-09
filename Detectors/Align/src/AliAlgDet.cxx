/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include "AliAlgDet.h"
#include "AliAlgSens.h"
#include "AliAlgDet.h"
#include "AliAlgSteer.h"
#include "AliAlgTrack.h"
#include "AliAlgDOFStat.h"
#include "AliAlgConstraint.h"
#include "AliLog.h"
#include "AliGeomManager.h"
#include "AliCDBManager.h"
#include "AliCDBMetaData.h"
#include "AliCDBEntry.h"
#include "AliAlignObj.h"
#include "AliCDBId.h"
#include "AliExternalTrackParam.h"
#include "AliAlignObjParams.h"
#include <TString.h>
#include <TH1.h>
#include <TTree.h>
#include <TFile.h>
#include <stdio.h>

ClassImp(AliAlgDet)

using namespace AliAlgAux;

//____________________________________________
AliAlgDet::AliAlgDet()
:  fNDOFs(0)
  ,fVolIDMin(-1)
  ,fVolIDMax(-1)
  ,fNSensors(0)
  ,fSID2VolID(0)
  ,fNProcPoints(0)
   //
  ,fNCalibDOF(0)
  ,fNCalibDOFFree(0)
  ,fCalibDOF(0)
  ,fFirstParGloID(-1)
  ,fParVals(0)
  ,fParErrs(0)
  ,fParLabs(0)   
   //
  ,fUseErrorParam(0)
  ,fSensors()
  ,fVolumes()
  //
  ,fNPoints(0)
  ,fPoolNPoints(0)
  ,fPoolFreePointID(0)
  ,fPointsPool()
  ,fAlgSteer(0)
{
  // def c-tor
  SetUniqueID(AliAlgSteer::kUndefined); // derived detectors must override this
  fAddError[0] = fAddError[1] = 0;
  //
  for (int i=0;i<kNTrackTypes;i++) {
    fDisabled[i] = kFALSE;
    fObligatory[i] = kFALSE;
    fTrackFlagSel[i] = 0;
    fNPointsSel[i] = 0;
  }
  //
}

//____________________________________________
AliAlgDet::~AliAlgDet()
{
  // d-tor
  fSensors.Clear(); // sensors are also attached as volumes, don't delete them here
  fVolumes.Delete(); // here all is deleted
  fPointsPool.Delete();
}


//____________________________________________
Int_t AliAlgDet::ProcessPoints(const AliESDtrack* esdTr, AliAlgTrack* algTrack, Bool_t inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X 
  // (i.e. upper leg of cosmic track)
  //
  const AliESDfriendTrack* trF(esdTr->GetFriendTrack());
  const AliTrackPointArray* trP(trF->GetTrackPointArray());
  //
  int np(trP->GetNPoints());
  int npSel(0);
  AliAlgPoint* apnt(0);
  for (int ip=0;ip<np;ip++) {
    int vid = trP->GetVolumeID()[ip];
    if (!SensorOfDetector(vid)) continue;
    apnt = GetSensorByVolId(vid)->TrackPoint2AlgPoint(ip, trP, esdTr);
    if (!apnt) continue;
    algTrack->AddPoint(apnt);
    if (inv) apnt->SetInvDir();
    npSel++;
    fNPoints++;
  }
  //
  return npSel;
}

//_________________________________________________________
void AliAlgDet::AcknowledgeNewRun(Int_t run)
{
  // update parameters needed to process this run

  // detector should be able to undo alignment/calibration used during the reco
  UpdateL2GRecoMatrices();

}

//_________________________________________________________
void AliAlgDet::UpdateL2GRecoMatrices()
{
  // Update L2G matrices used for data reconstruction
  //
  AliCDBManager* man = AliCDBManager::Instance();
  AliCDBEntry* ent = man->Get(Form("%s/Align/Data",GetName()));
  const TClonesArray *algArr = (const TClonesArray*)ent->GetObject();
  //
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) {
    AliAlgVol *vol = GetVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->GetParent()) vol->UpdateL2GRecoMatrices(algArr,0);
  }
  //
}


//_________________________________________________________
void  AliAlgDet::ApplyAlignmentFromMPSol()
{
  // apply alignment from millepede solution array to reference alignment level
  AliInfo("Applying alignment from Millepede solution");
  for (int isn=GetNSensors();isn--;) GetSensor(isn)->ApplyAlignmentFromMPSol();
}

//_________________________________________________________
void AliAlgDet::CacheReferenceOCDB()
{
  // if necessary, detector may fetch here some reference OCDB data
  //
  // cache global deltas to avoid preicision problem
  AliCDBManager* man = AliCDBManager::Instance();
  AliCDBEntry* ent = man->Get(Form("%s/Align/Data",GetName()));
  TObjArray* arr = (TObjArray*)ent->GetObject();
  for (int i=arr->GetEntriesFast();i--;) {
    const AliAlignObjParams* par = (const AliAlignObjParams*)arr->At(i);
    AliAlgVol* vol = GetVolume(par->GetSymName());
    if (!vol) {AliErrorF("Volume %s not found",par->GetSymName()); continue;}
    TGeoHMatrix delta;
    par->GetMatrix(delta);
    vol->SetGlobalDeltaRef(delta);
  }
}


//_________________________________________________________
AliAlgPoint* AliAlgDet::GetPointFromPool()
{
  // fetch or create new free point from the pool.
  // detector may override this method to create its own points derived from AliAlgPoint
  //
  if (fPoolFreePointID>=fPoolNPoints) { // expand pool
    fPointsPool.AddAtAndExpand(new AliAlgPoint(), fPoolNPoints++);
  }
  //
  AliAlgPoint* pnt = (AliAlgPoint*) fPointsPool.UncheckedAt(fPoolFreePointID++);
  pnt->Clear();
  return pnt;
  //
}

//_________________________________________________________
void AliAlgDet::ResetPool()
{
  // declare pool free
  fPoolFreePointID = 0;
  fNPoints = 0;
}
 
//_________________________________________________________
void AliAlgDet::DefineVolumes()
{
  // dummy method
  AliError("This method must be implemented by specific detector");
}

//_________________________________________________________
void AliAlgDet::AddVolume(AliAlgVol* vol)
{
  // add volume
  if (GetVolume(vol->GetSymName())) {
    AliFatalF("Volume %s was already added to %s",vol->GetName(),GetName());
  }
  fVolumes.AddLast(vol);
  if (vol->IsSensor()) {
    fSensors.AddLast(vol);
    ((AliAlgSens*)vol)->SetDetector(this);
    Int_t vid = ((AliAlgSens*)vol)->GetVolID();
    if (fVolIDMin<0 || vid<fVolIDMin) fVolIDMin = vid;
    if (fVolIDMax<0 || vid>fVolIDMax) fVolIDMax = vid;
  }
  //
}

//_________________________________________________________
void AliAlgDet::DefineMatrices()
{
  // define transformation matrices. Detectors may override this method
  //
  TGeoHMatrix mtmp;
  //
  TIter next(&fVolumes);
  AliAlgVol* vol(0);
  while ( (vol=(AliAlgVol*)next()) ) {
    // modified global-local matrix
    vol->PrepareMatrixL2G();
    // ideal global-local matrix
    vol->PrepareMatrixL2GIdeal();
    //
  }
  // Now set tracking-local matrix (MUST be done after ALL L2G matrices are done!)
  // Attention: for sensor it is a real tracking matrix extracted from
  // the geometry but for container alignable volumes the tracking frame
  // is used for as the reference for the alignment parameters only,
  // see its definition in the AliAlgVol::PrepateMatrixT2L
  next.Reset();
  while ( (vol=(AliAlgVol*)next()) ) {
    vol->PrepareMatrixT2L();
    if (vol->IsSensor()) ((AliAlgSens*)vol)->PrepareMatrixClAlg(); // alignment matrix
  }
  //
}

//_________________________________________________________
void AliAlgDet::SortSensors()
{
  // build local tables for internal numbering
  fNSensors = fSensors.GetEntriesFast();
  if (!fNSensors) {
    AliWarning("No sensors defined");
    return;
  }
  fSensors.Sort();
  fSID2VolID = new Int_t[fNSensors]; // cash id's for fast binary search
  for (int i=0;i<fNSensors;i++) {
    fSID2VolID[i] = GetSensor(i)->GetVolID();
    GetSensor(i)->SetSID(i);
  }
  //
}

//_________________________________________________________
Int_t AliAlgDet::InitGeom()
{
  // define hiearchy, initialize matrices, return number of global parameters
  if (GetInitGeomDone()) return 0;
  //
  DefineVolumes();
  SortSensors();    // VolID's must be in increasing order
  DefineMatrices();
  //
  // calculate number of global parameters
  int nvol = GetNVolumes();
  fNDOFs = 0;
  for (int iv=0;iv<nvol;iv++) {
    AliAlgVol *vol = GetVolume(iv);
    fNDOFs += vol->GetNDOFs();
  }
  //
  fNDOFs += fNCalibDOF;
  SetInitGeomDone();
  return fNDOFs;
}

//_________________________________________________________
Int_t AliAlgDet::AssignDOFs()
{
  // assign DOFs IDs, parameters
  //
  int gloCount0(fAlgSteer->GetNDOFs()), gloCount(fAlgSteer->GetNDOFs());
  Float_t* pars = fAlgSteer->GetGloParVal(); 
  Float_t* errs = fAlgSteer->GetGloParErr(); 
  Int_t*   labs = fAlgSteer->GetGloParLab();
  //
  // assign calibration DOFs
  fFirstParGloID = gloCount;
  fParVals = pars + gloCount;
  fParErrs = errs + gloCount;
  fParLabs = labs + gloCount;
  for (int icl=0;icl<fNCalibDOF;icl++) {
    fParLabs[icl] = (GetDetLabel() + 10000)*100 + icl;
    gloCount++;
  }
  //
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) {
    AliAlgVol *vol = GetVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->GetParent()) vol->AssignDOFs(gloCount,pars,errs,labs);
  }
  //

  if (fNDOFs != gloCount-gloCount0) AliFatalF("Mismatch between declared %d and initialized %d DOFs for %s",
					      fNDOFs,gloCount-gloCount0,GetName());
  
  return fNDOFs;
}

//_________________________________________________________
void AliAlgDet::InitDOFs()
{
  // initialize free parameters
  if (GetInitDOFsDone()) AliFatalF("Something is wrong, DOFs are already initialized for %s",GetName());
  //
  // process calibration DOFs
  for (int i=0;i<fNCalibDOF;i++) if (fParErrs[i]<-9999 && IsZeroAbs(fParVals[i])) FixDOF(i);
  //
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) GetVolume(iv)->InitDOFs();
  //
  CalcFree(kTRUE);
  //
  SetInitDOFsDone();
  return;
}

//_________________________________________________________
Int_t AliAlgDet::VolID2SID(Int_t vid) const 
{
  // find SID corresponding to VolID
  int mn(0),mx(fNSensors-1);
  while (mx>=mn) {
    int md( (mx+mn)>>1 ), vids(GetSensor(md)->GetVolID());
    if (vid<vids)      mx = md-1;
    else if (vid>vids) mn = md+1;
    else return md;
  }
  return -1;
}

//____________________________________________
void AliAlgDet::Print(const Option_t *opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("\nDetector:%5s %5d volumes %5d sensors {VolID: %5d-%5d} Def.Sys.Err: %.4e %.4e | Stat:%d\n",
	 GetName(),GetNVolumes(),GetNSensors(),GetVolIDMin(),
	 GetVolIDMax(),fAddError[0],fAddError[1],fNProcPoints);
  //
  printf("Errors assignment: ");
  if (fUseErrorParam) printf("param %d\n",fUseErrorParam);
  else printf("from TrackPoints\n");
  //
  printf("Allowed    in Collisions: %7s | Cosmic: %7s\n",
	 IsDisabled(kColl)   ? "  NO ":" YES ",IsDisabled(kCosm)   ? "  NO ":" YES ");
  //
  printf("Obligatory in Collisions: %7s | Cosmic: %7s\n",
	 IsObligatory(kColl) ? " YES ":"  NO ",IsObligatory(kCosm) ? " YES ":"  NO ");
  //
  printf("Sel. flags in Collisions: 0x%05lx | Cosmic: 0x%05lx\n",
	 fTrackFlagSel[kColl],fTrackFlagSel[kCosm]);
  //
  printf("Min.points in Collisions: %7d | Cosmic: %7d\n",
	 fNPointsSel[kColl],fNPointsSel[kCosm]);
  //
  if (!(IsDisabledColl()&&IsDisabledCosm()) && opts.Contains("long")) 
    for (int iv=0;iv<GetNVolumes();iv++) GetVolume(iv)->Print(opt);
  //
  for (int i=0;i<GetNCalibDOFs();i++) {
    printf("CalibDOF%2d: %-20s\t%e\n",i,GetCalibDOFName(i),GetCalibDOFValWithCal(i));
  }

}

//____________________________________________
void AliAlgDet::SetDetID(UInt_t tp)
{
  // assign type
  if (tp>=AliAlgSteer::kNDetectors) AliFatalF("Detector typeID %d exceeds allowed range %d:%d",
					      tp,0,AliAlgSteer::kNDetectors-1);
  SetUniqueID(tp);
}

//____________________________________________
void AliAlgDet::SetAddError(double sigy, double sigz)
{
  // add syst error to all sensors
  AliInfoF("Adding sys.error %.4e %.4e to all sensors",sigy,sigz);
  fAddError[0] = sigy;
  fAddError[1] = sigz;
  for (int isn=GetNSensors();isn--;) GetSensor(isn)->SetAddError(sigy,sigz);
  //
}

//____________________________________________
void AliAlgDet::SetUseErrorParam(Int_t v) 
{
  // set type of points error parameterization
  AliFatal("UpdatePointByTrackInfo is not implemented for this detector");
  //  
}

//____________________________________________
void AliAlgDet::UpdatePointByTrackInfo(AliAlgPoint* pnt, const AliExternalTrackParam* t) const
{
  // update point using specific error parameterization
  AliFatal("If needed, this method has to be implemented for specific detector");
}

//____________________________________________
void AliAlgDet::SetObligatory(Int_t tp,Bool_t v)
{
  // mark detector presence obligatory in the track
  fObligatory[tp] = v;
  fAlgSteer->SetObligatoryDetector(GetDetID(),tp,v);
}

//______________________________________________________
void AliAlgDet::WritePedeInfo(FILE* parOut, const Option_t *opt) const
{
  // contribute to params and constraints template files for PEDE
  fprintf(parOut,"\n!!\t\tDetector:\t%s\tNDOFs: %d\n",GetName(),GetNDOFs());
  //
  // parameters
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) {  // call for root level volumes, they will take care of their children
    AliAlgVol *vol = GetVolume(iv);
    if (!vol->GetParent()) vol->WritePedeInfo(parOut,opt);
  }
  //
}

//______________________________________________________
void AliAlgDet::WriteCalibrationResults() const
{
  // store calibration results
  WriteAlignmentResults();
  // 
  // eventually we may write other calibrations
}

//______________________________________________________
void AliAlgDet::WriteAlignmentResults() const
{
  // store updated alignment
  TClonesArray* arr = new TClonesArray("AliAlignObjParams",10);
  //
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) {
    AliAlgVol *vol = GetVolume(iv);
    // call only for top level objects, they will take care of children
    if (!vol->GetParent()) vol->CreateAlignmentObjects(arr);
  }
  //
  AliCDBManager* man = AliCDBManager::Instance();
  AliCDBMetaData* md = new AliCDBMetaData();
  md->SetResponsible(fAlgSteer->GetOutCDBResponsible());
  md->SetComment(fAlgSteer->GetOutCDBResponsible());
  //
  AliCDBId id(Form("%s/Align/Data",GetName()),fAlgSteer->GetOutCDBRunMin(),fAlgSteer->GetOutCDBRunMax());
  man->Put(arr,id,md); 
  //
  delete arr;
}

//______________________________________________________
Bool_t AliAlgDet::OwnsDOFID(Int_t id) const
{
  // check if DOF ID belongs to this detector
  for (int iv=GetNVolumes();iv--;) {
    AliAlgVol* vol = GetVolume(iv); // check only top level volumes
    if (!vol->GetParent() && vol->OwnsDOFID(id)) return kTRUE;
  }
  // calibration DOF?
  if (id>=fFirstParGloID && id<fFirstParGloID+fNCalibDOF) return kTRUE;
  //
  return kFALSE;
}

//______________________________________________________
AliAlgVol* AliAlgDet::GetVolOfDOFID(Int_t id) const
{
  // gets volume owning this DOF ID
  for (int iv=GetNVolumes();iv--;) {
    AliAlgVol* vol = GetVolume(iv);
    if (vol->GetParent()) continue; // check only top level volumes
    if ( (vol=vol->GetVolOfDOFID(id)) ) return vol;
  }
  return 0;
}

//______________________________________________________
void AliAlgDet::Terminate()
{
  // called at the end of processing
  //  if (IsDisabled()) return;
  int nvol = GetNVolumes();
  fNProcPoints = 0;
  AliAlgDOFStat* st = fAlgSteer->GetDOFStat();
  for (int iv=0;iv<nvol;iv++) {
    AliAlgVol *vol = GetVolume(iv);
    // call init for root level volumes, they will take care of their children
    if (!vol->GetParent()) fNProcPoints += vol->FinalizeStat(st);
  }
  FillDOFStat(st); // fill stat for calib dofs
}

//________________________________________
void AliAlgDet::AddAutoConstraints() const
{
  // adds automatic constraints
  int nvol = GetNVolumes();
  for (int iv=0;iv<nvol;iv++) {  // call for root level volumes, they will take care of their children
    AliAlgVol *vol = GetVolume(iv);
    if (!vol->GetParent()) vol->AddAutoConstraints((TObjArray*)fAlgSteer->GetConstraints());
  }
}

//________________________________________
void AliAlgDet::FixNonSensors()
{
  // fix all non-sensor volumes
  for (int i=GetNVolumes();i--;) {
    AliAlgVol *vol = GetVolume(i);
    if (vol->IsSensor()) continue;
    vol->SetFreeDOFPattern(0);
    vol->SetChildrenConstrainPattern(0);
  }
}

//________________________________________
int AliAlgDet::SelectVolumes(TObjArray* arr, int lev, const char* match)
{
  // select volumes matching to pattern and/or hierarchy level
  //
  if (!arr) return 0;
  int nadd = 0;
  TString mts=match, syms;
  for (int i=GetNVolumes();i--;) {
    AliAlgVol *vol = GetVolume(i);
    if (lev>=0 && vol->CountParents()!=lev) continue; // wrong level
    if (!mts.IsNull() && !(syms=vol->GetSymName()).Contains(mts)) continue; //wrong name
    arr->AddLast(vol);
    nadd++;
  }
  //
  return nadd;
}

//________________________________________
void AliAlgDet::SetFreeDOFPattern(UInt_t pat, int lev,const char* match)
{
  // set free DOFs to volumes matching either to hierarchy level or
  // whose name contains match
  //
  TString mts=match, syms;
  for (int i=GetNVolumes();i--;) {
    AliAlgVol *vol = GetVolume(i);
    if (lev>=0 && vol->CountParents()!=lev) continue; // wrong level
    if (!mts.IsNull() && !(syms=vol->GetSymName()).Contains(mts)) continue; //wrong name
    vol->SetFreeDOFPattern(pat);
  }
  //
}

//________________________________________
void AliAlgDet::SetDOFCondition(int dof, float condErr ,int lev,const char* match)
{
  // set condition for DOF of volumes matching either to hierarchy level or
  // whose name contains match
  //
  TString mts=match, syms;
  for (int i=GetNVolumes();i--;) {
    AliAlgVol *vol = GetVolume(i);
    if (lev>=0 && vol->CountParents()!=lev) continue; // wrong level
    if (!mts.IsNull() && !(syms=vol->GetSymName()).Contains(mts)) continue; //wrong name
    if (dof>=vol->GetNDOFs()) continue;
    vol->SetParErr(dof, condErr);
    if (condErr>=0 && !vol->IsFreeDOF(dof)) vol->SetFreeDOF(dof);
    //if (condErr<0  && vol->IsFreeDOF(dof)) vol->FixDOF(dof);
  }
  //
}

//________________________________________
void AliAlgDet::ConstrainOrphans(const double* sigma, const char* match)
{
  // additional constraint on volumes w/o parents (optionally containing "match" in symname)
  // sigma<0 : dof is not contrained
  // sigma=0 : dof constrained exactly (Lagrange multiplier)
  // sigma>0 : dof constrained by gaussian constraint
  //
  TString mts=match, syms;
  AliAlgConstraint* constr = new AliAlgConstraint();
  for (int i=0;i<AliAlgVol::kNDOFGeom;i++) {
    if (sigma[i]>=0) constr->ConstrainDOF(i);
    else             constr->UnConstrainDOF(i);
    constr->SetSigma(i,sigma[i]);
  }
  for (int i=GetNVolumes();i--;) {
    AliAlgVol *vol = GetVolume(i);
    if (vol->GetParent()) continue; // wrong level
    if (!mts.IsNull() && !(syms=vol->GetSymName()).Contains(mts)) continue; //wrong name
    constr->AddChild(vol);
  }
  //
  if (!constr->GetNChildren()) {
    AliInfoF("No volume passed filter %s",match);
    delete constr;
  }
  else ((TObjArray*)fAlgSteer->GetConstraints())->Add(constr);
  //
}

//________________________________________
void AliAlgDet::SetFreeDOF(Int_t dof) 
{
  // set detector free dof
  if (dof>=kNMaxKalibDOF) {AliFatalF("Detector CalibDOFs limited to %d, requested %d",kNMaxKalibDOF,dof);}
  fCalibDOF |= 0x1<<dof; 
  CalcFree();
}

//________________________________________
void AliAlgDet::FixDOF(Int_t dof)
{
  // fix detector dof
  if (dof>=kNMaxKalibDOF) {AliFatalF("Detector CalibDOFs limited to %d, requested %d",kNMaxKalibDOF,dof);}
  fCalibDOF &=~(0x1<<dof); 
  CalcFree();
}

//__________________________________________________________________
Bool_t AliAlgDet::IsCondDOF(Int_t i) const
{
  // is DOF free and conditioned?
  return (!IsZeroAbs(GetParVal(i)) || !IsZeroAbs(GetParErr(i)));
}

//__________________________________________________________________
void AliAlgDet::CalcFree(Bool_t condFix)
{
  // calculate free calib dofs. If condFix==true, condition parameter a la pede, i.e. error < 0
  fNCalibDOFFree = 0;
  for (int i=0;i<fNCalibDOF;i++) {
    if (!IsFreeDOF(i)) {
      if (condFix) SetParErr(i,-999);
      continue;
    }
    fNCalibDOFFree++;
  }
  //
}

//______________________________________________________
void AliAlgDet::FillDOFStat(AliAlgDOFStat* st) const
{
  // fill statistics info hist
  if (!st) return;
  int ndf = GetNCalibDOFs();
  int dof0 = GetFirstParGloID();
  int stat = GetNProcessedPoints();
  for (int idf=0;idf<ndf;idf++) {
    int dof = idf+dof0;
    st->AddStat(dof,stat);
  }
  //
}

//______________________________________________________
void AliAlgDet::WriteSensorPositions(const char* outFName)
{
  // create tree with sensors ideal, ref and reco positions
  int ns = GetNSensors();
  double loc[3]={0};
  // ------- local container type for dumping sensor positions ------
  typedef struct {
    int    volID;  // volume id
    double pId[3]; // ideal
    double pRf[3]; // reference
    double pRc[3]; // reco-time
  } snpos_t;
  snpos_t spos; // 
  TFile* fl = TFile::Open(outFName,"recreate");
  TTree* tr = new TTree("snpos",Form("sensor poisitions for %s",GetName()));
  tr->Branch("volID",&spos.volID,"volID/I");
  tr->Branch("pId",&spos.pId,"pId[3]/D");
  tr->Branch("pRf",&spos.pRf,"pRf[3]/D");
  tr->Branch("pRc",&spos.pRc,"pRc[3]/D");
  //
  for (int isn=0;isn<ns;isn++) {
    AliAlgSens* sens = GetSensor(isn);
    spos.volID = sens->GetVolID();
    sens->GetMatrixL2GIdeal().LocalToMaster(loc,spos.pId);
    sens->GetMatrixL2G().LocalToMaster(loc,spos.pRf);
    sens->GetMatrixL2GReco().LocalToMaster(loc,spos.pRc);
    tr->Fill();
  }
  tr->Write();
  delete tr;
  fl->Close();
  delete fl;
}
