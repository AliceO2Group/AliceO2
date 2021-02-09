// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgVol.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class of alignable volume

/*
  Alignment formalism:

  Vector l in the local frame of the volume_j (assuming hierarchy of nested volumes 0...J
  from most coarse to the end volume) is transformed to master frame vector
  g = G_j*l_j
  Matrix G_j is Local2Global matrix (L2G in the code). If the volume has a parent
  volume j-1, the global vector g can be transformed to the local volume of j-1 as
  l_{j-1} = G^-1_{j-1}* g
  Hence, the transormation from volume j to j-1 is
  l_{j-1} = G^-1_{j-1}*G_j l_j = R_j*l_j

  The alignment corrections in general can be defined either as a

  1) local delta:   l'_j = delta_j * l_j
  hence g'  = G_j * delta_j = G'_j*l_j
  or as
  2) global Delta:  g' = Delta_j * G_j * l_j = G'_j*l_j

  Hence Delta and delta are linked as
  Delta_j = G_j delta_j G^-1_j
  delta_j = G^-1_j Delta_j G_j

  In case the whole chain of nested volumes is aligned, the corrections pile-up as:

  G_0*delta_0 ... G^-1_{j-2}*G_{j-1}*delta_{j-1}*G^-1_{j-1}*G_j*delta_j =
  Delta_0 * Delta_{1} ... Delta_{j-1}*Delta_{j}... * G_j

  -> Delta_j = G'_{j-1} * G^-1_{j-1} * G_j * G'^-1_j
  where G and G' are modified and original L2G matrices


  From this by induction one gets relation between local and global deltas:

  Delta_j = Z_j * delta_j * Z^-1_j

  where Z_j = [ Prod_{k=0}^{j-1} (G_k * delta_k * G^-1_k) ] * G_j

  By convention, aliroot aligment framework stores global Deltas !

  In case the geometry was already prealigned by PDelta_j matrices, the result
  of the new incremental alignment Delta_j must be combined with PDelta_j to
  resulting matrix TDelta_j before writing new alignment object.

  Derivation: if G_j and IG_j are final and ideal L2G matrices for level j, then

  G_j = TDelta_j * TDelta_{j-1} ... TDelta_0 * IG_j
  =     (Delta_j * Delta_{j-1} ... Delta_0)  * (PDelta_j * PDelta_{j-1} ... PDelta_0) * IG_j

  Hence:
  TDelta_j = [Prod_{i=j}^0 Delta_i ] * [Prod_{k=j}^0 PDelta_k ] * [Prod_{l=0}^{j-1} TDelta_l]

  By induction we get combination rule:

  TDelta_j = Delta_j * X_{j-1} * PDelta_j * X^-1_{j-1}

  where X_i = Delta_i * Delta_{i-1} ... Delta_0

  ---------------------------------

  This alignment framework internally allows to find geometry corrections either in the
  volume LOCAL frame or in its TRACKING frame. The latter is defined for sensors as
  lab frame, rotated by the angle alpha in such a way that the X axis is normal to the
  sensor plane (note, that for ITS the rotated X axis origin is also moved to the sensor)
  For the non-sensor volumes the TRACKING frame is defined by rotation of the lab frame
  with the alpha angle = average angle of centers of its children, seen from the origin.

  The TRACKING and IDEAL LOCAL (before misalignment) frames are related by the
  tracking-to-local matrix (T2L in the code), i.e. the vectors in local and tracking frames
  are related as
  l = T2L * t

  The alignment can be done using both frames for different volumes of the same geometry
  branch.
  The alignments deltas in local and tracking frames are related as:

  l' = T2L * delta_t * t
  l' = delta_l * T2L * t
  -> delta_l = T2L * delta_t * T2L^-1

 */

#include "AliAlgVol.h"
#include "AliAlgDOFStat.h"
#include "AliAlgConstraint.h"
#include "AliAlignObjParams.h"
#include "AliGeomManager.h"
#include "AliAlgAux.h"
#include "AliLog.h"
#include <TString.h>
#include <TClonesArray.h>
#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TH1.h>
#include <TAxis.h>
#include <stdio.h>

ClassImp(o2::align::AliAlgVol)

  using namespace TMath;
using namespace o2::align::AliAlgAux;

namespace o2
{
namespace align
{

const char* AliAlgVol::fgkFrameName[AliAlgVol::kNVarFrames] = {"LOC", "TRA"};
//
UInt_t AliAlgVol::fgDefGeomFree =
  kDOFBitTX | kDOFBitTY | kDOFBitTZ | kDOFBitPS | kDOFBitTH | kDOFBitPH;
//
const char* AliAlgVol::fgkDOFName[AliAlgVol::kNDOFGeom] = {"TX", "TY", "TZ", "PSI", "THT", "PHI"};

//_________________________________________________________
AliAlgVol::AliAlgVol(const char* symname, int iid) : TNamed(symname, ""), fVarFrame(kTRA), fIntID(iid), fX(0), fAlp(0), fNDOFs(0), fDOF(0), fNDOFGeomFree(0), fNDOFFree(0), fConstrChild(kDefChildConstr)
                                                     //
                                                     ,
                                                     fParent(0),
                                                     fChildren(0)
                                                     //
                                                     ,
                                                     fNProcPoints(0),
                                                     fFirstParGloID(-1),
                                                     fParVals(0),
                                                     fParErrs(0),
                                                     fParLabs(0)
                                                     //
                                                     ,
                                                     fMatL2GReco(),
                                                     fMatL2G(),
                                                     fMatL2GIdeal(),
                                                     fMatT2L(),
                                                     fMatDeltaRefGlo()
{
  // def c-tor
  SetVolID(0);   // volumes have no VID, unless it is sensor
  if (symname) { // real volumes have at least geometric degrees of freedom
    SetNDOFs(kNDOFGeom);
  }
  SetFreeDOFPattern(fgDefGeomFree);
}

//_________________________________________________________
AliAlgVol::~AliAlgVol()
{
  // d-tor
  delete fChildren;
}

//_________________________________________________________
void AliAlgVol::Delta2Matrix(TGeoHMatrix& deltaM, const Double_t* delta) const
{
  // prepare delta matrix for the volume from its
  // local delta vector (AliAlignObj convension): dx,dy,dz,,theta,psi,phi
  const double *tr = &delta[0], *rt = &delta[3]; // translation(cm) and rotation(degree)
  AliAlignObjParams tempAlignObj;
  tempAlignObj.SetRotation(rt[0], rt[1], rt[2]);
  tempAlignObj.SetTranslation(tr[0], tr[1], tr[2]);
  tempAlignObj.GetMatrix(deltaM);
  //
}

//__________________________________________________________________
void AliAlgVol::GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const Double_t* delta) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of LOCAL frame:
  // tra' = tau*tra = tau*T2L^-1*loc = T2L^-1*loc' = T2L^-1*delta*loc
  // tau = T2L^-1*delta*T2L
  Delta2Matrix(matMod, delta);
  matMod.Multiply(&GetMatrixT2L());
  const TGeoHMatrix& t2li = GetMatrixT2L().Inverse();
  matMod.MultiplyLeft(&t2li);
}

//__________________________________________________________________
void AliAlgVol::GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const Double_t* delta, const TGeoHMatrix& relMat) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of LOCAL frame of its PARENT;
  // The relMat is matrix for transformation from child to parent frame: LOC = relMat*loc
  //
  // tra' = tau*tra = tau*T2L^-1*loc = T2L^-1*loc' = T2L^-1*relMat^-1*Delta*relMat*loc
  // tau = (relMat*T2L)^-1*Delta*(relMat*T2L)
  Delta2Matrix(matMod, delta);
  TGeoHMatrix tmp = relMat;
  tmp *= GetMatrixT2L();
  matMod.Multiply(&tmp);
  const TGeoHMatrix& tmpi = tmp.Inverse();
  matMod.MultiplyLeft(&tmpi);
}

//__________________________________________________________________
void AliAlgVol::GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const Double_t* delta) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of the same TRACKING frame:
  // tra' = tau*tra
  Delta2Matrix(matMod, delta);
}

//__________________________________________________________________
void AliAlgVol::GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const Double_t* delta, const TGeoHMatrix& relMat) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of TRACKING frame of its PARENT;
  // The relMat is matrix for transformation from child to parent frame: TRA = relMat*tra
  // (see DPosTraDParGeomTRA)
  //
  // tra' = tau*tra = tau*relMat^-1*TRA = relMat^-1*TAU*TRA = relMat^-1*TAU*relMat*tra
  // tau = relMat^-1*TAU*relMat
  Delta2Matrix(matMod, delta); // TAU
  matMod.Multiply(&relMat);
  const TGeoHMatrix& reli = relMat.Inverse();
  matMod.MultiplyLeft(&reli);
}

//_________________________________________________________
Int_t AliAlgVol::CountParents() const
{
  // count parents in the chain
  int cnt = 0;
  const AliAlgVol* p = this;
  while ((p = p->GetParent()))
    cnt++;
  return cnt;
}

//____________________________________________
void AliAlgVol::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("Lev:%2d IntID:%7d %s | %2d nodes | Effective X:%8.4f Alp:%+.4f | Used Points: %d\n",
         CountParents(), GetInternalID(), GetSymName(), GetNChildren(), fX, fAlp, fNProcPoints);
  printf("     DOFs: Tot: %d (offs: %5d) Free: %d  Geom: %d {", fNDOFs, fFirstParGloID, fNDOFFree, fNDOFGeomFree);
  for (int i = 0; i < kNDOFGeom; i++)
    printf("%d", IsFreeDOF(i) ? 1 : 0);
  printf("} in %s frame.", fgkFrameName[fVarFrame]);
  if (GetNChildren()) {
    printf(" Child.Constr: {");
    for (int i = 0; i < kNDOFGeom; i++)
      printf("%d", IsChildrenDOFConstrained(i) ? 1 : 0);
    printf("}");
  }
  if (GetExcludeFromParentConstraint())
    printf(" Excl.from parent constr.");
  printf("\n");
  //
  if (opts.Contains("par") && fParVals) {
    printf("     Lb: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%10d  ", GetParLab(i));
    printf("\n");
    printf("     Vl: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%+9.3e  ", GetParVal(i));
    printf("\n");
    printf("     Er: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%+9.3e  ", GetParErr(i));
    printf("\n");
  }

  if (opts.Contains("mat")) { // print matrices
    printf("L2G ideal   : ");
    GetMatrixL2GIdeal().Print();
    printf("L2G misalign: ");
    GetMatrixL2G().Print();
    printf("L2G RecoTime: ");
    GetMatrixL2GReco().Print();
    printf("T2L (fake)  : ");
    GetMatrixT2L().Print();
  }
  //
}

//____________________________________________
void AliAlgVol::PrepareMatrixL2G(Bool_t reco)
{
  // extract from geometry L2G matrix, depending on reco flag, set it as a reco-time
  // or current alignment matrix
  const char* path = GetSymName();
  if (gGeoManager->GetAlignableEntry(path)) {
    const TGeoHMatrix* l2g = AliGeomManager::GetMatrix(path);
    if (!l2g)
      AliFatalF("Failed to find L2G matrix for alignable %s", path);
    reco ? SetMatrixL2GReco(*l2g) : SetMatrixL2G(*l2g);
  } else { // extract from path
    if (!gGeoManager->CheckPath(path))
      AliFatalF("Volume path %s not valid!", path);
    TGeoPhysicalNode* node = (TGeoPhysicalNode*)gGeoManager->GetListOfPhysicalNodes()->FindObject(path);
    TGeoHMatrix l2g;
    if (!node) {
      AliWarningF("Attention: volume %s was not misaligned, extracting original matrix", path);
      if (!AliGeomManager::GetOrigGlobalMatrix(path, l2g)) {
        AliFatalF("Failed to find ideal L2G matrix for %s", path);
      }
    } else {
      l2g = *node->GetMatrix();
    }
    reco ? SetMatrixL2GReco(l2g) : SetMatrixL2G(l2g);
  }
  //
}

//____________________________________________
void AliAlgVol::PrepareMatrixL2GIdeal()
{
  // extract from geometry ideal L2G matrix
  TGeoHMatrix mtmp;
  if (!AliGeomManager::GetOrigGlobalMatrix(GetSymName(), mtmp))
    AliFatalF("Failed to find ideal L2G matrix for %s", GetSymName());
  SetMatrixL2GIdeal(mtmp);
  //
}

//____________________________________________
void AliAlgVol::PrepareMatrixT2L()
{
  // for non-sensors we define the fake tracking frame with the alpha angle being
  // the average angle of centers of its children
  //
  double tot[3] = {0, 0, 0}, loc[3] = {0, 0, 0}, glo[3];
  int nch = GetNChildren();
  for (int ich = nch; ich--;) {
    AliAlgVol* vol = GetChild(ich);
    vol->GetMatrixL2GIdeal().LocalToMaster(loc, glo);
    for (int j = 3; j--;)
      tot[j] += glo[j];
  }
  if (nch)
    for (int j = 3; j--;)
      tot[j] /= nch;
  //
  fAlp = TMath::ATan2(tot[1], tot[0]);
  AliAlgAux::BringToPiPM(fAlp);
  //
  fX = TMath::Sqrt(tot[0] * tot[0] + tot[1] * tot[1]);
  //
  // 1st create Tracking to Global matrix
  fMatT2L.Clear();
  fMatT2L.SetDx(fX);
  fMatT2L.RotateZ(fAlp * RadToDeg());
  // then convert it to Tracking to Local  matrix
  const TGeoHMatrix& l2gi = GetMatrixL2GIdeal().Inverse();
  fMatT2L.MultiplyLeft(&l2gi);
  //
}

//____________________________________________
void AliAlgVol::SetMatrixT2L(const TGeoHMatrix& m)
{
  // set the T2L matrix and define tracking frame
  // Note that this method is used for the externally set matrices
  // (in case of sensors). For other volumes the tracking frame and matrix
  // is defined in the PrepareMatrixT2L method
  fMatT2L = m;
  SetTrackingFrame();
}

//__________________________________________________________________
void AliAlgVol::SetTrackingFrame()
{
  // Define tracking frame of the sensor
  // This method should be implemented for sensors, which receive the T2L
  // matrix from the geometry
  AliErrorF("Volume %s was supposed to implement its own method", GetName());
}

//__________________________________________________________________
void AliAlgVol::AssignDOFs(Int_t& cntDOFs, Float_t* pars, Float_t* errs, Int_t* labs)
{
  // Assigns offset of the DOFS of this volume in global array of DOFs, attaches arrays to volumes
  //
  fParVals = pars + cntDOFs;
  fParErrs = errs + cntDOFs;
  fParLabs = labs + cntDOFs;
  SetFirstParGloID(cntDOFs);
  for (int i = 0; i < fNDOFs; i++)
    fParLabs[i] = GetInternalID() * 100 + i;
  cntDOFs += fNDOFs; // increment total DOFs count
  //
  int nch = GetNChildren(); // go over childs
  for (int ich = 0; ich < nch; ich++)
    GetChild(ich)->AssignDOFs(cntDOFs, pars, errs, labs);
  //
  return;
}

//__________________________________________________________________
void AliAlgVol::InitDOFs()
{
  // initialize degrees of freedom
  //
  // Do we need this strict condition?
  if (GetInitDOFsDone())
    AliFatalF("Something is wrong, DOFs are already initialized for %s", GetName());
  for (int i = 0; i < fNDOFs; i++)
    if (fParErrs[i] < -9999 && IsZeroAbs(fParVals[i]))
      FixDOF(i);
  CalcFree(kTRUE);
  //
  SetInitDOFsDone();
  //
}

//__________________________________________________________________
void AliAlgVol::CalcFree(Bool_t condFix)
{
  // calculate free dofs. If condFix==true, condition parameter a la pede, i.e. error < 0
  fNDOFFree = fNDOFGeomFree = 0;
  for (int i = 0; i < fNDOFs; i++) {
    if (!IsFreeDOF(i)) {
      if (condFix)
        SetParErr(i, -999);
      continue;
    }
    fNDOFFree++;
    if (i < kNDOFGeom)
      fNDOFGeomFree++;
  }
  //
}

//__________________________________________________________________
void AliAlgVol::SetNDOFs(Int_t n)
{
  // book global degrees of freedom
  if (n < kNDOFGeom)
    n = kNDOFGeom;
  fNDOFs = n;
}

//__________________________________________________________________
void AliAlgVol::AddChild(AliAlgVol* ch)
{
  // add child volume
  if (!fChildren) {
    fChildren = new TObjArray();
    fChildren->SetOwner(kFALSE);
  }
  fChildren->AddLast(ch);
}

//__________________________________________________________________
void AliAlgVol::SetParVals(Int_t npar, Double_t* vl, Double_t* er)
{
  // set parameters
  if (npar > fNDOFs)
    AliFatalF("Volume %s has %d dofs", GetName(), fNDOFs);
  for (int i = 0; i < npar; i++) {
    fParVals[i] = vl[i];
    fParErrs[i] = er ? er[i] : 0;
  }
}

//__________________________________________________________________
Bool_t AliAlgVol::IsCondDOF(Int_t i) const
{
  // is DOF free and conditioned?
  return (!IsZeroAbs(GetParVal(i)) || !IsZeroAbs(GetParErr(i)));
}

//______________________________________________________
Int_t AliAlgVol::FinalizeStat(AliAlgDOFStat* st)
{
  // finalize statistics on processed points
  fNProcPoints = 0;
  for (int ich = GetNChildren(); ich--;) {
    AliAlgVol* child = GetChild(ich);
    fNProcPoints += child->FinalizeStat(st);
  }
  if (st)
    FillDOFStat(st);
  return fNProcPoints;
}

//______________________________________________________
void AliAlgVol::WritePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params template file for PEDE
  enum { kOff,
         kOn,
         kOnOn };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyParam = "parameter";
  TString opts = opt;
  opts.ToLower();
  Bool_t showDef = opts.Contains("d"); // show free DOF even if not preconditioned
  Bool_t showFix = opts.Contains("f"); // show DOF even if fixed
  Bool_t showNam = opts.Contains("n"); // show volume name even if no nothing else is printable
  //
  // is there something to print ?
  int nCond(0), nFix(0), nDef(0);
  for (int i = 0; i < fNDOFs; i++) {
    if (!IsFreeDOF(i))
      nFix++;
    if (IsCondDOF(i))
      nCond++;
    if (!IsCondDOF(i) && IsFreeDOF(i))
      nDef++;
  }
  //
  int cmt = nCond > 0 || nFix > 0 ? kOff : kOn; // do we comment the "parameter" keyword for this volume
  if (!nFix)
    showFix = kFALSE;
  if (!nDef)
    showDef = kFALSE;
  //
  if (nCond || showDef || showFix || showNam)
    fprintf(parOut, "%s%s %s\t\tDOF/Free: %d/%d (%s) %s\n", comment[cmt], kKeyParam, comment[kOnOn],
            GetNDOFs(), GetNDOFFree(), fgkFrameName[fVarFrame], GetName());
  //
  if (nCond || showDef || showFix) {
    for (int i = 0; i < fNDOFs; i++) {
      cmt = kOn;
      if (IsCondDOF(i) || !IsFreeDOF(i))
        cmt = kOff; // free-conditioned : MUST print
      else if (!IsFreeDOF(i)) {
        if (!showFix)
          continue;
      } // Fixed: print commented if asked
      else if (!showDef)
        continue; // free-unconditioned: print commented if asked
      //
      fprintf(parOut, "%s %9d %+e %+e\t%s %s p%d\n", comment[cmt], GetParLab(i),
              GetParVal(i), GetParErr(i), comment[kOnOn], IsFreeDOF(i) ? "  " : "FX", i);
    }
    fprintf(parOut, "\n");
  }
  // children volume
  int nch = GetNChildren();
  //
  for (int ich = 0; ich < nch; ich++)
    GetChild(ich)->WritePedeInfo(parOut, opt);
  //
}

//_________________________________________________________________
void AliAlgVol::CreateGloDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create global matrix deltaM from fParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  //
  CreateLocDeltaMatrix(deltaM);
  const TGeoHMatrix& l2g = GetMatrixL2G();
  const TGeoHMatrix& l2gi = l2g.Inverse();
  deltaM.Multiply(&l2gi);
  deltaM.MultiplyLeft(&l2g);
  //
}

/*
//_________________________________________________________________
void AliAlgVol::CreateGloDeltaMatrix(TGeoHMatrix &deltaM) const
{
  // Create global matrix deltaM from fParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  //
  // deltaM = Z * deltaL * Z^-1
  // where deltaL is local correction matrix and Z is matrix defined as
  // Z = [ Prod_{k=0}^{j-1} G_k * deltaL_k * G^-1_k ] * G_j
  // with j=being the level of the volume in the hierarchy
  //
  CreateLocDeltaMatrix(deltaM);
  TGeoHMatrix zMat = GetMatrixL2G();
  const AliAlgVol* par = this;
  while( (par=par->GetParent()) ) {
    TGeoHMatrix locP;
    par->CreateLocDeltaMatrix(locP);
    locP.MultiplyLeft( &par->GetMatrixL2G() );
    locP.Multiply( &par->GetMatrixL2G().Inverse() );
    zMat.MultiplyLeft( &locP );
  }
  deltaM.MultiplyLeft( &zMat );
  deltaM.Multiply( &zMat.Inverse() );
  //
}
*/

//_________________________________________________________________
void AliAlgVol::CreatePreGloDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create prealignment global matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = Delta_j * Delta_{j-1} ... Delta_0 * GO_j
  // where Delta_j is global prealignment matrix for volume at level j
  // we get by induction
  // Delta_j = G_j * GO^-1_j * GO_{j-1} * G^-1_{j-1}
  //
  deltaM = GetMatrixL2G();
  deltaM *= GetMatrixL2GIdeal().Inverse();
  const AliAlgVol* par = GetParent();
  if (par) {
    deltaM *= par->GetMatrixL2GIdeal();
    deltaM *= par->GetMatrixL2G().Inverse();
  }
  //
}

/*
  // this is an alternative lengthy way !
//_________________________________________________________________
void AliAlgVol::CreatePreGloDeltaMatrix(TGeoHMatrix &deltaM) const
{
  // Create prealignment global matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = Delta_j * Delta_{j-1} ... Delta_0 * GO_j
  // where Delta_j is global prealignment matrix for volume at level j
  // we get by induction
  // Delta_j = G_j * GO^-1_j * GO_{j-1} * G^-1_{j-1}
  //
  CreatePreLocDeltaMatrix(deltaM);
  TGeoHMatrix zMat = GetMatrixL2GIdeal();
  const AliAlgVol* par = this;
  while( (par=par->GetParent()) ) {
    TGeoHMatrix locP;
    par->CreatePreLocDeltaMatrix(locP);
    locP.MultiplyLeft( &par->GetMatrixL2GIdeal() );
    locP.Multiply( &par->GetMatrixL2GIdeal().Inverse() );
    zMat.MultiplyLeft( &locP );
  }
  deltaM.MultiplyLeft( &zMat );
  deltaM.Multiply( &zMat.Inverse() );
  //
}
*/

//_________________________________________________________________
void AliAlgVol::CreatePreLocDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create prealignment local matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = GO_0 * delta_0 * GO^-1_0 * GO_1 * delta_1 ... GO^-1_{j-1}*GO_{j}*delta_j
  // where delta_j is local prealignment matrix for volume at level j
  // we get by induction
  // delta_j = GO^-1_j * GO_{j-1} * G^-1_{j-1} * G^_{j}
  //
  const AliAlgVol* par = GetParent();
  deltaM = GetMatrixL2GIdeal().Inverse();
  if (par) {
    deltaM *= par->GetMatrixL2GIdeal();
    deltaM *= par->GetMatrixL2G().Inverse();
  }
  deltaM *= GetMatrixL2G();
  //
}

//_________________________________________________________________
void AliAlgVol::CreateLocDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create local matrix deltaM from fParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  double corr[kNDOFGeom];
  for (int i = kNDOFGeom; i--;)
    corr[i] = fParVals[i]; // we need doubles
  Delta2Matrix(deltaM, corr);
  if (IsFrameTRA()) { // we need corrections in local frame!
    // l' = T2L * delta_t * t = T2L * delta_t * T2L^-1 * l = delta_l * l
    // -> delta_l = T2L * delta_t * T2L^-1
    const TGeoHMatrix& t2l = GetMatrixT2L();
    const TGeoHMatrix& t2li = t2l.Inverse();
    deltaM.Multiply(&t2li);
    deltaM.MultiplyLeft(&t2l);
  }
  //
}

//_________________________________________________________________
void AliAlgVol::CreateAlignmenMatrix(TGeoHMatrix& alg) const
{
  // create final alignment matrix, accounting for eventual prealignment
  //
  // if the correction for this volume at level j is TAU (global delta) then the combined
  // correction (accounting for reference prealignment) is
  // (Delta_0 * .. Delta_{j-1})^-1 * TAU ( Delta_0 * .. Delta_j)
  // where Delta_i is prealigment global delta of volume i (0 is top)
  // In principle, it can be obtained as:
  // GIdeal_{j-1} * G_{j-1}^-1 * TAU * G_{j}^-1 * GIdeal_{j}^-1
  // where G_i is pre-misaligned reference L2G and GIdeal_i is L2GIdeal,
  // but this creates precision problem.
  // Therefore we use explicitly cached Deltas from prealignment object.
  //
  CreateGloDeltaMatrix(alg);
  //
  const AliAlgVol* par = GetParent();
  if (par) {
    TGeoHMatrix dchain;
    while (par) {
      dchain.MultiplyLeft(&par->GetGlobalDeltaRef());
      par = par->GetParent();
    }
    const TGeoHMatrix& dchaini = dchain.Inverse();
    alg.Multiply(&dchain);
    alg.MultiplyLeft(&dchaini);
  }
  alg *= GetGlobalDeltaRef();

  /* // bad precision ?
  alg.Multiply(&GetMatrixL2G());
  alg.Multiply(&GetMatrixL2GIdeal().Inverse());
  if (par) {
    alg.MultiplyLeft(&par->GetMatrixL2G().Inverse());
    alg.MultiplyLeft(&par->GetMatrixL2GIdeal());
  }
  */
  //
}

/*
//_________________________________________________________________
void AliAlgVol::CreateAlignmenMatrix(TGeoHMatrix& alg) const
{
  // create final alignment matrix, accounting for eventual prealignment
  //
  // deltaGlo_j * X_{j-1} * PdeltaGlo_j * X^-1_{j-1}
  //
  // where deltaGlo_j is global alignment matrix for this volume at level j
  // of herarchy, obtained from CreateGloDeltaMatrix.
  // PdeltaGlo_j is prealignment global matrix and
  // X_i = deltaGlo_i * deltaGlo_{i-1} .. deltaGle_0
  //
  TGeoHMatrix delGloPre,matX;
  CreateGloDeltaMatrix(alg);
  CreatePreGloDeltaMatrix(delGloPre);
  const AliAlgVol* par = this;
  while( (par=par->GetParent()) ) {
    TGeoHMatrix parDelGlo;
    par->CreateGloDeltaMatrix(parDelGlo);
    matX *= parDelGlo;
  }
  alg *= matX;
  alg *= delGloPre;
  alg *= matX.Inverse();
  //
}
*/

//_________________________________________________________________
void AliAlgVol::CreateAlignmentObjects(TClonesArray* arr) const
{
  // add to supplied array alignment object for itself and children
  TClonesArray& parr = *arr;
  TGeoHMatrix algM;
  CreateAlignmenMatrix(algM);
  new (parr[parr.GetEntriesFast()]) AliAlignObjParams(GetName(), GetVolID(), algM, kTRUE);
  int nch = GetNChildren();
  for (int ich = 0; ich < nch; ich++)
    GetChild(ich)->CreateAlignmentObjects(arr);
  //
}

//_________________________________________________________________
void AliAlgVol::UpdateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta)
{
  // recreate fMatL2GReco matrices from ideal L2G matrix and alignment objects
  // used during data reconstruction. For the volume at level J we have
  // L2G' = Delta_J * Delta_{J-1} *...* Delta_0 * L2GIdeal
  // cumulDelta is Delta_{J-1} * ... * Delta_0, supplied by the parent
  //
  fMatL2GReco = fMatL2GIdeal;
  // find alignment object for this volume
  int nalg = algArr->GetEntriesFast();
  const AliAlignObjParams* par = 0;
  for (int i = 0; i < nalg; i++) {
    par = (AliAlignObjParams*)algArr->At(i);
    if (!strcmp(par->GetSymName(), GetSymName()))
      break;
    par = 0;
  }
  TGeoHMatrix delta;
  if (!par)
    AliInfoF("Alignment for %s is absent in Reco-Time alignment object", GetSymName());
  else
    par->GetMatrix(delta);
  if (cumulDelta)
    delta *= *cumulDelta;
  //
  fMatL2GReco.MultiplyLeft(&delta);
  // propagate to children
  for (int ich = GetNChildren(); ich--;)
    GetChild(ich)->UpdateL2GRecoMatrices(algArr, &delta);
  //
}

//______________________________________________________
Bool_t AliAlgVol::OwnsDOFID(Int_t id) const
{
  // check if DOF ID belongs to this volume or its children
  if (id >= fFirstParGloID && id < fFirstParGloID + fNDOFs)
    return kTRUE;
  //
  for (int iv = GetNChildren(); iv--;) {
    AliAlgVol* vol = GetChild(iv);
    if (vol->OwnsDOFID(id))
      return kTRUE;
  }
  return kFALSE;
}

//______________________________________________________
AliAlgVol* AliAlgVol::GetVolOfDOFID(Int_t id) const
{
  // gets volume owning this DOF ID
  if (id >= fFirstParGloID && id < fFirstParGloID + fNDOFs)
    return (AliAlgVol*)this;
  //
  for (int iv = GetNChildren(); iv--;) {
    AliAlgVol* vol = GetChild(iv);
    if ((vol = vol->GetVolOfDOFID(id)))
      return vol;
  }
  return 0;
}

//______________________________________________________
const char* AliAlgVol::GetDOFName(int i) const
{
  // get name of DOF
  return GetGeomDOFName(i);
}

//______________________________________________________
void AliAlgVol::FillDOFStat(AliAlgDOFStat* h) const
{
  // fill statistics info hist
  if (!h)
    return;
  int ndf = GetNDOFs();
  int dof0 = GetFirstParGloID();
  int stat = GetNProcessedPoints();
  for (int idf = 0; idf < ndf; idf++) {
    int dof = idf + dof0;
    h->AddStat(dof, stat);
  }
}

//________________________________________
void AliAlgVol::AddAutoConstraints(TObjArray* constrArr)
{
  // adds automatic constraints
  int nch = GetNChildren();
  //
  if (HasChildrenConstraint()) {
    AliAlgConstraint* constr = new AliAlgConstraint();
    constr->SetConstrainPattern(fConstrChild);
    constr->SetParent(this);
    for (int ich = nch; ich--;) {
      AliAlgVol* child = GetChild(ich);
      if (child->GetExcludeFromParentConstraint())
        continue;
      constr->AddChild(child);
    }
    if (constr->GetNChildren())
      constrArr->Add(constr);
    else
      delete constr;
  }
  //
  for (int ich = 0; ich < nch; ich++)
    GetChild(ich)->AddAutoConstraints(constrArr);
  //
}

} // namespace align
} // namespace o2
