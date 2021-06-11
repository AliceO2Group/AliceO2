// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableVolume.h
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

#include "Align/AlignableVolume.h"
#include "Align/DOFStatistics.h"
#include "Align/GeometricalConstraint.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsBase/GeometryManager.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include <TString.h>
#include <TClonesArray.h>
#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TH1.h>
#include <TAxis.h>
#include <cstdio>

ClassImp(o2::align::AlignableVolume);

using namespace TMath;
using namespace o2::align::utils;

namespace o2
{
namespace align
{

const char* AlignableVolume::sFrameName[AlignableVolume::kNVarFrames] = {"LOC", "TRA"};
//
uint32_t AlignableVolume::sDefGeomFree =
  kDOFBitTX | kDOFBitTY | kDOFBitTZ | kDOFBitPS | kDOFBitTH | kDOFBitPH;
//
const char* AlignableVolume::sDOFName[AlignableVolume::kNDOFGeom] = {"TX", "TY", "TZ", "PSI", "THT", "PHI"};

//_________________________________________________________
AlignableVolume::AlignableVolume(const char* symname, int iid) : TNamed(symname, ""), mVarFrame(kTRA), mIntID(iid), mX(0), mAlp(0), mNDOFs(0), mDOF(0), mNDOFGeomFree(0), mNDOFFree(0), mConstrChild(kDefChildConstr)
                                                                 //
                                                                 ,
                                                                 mParent(nullptr),
                                                                 mChildren(nullptr)
                                                                 //
                                                                 ,
                                                                 mNProcPoints(0),
                                                                 mFirstParGloID(-1),
                                                                 mParVals(nullptr),
                                                                 mParErrs(nullptr),
                                                                 mParLabs(nullptr)
                                                                 //
                                                                 ,
                                                                 mMatL2GReco(),
                                                                 mMatL2G(),
                                                                 mMatL2GIdeal(),
                                                                 mMatT2L(),
                                                                 mMatDeltaRefGlo()
{
  // def c-tor
  setVolID(0);   // volumes have no VID, unless it is sensor
  if (symname) { // real volumes have at least geometric degrees of freedom
    setNDOFs(kNDOFGeom);
  }
  setFreeDOFPattern(sDefGeomFree);
}

//_________________________________________________________
AlignableVolume::~AlignableVolume()
{
  // d-tor
  delete mChildren;
}

//_________________________________________________________
void AlignableVolume::delta2Matrix(TGeoHMatrix& deltaM, const double* delta) const
{
  // prepare delta matrix for the volume from its
  // local delta vector (AliAlignObj convension): dx,dy,dz,,theta,psi,phi
  const double *tr = &delta[0], *rt = &delta[3]; // translation(cm) and rotation(degree)

  //    AliAlignObjParams tempAlignObj;
  //    tempAlignObj.SetRotation(rt[0], rt[1], rt[2]);
  //    tempAlignObj.SetTranslation(tr[0], tr[1], tr[2]);
  //    tempAlignObj.GetMatrix(deltaM);

  detectors::AlignParam tempAlignObj;
  tempAlignObj.setRotation(rt[0], rt[1], rt[2]);
  tempAlignObj.setTranslation(tr[0], tr[1], tr[2]);
  deltaM = tempAlignObj.createMatrix();
}

//__________________________________________________________________
void AlignableVolume::getDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of LOCAL frame:
  // tra' = tau*tra = tau*T2L^-1*loc = T2L^-1*loc' = T2L^-1*delta*loc
  // tau = T2L^-1*delta*T2L
  delta2Matrix(matMod, delta);
  matMod.Multiply(&getMatrixT2L());
  const TGeoHMatrix& t2li = getMatrixT2L().Inverse();
  matMod.MultiplyLeft(&t2li);
}

//__________________________________________________________________
void AlignableVolume::getDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of LOCAL frame of its PARENT;
  // The relMat is matrix for transformation from child to parent frame: LOC = relMat*loc
  //
  // tra' = tau*tra = tau*T2L^-1*loc = T2L^-1*loc' = T2L^-1*relMat^-1*Delta*relMat*loc
  // tau = (relMat*T2L)^-1*Delta*(relMat*T2L)
  delta2Matrix(matMod, delta);
  TGeoHMatrix tmp = relMat;
  tmp *= getMatrixT2L();
  matMod.Multiply(&tmp);
  const TGeoHMatrix& tmpi = tmp.Inverse();
  matMod.MultiplyLeft(&tmpi);
}

//__________________________________________________________________
void AlignableVolume::getDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of the same TRACKING frame:
  // tra' = tau*tra
  delta2Matrix(matMod, delta);
}

//__________________________________________________________________
void AlignableVolume::getDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const
{
  // prepare the variation matrix tau in volume TRACKING frame by applying
  // local delta of modification of TRACKING frame of its PARENT;
  // The relMat is matrix for transformation from child to parent frame: TRA = relMat*tra
  // (see DPosTraDParGeomTRA)
  //
  // tra' = tau*tra = tau*relMat^-1*TRA = relMat^-1*TAU*TRA = relMat^-1*TAU*relMat*tra
  // tau = relMat^-1*TAU*relMat
  delta2Matrix(matMod, delta); // TAU
  matMod.Multiply(&relMat);
  const TGeoHMatrix& reli = relMat.Inverse();
  matMod.MultiplyLeft(&reli);
}

//_________________________________________________________
int AlignableVolume::countParents() const
{
  // count parents in the chain
  int cnt = 0;
  const AlignableVolume* p = this;
  while ((p = p->getParent())) {
    cnt++;
  }
  return cnt;
}

//____________________________________________
void AlignableVolume::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("Lev:%2d IntID:%7d %s | %2d nodes | Effective X:%8.4f Alp:%+.4f | Used Points: %d\n",
         countParents(), getInternalID(), getSymName(), getNChildren(), mX, mAlp, mNProcPoints);
  printf("     DOFs: Tot: %d (offs: %5d) Free: %d  Geom: %d {", mNDOFs, mFirstParGloID, mNDOFFree, mNDOFGeomFree);
  for (int i = 0; i < kNDOFGeom; i++) {
    printf("%d", isFreeDOF(i) ? 1 : 0);
  }
  printf("} in %s frame.", sFrameName[mVarFrame]);
  if (getNChildren()) {
    printf(" Child.Constr: {");
    for (int i = 0; i < kNDOFGeom; i++) {
      printf("%d", isChildrenDOFConstrained(i) ? 1 : 0);
    }
    printf("}");
  }
  if (getExcludeFromParentConstraint()) {
    printf(" Excl.from parent constr.");
  }
  printf("\n");
  //
  if (opts.Contains("par") && mParVals) {
    printf("     Lb: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%10d  ", getParLab(i));
    }
    printf("\n");
    printf("     Vl: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%+9.3e  ", getParVal(i));
    }
    printf("\n");
    printf("     Er: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%+9.3e  ", getParErr(i));
    }
    printf("\n");
  }

  if (opts.Contains("mat")) { // print matrices
    printf("L2G ideal   : ");
    getMatrixL2GIdeal().Print();
    printf("L2G misalign: ");
    getMatrixL2G().Print();
    printf("L2G RecoTime: ");
    getMatrixL2GReco().Print();
    printf("T2L (fake)  : ");
    getMatrixT2L().Print();
  }
  //
}

//____________________________________________
void AlignableVolume::prepareMatrixL2G(bool reco)
{
  // extract from geometry L2G matrix, depending on reco flag, set it as a reco-time
  // or current alignment matrix
  const char* path = getSymName();
  if (gGeoManager->GetAlignableEntry(path)) {
    const TGeoHMatrix* l2g = base::GeometryManager::getMatrix(path);
    if (!l2g) {
      LOG(FATAL) << "Failed to find L2G matrix for alignable " << path;
    }
    reco ? setMatrixL2GReco(*l2g) : setMatrixL2G(*l2g);
  } else { // extract from path
    if (!gGeoManager->CheckPath(path)) {
      LOG(FATAL) << "Volume path " << path << " is not valid!";
    }
    TGeoPhysicalNode* node = (TGeoPhysicalNode*)gGeoManager->GetListOfPhysicalNodes()->FindObject(path);
    TGeoHMatrix l2g;
    if (!node) {
      LOG(WARNING) << "volume " << path << " was not misaligned, extracting original matrix";
      if (!base::GeometryManager::getOriginalMatrix(path, l2g)) {
        LOG(FATAL) << "Failed to find ideal L2G matrix for " << path;
      }
    } else {
      l2g = *node->GetMatrix();
    }
    reco ? setMatrixL2GReco(l2g) : setMatrixL2G(l2g);
  }
}

//____________________________________________
void AlignableVolume::prepareMatrixL2GIdeal()
{
  // extract from geometry ideal L2G matrix
  TGeoHMatrix mtmp;
  if (!base::GeometryManager::getOriginalMatrix(getSymName(), mtmp)) {
    LOG(FATAL) << "Failed to find ideal L2G matrix for " << getSymName();
  }
  setMatrixL2GIdeal(mtmp);
}

//____________________________________________
void AlignableVolume::prepareMatrixT2L()
{
  // for non-sensors we define the fake tracking frame with the alpha angle being
  // the average angle of centers of its children
  //
  double tot[3] = {0, 0, 0}, loc[3] = {0, 0, 0}, glo[3];
  int nch = getNChildren();
  for (int ich = nch; ich--;) {
    AlignableVolume* vol = getChild(ich);
    vol->getMatrixL2GIdeal().LocalToMaster(loc, glo);
    for (int j = 3; j--;) {
      tot[j] += glo[j];
    }
  }
  if (nch) {
    for (int j = 3; j--;) {
      tot[j] /= nch;
    }
  }
  //
  mAlp = TMath::ATan2(tot[1], tot[0]);
  utils::bringToPiPM(mAlp);
  //
  mX = TMath::Sqrt(tot[0] * tot[0] + tot[1] * tot[1]);
  //
  // 1st create Tracking to Global matrix
  mMatT2L.Clear();
  mMatT2L.SetDx(mX);
  mMatT2L.RotateZ(mAlp * RadToDeg());
  // then convert it to Tracking to Local  matrix
  const TGeoHMatrix& l2gi = getMatrixL2GIdeal().Inverse();
  mMatT2L.MultiplyLeft(&l2gi);
  //
}

//____________________________________________
void AlignableVolume::setMatrixT2L(const TGeoHMatrix& m)
{
  // set the T2L matrix and define tracking frame
  // Note that this method is used for the externally set matrices
  // (in case of sensors). For other volumes the tracking frame and matrix
  // is defined in the prepareMatrixT2L method
  mMatT2L = m;
  setTrackingFrame();
}

//__________________________________________________________________
void AlignableVolume::setTrackingFrame()
{
  // Define tracking frame of the sensor
  // This method should be implemented for sensors, which receive the T2L
  // matrix from the geometry
  LOG(ERROR) << "Volume " << GetName() << " was supposed to implement its own method";
}

//__________________________________________________________________
void AlignableVolume::assignDOFs(int& cntDOFs, float* pars, float* errs, int* labs)
{
  // Assigns offset of the DOFS of this volume in global array of DOFs, attaches arrays to volumes
  //
  mParVals = pars + cntDOFs;
  mParErrs = errs + cntDOFs;
  mParLabs = labs + cntDOFs;
  setFirstParGloID(cntDOFs);
  for (int i = 0; i < mNDOFs; i++) {
    mParLabs[i] = getInternalID() * 100 + i;
  }
  cntDOFs += mNDOFs; // increment total DOFs count
  //
  int nch = getNChildren(); // go over childs
  for (int ich = 0; ich < nch; ich++) {
    getChild(ich)->assignDOFs(cntDOFs, pars, errs, labs);
  }
  //
  return;
}

//__________________________________________________________________
void AlignableVolume::initDOFs()
{
  // initialize degrees of freedom
  //
  // Do we need this strict condition?
  if (getInitDOFsDone()) {
    LOG(FATAL) << "DOFs are already initialized for " << GetName();
  }
  for (int i = 0; i < mNDOFs; i++) {
    if (mParErrs[i] < -9999 && isZeroAbs(mParVals[i])) {
      fixDOF(i);
    }
  }
  calcFree(true);
  setInitDOFsDone();
}

//__________________________________________________________________
void AlignableVolume::calcFree(bool condFix)
{
  // calculate free dofs. If condFix==true, condition parameter a la pede, i.e. error < 0
  mNDOFFree = mNDOFGeomFree = 0;
  for (int i = 0; i < mNDOFs; i++) {
    if (!isFreeDOF(i)) {
      if (condFix) {
        setParErr(i, -999);
      }
      continue;
    }
    mNDOFFree++;
    if (i < kNDOFGeom) {
      mNDOFGeomFree++;
    }
  }
  //
}

//__________________________________________________________________
void AlignableVolume::setNDOFs(int n)
{
  // book global degrees of freedom
  if (n < kNDOFGeom) {
    n = kNDOFGeom;
  }
  mNDOFs = n;
}

//__________________________________________________________________
void AlignableVolume::addChild(AlignableVolume* ch)
{
  // add child volume
  if (!mChildren) {
    mChildren = new TObjArray();
    mChildren->SetOwner(false);
  }
  mChildren->AddLast(ch);
}

//__________________________________________________________________
void AlignableVolume::setParVals(int npar, double* vl, double* er)
{
  // set parameters
  if (npar > mNDOFs) {
    LOG(FATAL) << "Volume " << GetName() << " has " << mNDOFs << " dofs";
  }
  for (int i = 0; i < npar; i++) {
    mParVals[i] = vl[i];
    mParErrs[i] = er ? er[i] : 0;
  }
}

//__________________________________________________________________
bool AlignableVolume::isCondDOF(int i) const
{
  // is DOF free and conditioned?
  return (!isZeroAbs(getParVal(i)) || !isZeroAbs(getParErr(i)));
}

//______________________________________________________
int AlignableVolume::finalizeStat(DOFStatistics* st)
{
  // finalize statistics on processed points
  mNProcPoints = 0;
  for (int ich = getNChildren(); ich--;) {
    AlignableVolume* child = getChild(ich);
    mNProcPoints += child->finalizeStat(st);
  }
  if (st) {
    fillDOFStat(st);
  }
  return mNProcPoints;
}

//______________________________________________________
void AlignableVolume::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params template file for PEDE
  enum { kOff,
         kOn,
         kOnOn };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyParam = "parameter";
  TString opts = opt;
  opts.ToLower();
  bool showDef = opts.Contains("d"); // show free DOF even if not preconditioned
  bool showFix = opts.Contains("f"); // show DOF even if fixed
  bool showNam = opts.Contains("n"); // show volume name even if no nothing else is printable
  //
  // is there something to print ?
  int nCond(0), nFix(0), nDef(0);
  for (int i = 0; i < mNDOFs; i++) {
    if (!isFreeDOF(i)) {
      nFix++;
    }
    if (isCondDOF(i)) {
      nCond++;
    }
    if (!isCondDOF(i) && isFreeDOF(i)) {
      nDef++;
    }
  }
  //
  int cmt = nCond > 0 || nFix > 0 ? kOff : kOn; // do we comment the "parameter" keyword for this volume
  if (!nFix) {
    showFix = false;
  }
  if (!nDef) {
    showDef = false;
  }
  //
  if (nCond || showDef || showFix || showNam) {
    fprintf(parOut, "%s%s %s\t\tDOF/Free: %d/%d (%s) %s\n", comment[cmt], kKeyParam, comment[kOnOn],
            getNDOFs(), getNDOFFree(), sFrameName[mVarFrame], GetName());
  }
  //
  if (nCond || showDef || showFix) {
    for (int i = 0; i < mNDOFs; i++) {
      cmt = kOn;
      if (isCondDOF(i) || !isFreeDOF(i)) {
        cmt = kOff;
      } // free-conditioned : MUST print
      else if (!isFreeDOF(i)) {
        if (!showFix) {
          continue;
        }
      } // Fixed: print commented if asked
      else if (!showDef) {
        continue;
      } // free-unconditioned: print commented if asked
      //
      fprintf(parOut, "%s %9d %+e %+e\t%s %s p%d\n", comment[cmt], getParLab(i),
              getParVal(i), getParErr(i), comment[kOnOn], isFreeDOF(i) ? "  " : "FX", i);
    }
    fprintf(parOut, "\n");
  }
  // children volume
  int nch = getNChildren();
  //
  for (int ich = 0; ich < nch; ich++) {
    getChild(ich)->writePedeInfo(parOut, opt);
  }
  //
}

//_________________________________________________________________
void AlignableVolume::createGloDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create global matrix deltaM from mParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  //
  createLocDeltaMatrix(deltaM);
  const TGeoHMatrix& l2g = getMatrixL2G();
  const TGeoHMatrix& l2gi = l2g.Inverse();
  deltaM.Multiply(&l2gi);
  deltaM.MultiplyLeft(&l2g);
  //
}

/*
//_________________________________________________________________
void AlignableVolume::createGloDeltaMatrix(TGeoHMatrix &deltaM) const
{
  // Create global matrix deltaM from mParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  //
  // deltaM = Z * deltaL * Z^-1
  // where deltaL is local correction matrix and Z is matrix defined as
  // Z = [ Prod_{k=0}^{j-1} G_k * deltaL_k * G^-1_k ] * G_j
  // with j=being the level of the volume in the hierarchy
  //
  createLocDeltaMatrix(deltaM);
  TGeoHMatrix zMat = getMatrixL2G();
  const AlignableVolume* par = this;
  while( (par=par->getParent()) ) {
    TGeoHMatrix locP;
    par->createLocDeltaMatrix(locP);
    locP.MultiplyLeft( &par->getMatrixL2G() );
    locP.Multiply( &par->getMatrixL2G().Inverse() );
    zMat.MultiplyLeft( &locP );
  }
  deltaM.MultiplyLeft( &zMat );
  deltaM.Multiply( &zMat.Inverse() );
  //
}
*/

//_________________________________________________________________
void AlignableVolume::createPreGloDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create prealignment global matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = Delta_j * Delta_{j-1} ... Delta_0 * GO_j
  // where Delta_j is global prealignment matrix for volume at level j
  // we get by induction
  // Delta_j = G_j * GO^-1_j * GO_{j-1} * G^-1_{j-1}
  //
  deltaM = getMatrixL2G();
  deltaM *= getMatrixL2GIdeal().Inverse();
  const AlignableVolume* par = getParent();
  if (par) {
    deltaM *= par->getMatrixL2GIdeal();
    deltaM *= par->getMatrixL2G().Inverse();
  }
  //
}

/*
  // this is an alternative lengthy way !
//_________________________________________________________________
void AlignableVolume::createPreGloDeltaMatrix(TGeoHMatrix &deltaM) const
{
  // Create prealignment global matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = Delta_j * Delta_{j-1} ... Delta_0 * GO_j
  // where Delta_j is global prealignment matrix for volume at level j
  // we get by induction
  // Delta_j = G_j * GO^-1_j * GO_{j-1} * G^-1_{j-1}
  //
  createPreLocDeltaMatrix(deltaM);
  TGeoHMatrix zMat = getMatrixL2GIdeal();
  const AlignableVolume* par = this;
  while( (par=par->getParent()) ) {
    TGeoHMatrix locP;
    par->createPreLocDeltaMatrix(locP);
    locP.MultiplyLeft( &par->getMatrixL2GIdeal() );
    locP.Multiply( &par->getMatrixL2GIdeal().Inverse() );
    zMat.MultiplyLeft( &locP );
  }
  deltaM.MultiplyLeft( &zMat );
  deltaM.Multiply( &zMat.Inverse() );
  //
}
*/

//_________________________________________________________________
void AlignableVolume::createPreLocDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create prealignment local matrix deltaM from prealigned G and
  // original GO local-to-global matrices
  //
  // From G_j = GO_0 * delta_0 * GO^-1_0 * GO_1 * delta_1 ... GO^-1_{j-1}*GO_{j}*delta_j
  // where delta_j is local prealignment matrix for volume at level j
  // we get by induction
  // delta_j = GO^-1_j * GO_{j-1} * G^-1_{j-1} * G^_{j}
  //
  const AlignableVolume* par = getParent();
  deltaM = getMatrixL2GIdeal().Inverse();
  if (par) {
    deltaM *= par->getMatrixL2GIdeal();
    deltaM *= par->getMatrixL2G().Inverse();
  }
  deltaM *= getMatrixL2G();
  //
}

//_________________________________________________________________
void AlignableVolume::createLocDeltaMatrix(TGeoHMatrix& deltaM) const
{
  // Create local matrix deltaM from mParVals array containing corrections.
  // This deltaM does not account for eventual prealignment
  // Volume knows if its variation was done in TRA or LOC frame
  double corr[kNDOFGeom];
  for (int i = kNDOFGeom; i--;) {
    corr[i] = mParVals[i];
  } // we need doubles
  delta2Matrix(deltaM, corr);
  if (isFrameTRA()) { // we need corrections in local frame!
    // l' = T2L * delta_t * t = T2L * delta_t * T2L^-1 * l = delta_l * l
    // -> delta_l = T2L * delta_t * T2L^-1
    const TGeoHMatrix& t2l = getMatrixT2L();
    const TGeoHMatrix& t2li = t2l.Inverse();
    deltaM.Multiply(&t2li);
    deltaM.MultiplyLeft(&t2l);
  }
  //
}

//_________________________________________________________________
void AlignableVolume::createAlignmenMatrix(TGeoHMatrix& alg) const
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
  createGloDeltaMatrix(alg);
  //
  const AlignableVolume* par = getParent();
  if (par) {
    TGeoHMatrix dchain;
    while (par) {
      dchain.MultiplyLeft(&par->getGlobalDeltaRef());
      par = par->getParent();
    }
    const TGeoHMatrix& dchaini = dchain.Inverse();
    alg.Multiply(&dchain);
    alg.MultiplyLeft(&dchaini);
  }
  alg *= getGlobalDeltaRef();

  /* // bad precision ?
  alg.Multiply(&getMatrixL2G());
  alg.Multiply(&getMatrixL2GIdeal().Inverse());
  if (par) {
    alg.MultiplyLeft(&par->getMatrixL2G().Inverse());
    alg.MultiplyLeft(&par->getMatrixL2GIdeal());
  }
  */
}

/*
//_________________________________________________________________
void AlignableVolume::createAlignmenMatrix(TGeoHMatrix& alg) const
{
  // create final alignment matrix, accounting for eventual prealignment
  //
  // deltaGlo_j * X_{j-1} * PdeltaGlo_j * X^-1_{j-1}
  //
  // where deltaGlo_j is global alignment matrix for this volume at level j
  // of herarchy, obtained from createGloDeltaMatrix.
  // PdeltaGlo_j is prealignment global matrix and
  // X_i = deltaGlo_i * deltaGlo_{i-1} .. deltaGle_0
  //
  TGeoHMatrix delGloPre,matX;
  createGloDeltaMatrix(alg);
  createPreGloDeltaMatrix(delGloPre);
  const AlignableVolume* par = this;
  while( (par=par->getParent()) ) {
    TGeoHMatrix parDelGlo;
    par->createGloDeltaMatrix(parDelGlo);
    matX *= parDelGlo;
  }
  alg *= matX;
  alg *= delGloPre;
  alg *= matX.Inverse();
  //
}
*/

//_________________________________________________________________
void AlignableVolume::createAlignmentObjects(TClonesArray* arr) const
{
  // add to supplied array alignment object for itself and children
  TClonesArray& parr = *arr;
  TGeoHMatrix algM;
  createAlignmenMatrix(algM);
  //  new (parr[parr.GetEntriesFast()]) AliAlignObjParams(GetName(), getVolID(), algM, true);
  const double* translation = algM.GetTranslation();
  const double* rotation = algM.GetRotationMatrix();
  new (parr[parr.GetEntriesFast()]) detectors::AlignParam(GetName(), getVolID(),
                                                          translation[0], translation[1], translation[2],
                                                          rotation[0], rotation[1], rotation[2], true);
  int nch = getNChildren();
  for (int ich = 0; ich < nch; ich++) {
    getChild(ich)->createAlignmentObjects(arr);
  }
}

//_________________________________________________________________
void AlignableVolume::updateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta)
{
  // recreate mMatL2GReco matrices from ideal L2G matrix and alignment objects
  // used during data reconstruction. For the volume at level J we have
  // L2G' = Delta_J * Delta_{J-1} *...* Delta_0 * L2GIdeal
  // cumulDelta is Delta_{J-1} * ... * Delta_0, supplied by the parent
  //
  mMatL2GReco = mMatL2GIdeal;
  // find alignment object for this volume
  int nalg = algArr->GetEntriesFast();
  const detectors::AlignParam* par = nullptr;
  for (int i = 0; i < nalg; i++) {
    par = (detectors::AlignParam*)algArr->At(i);
    if (!strcmp(par->getSymName().c_str(), getSymName())) {
      break;
    }
    par = nullptr;
  }
  TGeoHMatrix delta;
  if (!par) {
    LOG(INFO) << "Alignment for " << getSymName() << " is absent in Reco-Time alignment object";
  } else {
    delta = par->createMatrix();
  }
  //    par->GetMatrix(delta);
  if (cumulDelta) {
    delta *= *cumulDelta;
  }
  //
  mMatL2GReco.MultiplyLeft(&delta);
  // propagate to children
  for (int ich = getNChildren(); ich--;) {
    getChild(ich)->updateL2GRecoMatrices(algArr, &delta);
  }
  //
}

//______________________________________________________
bool AlignableVolume::ownsDOFID(int id) const
{
  // check if DOF ID belongs to this volume or its children
  if (id >= mFirstParGloID && id < mFirstParGloID + mNDOFs) {
    return true;
  }
  //
  for (int iv = getNChildren(); iv--;) {
    AlignableVolume* vol = getChild(iv);
    if (vol->ownsDOFID(id)) {
      return true;
    }
  }
  return false;
}

//______________________________________________________
AlignableVolume* AlignableVolume::getVolOfDOFID(int id) const
{
  // gets volume owning this DOF ID
  if (id >= mFirstParGloID && id < mFirstParGloID + mNDOFs) {
    return (AlignableVolume*)this;
  }
  //
  for (int iv = getNChildren(); iv--;) {
    AlignableVolume* vol = getChild(iv);
    if ((vol = vol->getVolOfDOFID(id))) {
      return vol;
    }
  }
  return nullptr;
}

//______________________________________________________
const char* AlignableVolume::getDOFName(int i) const
{
  // get name of DOF
  return getGeomDOFName(i);
}

//______________________________________________________
void AlignableVolume::fillDOFStat(DOFStatistics* h) const
{
  // fill statistics info hist
  if (!h) {
    return;
  }
  int ndf = getNDOFs();
  int dof0 = getFirstParGloID();
  int stat = getNProcessedPoints();
  for (int idf = 0; idf < ndf; idf++) {
    int dof = idf + dof0;
    h->addStat(dof, stat);
  }
}

//________________________________________
void AlignableVolume::addAutoConstraints(TObjArray* constrArr)
{
  // adds automatic constraints
  int nch = getNChildren();
  //
  if (hasChildrenConstraint()) {
    GeometricalConstraint* constr = new GeometricalConstraint();
    constr->setConstrainPattern(mConstrChild);
    constr->setParent(this);
    for (int ich = nch; ich--;) {
      AlignableVolume* child = getChild(ich);
      if (child->getExcludeFromParentConstraint()) {
        continue;
      }
      constr->addChild(child);
    }
    if (constr->getNChildren()) {
      constrArr->Add(constr);
    } else {
      delete constr;
    }
  }
  for (int ich = 0; ich < nch; ich++) {
    getChild(ich)->addAutoConstraints(constrArr);
  }
}

} // namespace align
} // namespace o2
