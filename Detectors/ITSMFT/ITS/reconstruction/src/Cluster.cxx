/// \file Cluster.cxx
/// \brief Implementation of the ITS cluster

#include "ITSReconstruction/Cluster.h"
#include "FairLogger.h"

#include <TGeoMatrix.h>
#include <TMath.h>
#include <TString.h>

#include <cstdlib>

using namespace AliceO2::ITS;

ClassImp(AliceO2::ITS::Cluster)

GeometryTGeo* Cluster::sGeom = 0;
UInt_t Cluster::sMode = 0;

//_____________________________________________________
Cluster::Cluster()
  : mTracks{ -1, -1, -1 },
    mX(0),
    mY(0),
    mZ(0),
    mSigmaY2(0),
    mSigmaZ2(0),
    mSigmaYZ(0),
    mVolumeId(0),
    mCharge(0),
    mRecoInfo(0),
    mNxNzN(0)
//#ifdef _ClusterTopology_
    ,
    mPatternNRows(0),
    mPatternNCols(0),
    mPatternMinRow(0),
    mPatternMinCol(0)
//#endif
{
// default constructor
//#ifdef _ClusterTopology_
  memset(mPattern, 0, kMaxPatternBytes * sizeof(UChar_t));
//#endif
}

//_____________________________________________________
Cluster::~Cluster()
{
  // default destructor
}

//_____________________________________________________
Cluster::Cluster(const Cluster& cluster)
  : FairTimeStamp(cluster),
    mTracks{ cluster.mTracks[0], cluster.mTracks[1], cluster.mTracks[2] },
    mX(cluster.mX),
    mY(cluster.mY),
    mZ(cluster.mZ),
    mSigmaY2(cluster.mSigmaY2),
    mSigmaZ2(cluster.mSigmaZ2),
    mSigmaYZ(cluster.mSigmaYZ),
    mVolumeId(cluster.mVolumeId),
    mCharge(cluster.mCharge),
    mRecoInfo(cluster.mRecoInfo),
    mNxNzN(cluster.mNxNzN)
//#ifdef _ClusterTopology_
    ,
    mPatternNRows(cluster.mPatternNRows),
    mPatternNCols(cluster.mPatternNCols),
    mPatternMinRow(cluster.mPatternMinRow),
    mPatternMinCol(cluster.mPatternMinCol)
//#endif
{
// copy constructor
//#ifdef _ClusterTopology_
  memcpy(mPattern, cluster.mPattern, kMaxPatternBytes * sizeof(UChar_t));
//#endif
}

//______________________________________________________________________________
const TGeoHMatrix* Cluster::getTracking2LocalMatrix() const
{
  // get tracking to local matrix (sensor!!!)
  return (TGeoHMatrix*)sGeom->getMatrixT2L(getVolumeId());
}

//______________________________________________________________________________
TGeoHMatrix* Cluster::getMatrix(Bool_t) const
{
  // get chip matrix (sensor!)
  return (TGeoHMatrix*)sGeom->getMatrixSensor(getVolumeId());
}

//______________________________________________________________________________
void Cluster::print(Option_t* option) const
{
  // Print cluster information.
  TString str = option;
  str.ToLower();
  printf("Cl.in mod %5d, nx:%3d nz:%3d n:%d |Err^2:%.3e %.3e %+.3e |", getVolumeId(), getNx(), getNz(), getNPix(),
         getSigmaY2(), getSigmaZ2(), getSigmaYZ());
  printf("XYZ: (%+.4e %+.4e %+.4e ", getX(), getY(), getZ());
  if (isFrameLoc())
    printf("LOC)");
  else if (isFrameGlo())
    printf("GLO)");
  else if (isFrameTrk())
    printf("TRK)");
  if (str.Contains("glo") && !isFrameGlo() && sGeom) {
    Float_t g[3];
    getGlobalXYZ(g);
    printf(" (%+.4e %+.4e %+.4e GLO)", g[0], g[1], g[2]);
  }
  printf(" MClb:");
  for (int i = 0; i < 3; i++)
    printf(" %5d", getLabel(i));
  if (TestBit(kSplit))
    printf(" Spl");
  printf("\n");
//
//#ifdef _ClusterTopology_
  if (str.Contains("p")) { // print pattern
    int nr = getPatternRowSpan();
    int nc = getPatternColSpan();
    printf("Pattern: %d rows from %d", nr, mPatternMinRow);
    if (isPatternRowsTruncated())
      printf("(truncated)");
    printf(", %d columns from %d", nc, mPatternMinCol);
    if (isPatternColsTruncated())
      printf("(truncated)");
    printf("\n");
    for (int ir = 0; ir < nr; ir++) {
      for (int ic = 0; ic < nc; ic++)
        printf("%c", testPixel(ir, ic) ? '+' : '-');
      printf("\n");
    }
  }
//#endif
  //
}

//#ifdef _ClusterTopology_
//______________________________________________________________________________
void Cluster::resetPattern()
{
  // reset pixels pattern
  memset(mPattern, 0, kMaxPatternBytes * sizeof(UChar_t));
}

//______________________________________________________________________________
Bool_t Cluster::testPixel(UShort_t row, UShort_t col) const
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return kFALSE;
  int bytn = nbits >> 3; // 1/8
  int bitn = nbits % 8;
  return (mPattern[bytn] & (0x1 << bitn)) != 0;
  //
}

//______________________________________________________________________________
void Cluster::setPixel(UShort_t row, UShort_t col, Bool_t fired)
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return;
  int bytn = nbits >> 3; // 1/8
  int bitn = nbits % 8;
  if (nbits >= kMaxPatternBits)
    exit(1);
  if (fired)
    mPattern[bytn] |= (0x1 << bitn);
  else
    mPattern[bytn] &= (0xff ^ (0x1 << bitn));
  //
}

//______________________________________________________________________________
void Cluster::setPatternRowSpan(UShort_t nr, Bool_t truncated)
{
  // set pattern span in rows, flag if truncated
  mPatternNRows = kSpanMask & nr;
  if (truncated)
    mPatternNRows |= kTruncateMask;
}

//______________________________________________________________________________
void Cluster::setPatternColSpan(UShort_t nc, Bool_t truncated)
{
  // set pattern span in columns, flag if truncated
  mPatternNCols = kSpanMask & nc;
  if (truncated)
    mPatternNCols |= kTruncateMask;
}

//#endif

//______________________________________________________________________________
Bool_t Cluster::getGlobalXYZ(Float_t xyz[3]) const
{
  // Get the global coordinates of the cluster
  // All the needed information is taken only
  // from TGeo (single precision).
  if (isFrameGlo()) {
    xyz[0] = getX();
    xyz[1] = getY();
    xyz[2] = getZ();
    return kTRUE;
  }
  //
  Double_t lxyz[3] = { 0, 0, 0 };
  if (isFrameTrk()) {
    const TGeoHMatrix* mt = getTracking2LocalMatrix();
    if (!mt)
      return kFALSE;
    Double_t txyz[3] = { getX(), getY(), getZ() };
    mt->LocalToMaster(txyz, lxyz);
  } else {
    lxyz[0] = getX();
    lxyz[1] = getY();
    lxyz[2] = getZ();
  }
  //
  TGeoHMatrix* ml = getMatrix();
  if (!ml)
    return kFALSE;
  Double_t gxyz[3] = { 0, 0, 0 };
  ml->LocalToMaster(lxyz, gxyz);
  xyz[0] = gxyz[0];
  xyz[1] = gxyz[1];
  xyz[2] = gxyz[2];
  return kTRUE;
}

//______________________________________________________________________________
Bool_t Cluster::getGlobalCov(Float_t cov[6]) const
{
  // Get the global covariance matrix of the cluster coordinates
  // All the needed information is taken only
  // from TGeo.
  // Note: regardless on in which frame the coordinates are, the errors are always in tracking frame
  //

  const TGeoHMatrix* mt = getTracking2LocalMatrix();
  if (!mt)
    return kFALSE;

  TGeoHMatrix* ml = getMatrix();
  if (!ml)
    return kFALSE;

  TGeoHMatrix m;
  Double_t tcov[9] = { 0, 0, 0, 0, mSigmaY2, mSigmaYZ, 0, mSigmaYZ, mSigmaZ2 };
  m.SetRotation(tcov);
  m.Multiply(&mt->Inverse());
  m.Multiply(&ml->Inverse());
  m.MultiplyLeft(mt);
  m.MultiplyLeft(ml);
  Double_t* ncov = m.GetRotationMatrix();
  cov[0] = ncov[0];
  cov[1] = ncov[1];
  cov[2] = ncov[2];
  cov[3] = ncov[4];
  cov[4] = ncov[5];
  cov[5] = ncov[8];

  return kTRUE;
}

//______________________________________________________________________________
Bool_t Cluster::getXRefPlane(Float_t& xref) const
{
  // Get the distance between the origin and the ref.plane.
  // All the needed information is taken only from TGeo.

  const TGeoHMatrix* mt = getTracking2LocalMatrix();
  if (!mt)
    return kFALSE;

  TGeoHMatrix* ml = getMatrix();
  if (!ml)
    return kFALSE;

  TGeoHMatrix m = *mt;
  m.MultiplyLeft(ml);

  xref = -(m.Inverse()).GetTranslation()[0];
  return kTRUE;
}

Bool_t Cluster::getXAlphaRefPlane(Float_t& x, Float_t& alpha) const
{
  // Get the distance between the origin and the ref. plane together with
  // the rotation anlge of the ref. plane.
  // All the needed information is taken only
  // from TGeo.
  const TGeoHMatrix* mt = getTracking2LocalMatrix();
  if (!mt)
    return kFALSE;

  const TGeoHMatrix* ml = getMatrix();
  if (!ml)
    return kFALSE;

  TGeoHMatrix m(*ml);
  m.Multiply(mt);
  const Double_t txyz[3] = { 0. };
  Double_t xyz[3] = { 0. };
  m.LocalToMaster(txyz, xyz);

  x = TMath::Sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);

  Double_t a = TMath::ATan2(xyz[1], xyz[0]);
  if (a < 0)
    a += TMath::TwoPi();
  else if (a >= TMath::TwoPi())
    a -= TMath::TwoPi();
  alpha = a;

  return kTRUE;
}

//______________________________________________________________________________
void Cluster::goToFrameGlo()
{
  // convert to global frame
  if (isFrameGlo())
    return;
  double loc[3], glo[3];
  //
  if (isFrameTrk()) {
    double curr[3] = { getX(), getY(), getZ() };
    getTracking2LocalMatrix()->LocalToMaster(curr, loc);
    ResetBit(kFrameTrk);
  } else {
    loc[0] = getX();
    loc[1] = getY();
    loc[2] = getZ();
    ResetBit(kFrameLoc);
  }
  getMatrix()->LocalToMaster(loc, glo);
  setX(glo[0]);
  setY(glo[1]);
  setZ(glo[2]);
  SetBit(kFrameGlo);
  //
}

//______________________________________________________________________________
void Cluster::goToFrameLoc()
{
  // convert to local frame
  if (isFrameLoc())
    return;
  //
  double loc[3], glo[3];
  if (isFrameTrk()) {
    double curr[3] = { getX(), getY(), getZ() };
    getTracking2LocalMatrix()->LocalToMaster(curr, loc);
    ResetBit(kFrameTrk);
  } else {
    glo[0] = getX();
    glo[1] = getY();
    glo[2] = getZ();
    getMatrix()->MasterToLocal(glo, loc);
    ResetBit(kFrameLoc);
  }
  SetBit(kFrameLoc);
  setX(loc[0]);
  setY(loc[1]);
  setZ(loc[2]);
  //
}

//______________________________________________________________________________
void Cluster::getLocalXYZ(Float_t xyz[3]) const
{
  // get local coordinates
  if (isFrameLoc()) {
    xyz[0] = getX();
    xyz[1] = 0;
    xyz[2] = getZ();
    return;
  }
  double loc[3], glo[3];
  if (isFrameTrk()) {
    double curr[3] = { getX(), getY(), getZ() };
    getTracking2LocalMatrix()->LocalToMaster(curr, loc);
  } else {
    glo[0] = getX();
    glo[1] = getY();
    glo[2] = getZ();
    getMatrix()->MasterToLocal(glo, loc);
  }
  for (int i = 3; i--;)
    xyz[i] = loc[i];
  //
}

//______________________________________________________________________________
void Cluster::goToFrameTrk()
{
  // convert to tracking frame
  if (isFrameTrk())
    return;
  //
  double loc[3], trk[3];
  if (isFrameGlo()) {
    double glo[3] = { getX(), getY(), getZ() };
    getMatrix()->MasterToLocal(glo, loc);
    ResetBit(kFrameGlo);
  } else {
    loc[0] = getX();
    loc[1] = getY();
    loc[2] = getZ();
    ResetBit(kFrameLoc);
  }
  // now in local frame
  getTracking2LocalMatrix()->MasterToLocal(loc, trk);
  SetBit(kFrameTrk);
  setX(trk[0]);
  setY(trk[1]);
  setZ(trk[2]);
  //
}

//______________________________________________________________________________
void Cluster::getTrackingXYZ(Float_t xyz[3]) const
{
  // convert to tracking frame
  if (isFrameTrk()) {
    xyz[0] = getX();
    xyz[1] = getY();
    xyz[2] = getZ();
    return;
  }
  //
  double loc[3], trk[3];
  if (isFrameGlo()) {
    double glo[3] = { getX(), getY(), getZ() };
    getMatrix()->MasterToLocal(glo, loc);
  } else {
    loc[0] = getX();
    loc[1] = getY();
    loc[2] = getZ();
  }
  // now in local frame
  getTracking2LocalMatrix()->MasterToLocal(loc, trk);
  for (int i = 3; i--;)
    xyz[i] = trk[i];
  //
}

//______________________________________________________________________________
Int_t Cluster::Compare(const TObject* obj) const
{
  // compare clusters accodring to specific mode
  const Cluster* px = (const Cluster*)obj;
  float xyz[3], xyz1[3];
  if (sMode & kSortIdLocXZ) { // sorting in local frame
    if (getVolumeId() == px->getVolumeId()) {
      getLocalXYZ(xyz);
      px->getLocalXYZ(xyz1);
      if (xyz[0] < xyz1[0])
        return -1; // sort in X
      if (xyz[0] > xyz1[0])
        return 1;
      if (xyz[2] < xyz1[2])
        return -1; // then in Z
      if (xyz[2] > xyz1[2])
        return 1;
      return 0;
    }
    return int(getVolumeId()) - int(px->getVolumeId());
  }
  if (sMode & kSortIdTrkYZ) { // sorting in tracking frame
    if (getVolumeId() == px->getVolumeId()) {
      getTrackingXYZ(xyz);
      px->getTrackingXYZ(xyz1);
      if (xyz[1] < xyz1[1])
        return -1; // sort in Y
      if (xyz[1] > xyz1[1])
        return 1;
      if (xyz[2] < xyz1[2])
        return -1; // then in Z
      if (xyz[2] > xyz1[2])
        return 1;
      return 0;
    }
    return int(getVolumeId()) - int(px->getVolumeId());
  }
  LOG(FATAL) << "Unknown mode for sorting: " << sMode << FairLogger::endl;
  return 0;
}

//______________________________________________________________________________
Bool_t Cluster::isEqual(const TObject* obj) const
{
  // compare clusters accodring to specific mode
  const Cluster* px = (const Cluster*)obj;
  const Float_t kTol = 1e-5;
  float xyz[3], xyz1[3];
  if (sMode & kSortIdLocXZ) { // sorting in local frame
    if (getVolumeId() != px->getVolumeId())
      return kFALSE;
    getLocalXYZ(xyz);
    px->getLocalXYZ(xyz1);
    return (TMath::Abs(xyz[0] - xyz1[0]) < kTol && TMath::Abs(xyz[2] - xyz1[2]) < kTol) ? kTRUE : kFALSE;
  }
  if (sMode & kSortIdTrkYZ) { // sorting in tracking frame
    if (getVolumeId() != px->getVolumeId())
      return kFALSE;
    getTrackingXYZ(xyz);
    px->getTrackingXYZ(xyz1);
    return (TMath::Abs(xyz[1] - xyz1[1]) < kTol && TMath::Abs(xyz[2] - xyz1[2]) < kTol) ? kTRUE : kFALSE;
  }
  LOG(FATAL) << "Unknown mode for sorting: " << sMode << FairLogger::endl;
  return kFALSE;
}

//______________________________________________________________________________
Bool_t Cluster::hasCommonTrack(const Cluster* cl) const
{
  // check if clusters have common tracks
  int lbi, lbj;
  for (int i = 0; i < 3; i++) {
    if ((lbi = getLabel(i)) < 0)
      break;
    for (int j = 0; j < 3; j++) {
      if ((lbj = cl->getLabel(j)) < 0)
        break;
      if (lbi == lbj)
        return kTRUE;
    }
  }
  return kFALSE;
}
