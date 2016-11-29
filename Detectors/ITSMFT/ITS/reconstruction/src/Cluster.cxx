/// \file Cluster.cxx
/// \brief Cluster structure for ITS clusters

#include "ITSReconstruction/Cluster.h"
#include "ITSBase/GeometryTGeo.h"
#include "FairLogger.h"

#include <TGeoMatrix.h>
#include <TString.h>

#include <cstdlib>

using namespace TMath;
using namespace AliceO2::ITS;

ClassImp(AliceO2::ITS::Cluster)



GeometryTGeo *Cluster::fgGeom = 0;
UInt_t               Cluster::fgMode = 0;

//_____________________________________________________
Cluster::Cluster():
    fX(0),fY(0),fZ(0)
  , fSigmaY2(0),fSigmaZ2(0),fSigmaYZ(0)
  , fVolumeId(0)
  , fCharge(0)
  , fRecoInfo(0)
  , fNxNzN(0)
#ifdef _ClusterTopology_
  ,fPatternNRows(0)
  ,fPatternNCols(0)
  ,fPatternMinRow(0)
  ,fPatternMinCol(0)
#endif
{
  fTracks[0]=-1;
  fTracks[1]=-1;
  fTracks[2]=-1;
  // default constructor
#ifdef _ClusterTopology_
  memset(fPattern,0,kMaxPatternBytes*sizeof(UChar_t));
#endif

}

//_____________________________________________________
Cluster::~Cluster()
{
  // default destructor
}

//_____________________________________________________
Cluster::Cluster(const Cluster& cluster)
  : FairTimeStamp(cluster)
  , fX(cluster.fX),fY(cluster.fY),fZ(cluster.fZ)
  , fSigmaY2(cluster.fSigmaY2),fSigmaZ2(cluster.fSigmaZ2)
  , fSigmaYZ(cluster.fSigmaYZ)
  , fVolumeId(cluster.fVolumeId)
  , fCharge(cluster.fCharge)
  ,fRecoInfo(cluster.fRecoInfo)
  ,fNxNzN(cluster.fNxNzN)
#ifdef _ClusterTopology_
  ,fPatternNRows(cluster.fPatternNRows)
  ,fPatternNCols(cluster.fPatternNCols)
  ,fPatternMinRow(cluster.fPatternMinRow)
  ,fPatternMinCol(cluster.fPatternMinCol)
#endif
{
  fTracks[0]=cluster.fTracks[0];
  fTracks[1]=cluster.fTracks[1];
  fTracks[2]=cluster.fTracks[2];
  // copy constructor
#ifdef _ClusterTopology_
  memcpy(fPattern,cluster.fPattern,kMaxPatternBytes*sizeof(UChar_t));
#endif
}

//______________________________________________________________________________
const TGeoHMatrix*  Cluster::GetTracking2LocalMatrix() const
{
  // get tracking to local matrix (sensor!!!)
  return (TGeoHMatrix*)fgGeom->getMatrixT2L(GetVolumeId());
}

//______________________________________________________________________________
TGeoHMatrix* Cluster::GetMatrix(Bool_t ) const
{
  // get chip matrix (sensor!)
  return (TGeoHMatrix*)fgGeom->getMatrixSensor(GetVolumeId());
}

//______________________________________________________________________________
void Cluster::Print(Option_t* option) const
{
  // Print cluster information.
  TString str = option; 
  str.ToLower();
  printf("Cl.in mod %5d, nx:%3d nz:%3d n:%d |Err^2:%.3e %.3e %+.3e |",GetVolumeId(),GetNx(),GetNz(),
	 GetNPix(),GetSigmaY2(),GetSigmaZ2(),GetSigmaYZ());
  printf("XYZ: (%+.4e %+.4e %+.4e ",GetX(),GetY(),GetZ());
  if      (IsFrameLoc()) printf("LOC)");
  else if (IsFrameGlo()) printf("GLO)");
  else if (IsFrameTrk()) printf("TRK)");
  if (str.Contains("glo") && !IsFrameGlo() && fgGeom) {
    Float_t g[3];
    GetGlobalXYZ(g);
    printf(" (%+.4e %+.4e %+.4e GLO)",g[0],g[1],g[2]);
  }
  printf(" MClb:");
  for (int i=0;i<3;i++) printf(" %5d",GetLabel(i));
  if (TestBit(kSplit)) printf(" Spl");
  printf("\n");
  //
#ifdef _ClusterTopology_
  if (str.Contains("p")) { // print pattern
    int nr = GetPatternRowSpan();
    int nc = GetPatternColSpan();    
    printf("Pattern: %d rows from %d",nr,fPatternMinRow);
    if (IsPatternRowsTruncated()) printf("(truncated)");
    printf(", %d columns from %d",nc,fPatternMinCol);
    if (IsPatternColsTruncated()) printf("(truncated)");
    printf("\n");
    for (int ir=0;ir<nr;ir++) {
      for (int ic=0;ic<nc;ic++) printf("%c",TestPixel(ir,ic) ? '+':'-');
      printf("\n");
    }
  }
#endif
  //
}

#ifdef _ClusterTopology_
//______________________________________________________________________________
void Cluster::ResetPattern()
{
  // reset pixels pattern
  memset(fPattern,0,kMaxPatternBytes*sizeof(UChar_t));
}

//______________________________________________________________________________
Bool_t Cluster::TestPixel(UShort_t row,UShort_t col) const
{
  // test if pixel at relative row,col is fired
  int nbits = row*GetPatternColSpan()+col;
  if (nbits>=kMaxPatternBits) return kFALSE;
  int bytn = nbits>>3; // 1/8  
  int bitn = nbits%8;
  return (fPattern[bytn]&(0x1<<bitn))!=0;
  //
}

//______________________________________________________________________________
void Cluster::SetPixel(UShort_t row,UShort_t col, Bool_t fired) 
{
  // test if pixel at relative row,col is fired
  int nbits = row*GetPatternColSpan()+col;
  if (nbits>=kMaxPatternBits) return;
  int bytn = nbits>>3; // 1/8  
  int bitn = nbits%8;
  if (nbits>=kMaxPatternBits) exit(1);
  if (fired) fPattern[bytn] |= (0x1<<bitn);
  else       fPattern[bytn] &= (0xff ^ (0x1<<bitn));
  //
}

//______________________________________________________________________________
void Cluster::SetPatternRowSpan(UShort_t nr, Bool_t truncated)
{
  // set pattern span in rows, flag if truncated
  fPatternNRows = kSpanMask&nr;
  if (truncated) fPatternNRows |= kTruncateMask; 
}

//______________________________________________________________________________
void Cluster::SetPatternColSpan(UShort_t nc, Bool_t truncated)
{
  // set pattern span in columns, flag if truncated
  fPatternNCols = kSpanMask&nc;
  if (truncated) fPatternNCols |= kTruncateMask; 
}

#endif

//______________________________________________________________________________
Bool_t Cluster::GetGlobalXYZ(Float_t xyz[3]) const
{
  // Get the global coordinates of the cluster
  // All the needed information is taken only
  // from TGeo (single precision).
  if (IsFrameGlo()) {
    xyz[0] = GetX();
    xyz[1] = GetY();
    xyz[2] = GetZ();
    return kTRUE;
  }
  //
  Double_t lxyz[3] = {0, 0, 0};
  if (IsFrameTrk()) {
    const TGeoHMatrix *mt = GetTracking2LocalMatrix();
    if (!mt) return kFALSE;
    Double_t txyz[3] = {GetX(), GetY(), GetZ()};
    mt->LocalToMaster(txyz,lxyz);
  }
  else {
    lxyz[0] = GetX(); lxyz[1] = GetY(); lxyz[2] = GetZ();
  }
  //
  TGeoHMatrix *ml = GetMatrix();
  if (!ml) return kFALSE;
  Double_t gxyz[3] = {0, 0, 0};
  ml->LocalToMaster(lxyz,gxyz);
  xyz[0] = gxyz[0]; xyz[1] = gxyz[1]; xyz[2] = gxyz[2];
  return kTRUE;
}

//______________________________________________________________________________
Bool_t Cluster::GetGlobalCov(Float_t cov[6]) const
{
  // Get the global covariance matrix of the cluster coordinates
  // All the needed information is taken only
  // from TGeo.
  // Note: regardless on in which frame the coordinates are, the errors are always in tracking frame
  //

  const TGeoHMatrix *mt = GetTracking2LocalMatrix();
  if (!mt) return kFALSE;

  TGeoHMatrix *ml = GetMatrix();
  if (!ml) return kFALSE;

  TGeoHMatrix m;
  Double_t tcov[9] = { 0, 0, 0, 0, fSigmaY2, fSigmaYZ, 0, fSigmaYZ, fSigmaZ2 };
  m.SetRotation(tcov);
  m.Multiply(&mt->Inverse());
  m.Multiply(&ml->Inverse());
  m.MultiplyLeft(mt);
  m.MultiplyLeft(ml);
  Double_t *ncov = m.GetRotationMatrix();
  cov[0] = ncov[0]; cov[1] = ncov[1]; cov[2] = ncov[2];
  cov[3] = ncov[4]; cov[4] = ncov[5];
  cov[5] = ncov[8];

  return kTRUE;
}

//______________________________________________________________________________
Bool_t Cluster::GetXRefPlane(Float_t &xref) const
{
  // Get the distance between the origin and the ref.plane.
  // All the needed information is taken only from TGeo.

  const TGeoHMatrix *mt = GetTracking2LocalMatrix();
  if (!mt) return kFALSE;

  TGeoHMatrix *ml = GetMatrix();
  if (!ml) return kFALSE;

  TGeoHMatrix m = *mt;
  m.MultiplyLeft(ml);

  xref = -(m.Inverse()).GetTranslation()[0];
  return kTRUE;
}

//______________________________________________________________________________
void Cluster::GoToFrameGlo()
{
  // convert to global frame
  if (IsFrameGlo()) return;
  double loc[3],glo[3];
  //
  if (IsFrameTrk()) {
    double curr[3]={GetX(),GetY(),GetZ()};
    GetTracking2LocalMatrix()->LocalToMaster(curr,loc);
    ResetBit(kFrameTrk);
  }
  else {
    loc[0] = GetX(); loc[1] = GetY(); loc[2] = GetZ();
    ResetBit(kFrameLoc);
  }
  GetMatrix()->LocalToMaster(loc,glo);
  SetX(glo[0]);  
  SetY(glo[1]); 
  SetZ(glo[2]);
  SetBit(kFrameGlo);
  //
}

//______________________________________________________________________________
void Cluster::GoToFrameLoc()
{
  // convert to local frame
  if (IsFrameLoc()) return;
  //
  double loc[3],glo[3];
  if (IsFrameTrk()) {
    double curr[3]={GetX(),GetY(),GetZ()};
    GetTracking2LocalMatrix()->LocalToMaster(curr,loc);
    ResetBit(kFrameTrk);
  }
  else {
    glo[0] = GetX(); glo[1] = GetY(); glo[2] = GetZ();
    GetMatrix()->MasterToLocal(glo,loc);
    ResetBit(kFrameLoc);
  }
  SetBit(kFrameLoc);
  SetX(loc[0]); 
  SetY(loc[1]); 
  SetZ(loc[2]);
  //
}

//______________________________________________________________________________
void Cluster::GetLocalXYZ(Float_t xyz[3]) const
{
  // get local coordinates
  if (IsFrameLoc()) {
    xyz[0] = GetX(); xyz[1] = 0; xyz[2] = GetZ();
    return;
  }
  double loc[3],glo[3];
  if (IsFrameTrk()) {
    double curr[3]={GetX(),GetY(),GetZ()};
    GetTracking2LocalMatrix()->LocalToMaster(curr,loc);
  }
  else {
    glo[0] = GetX(); glo[1] = GetY(); glo[2] = GetZ();
    GetMatrix()->MasterToLocal(glo,loc);
  }
  for (int i=3;i--;) xyz[i] = loc[i];
  //
}

//______________________________________________________________________________
void Cluster::GoToFrameTrk()
{
  // convert to tracking frame
  if (IsFrameTrk()) return;
  //
  double loc[3],trk[3];
  if (IsFrameGlo()) {
    double glo[3]={GetX(),GetY(),GetZ()};
    GetMatrix()->MasterToLocal(glo,loc);
    ResetBit(kFrameGlo);
  }
  else {
    loc[0] = GetX(); loc[1] = GetY(); loc[2] = GetZ();
    ResetBit(kFrameLoc);    
  }
  // now in local frame
  GetTracking2LocalMatrix()->MasterToLocal(loc,trk);
  SetBit(kFrameTrk);
  SetX(trk[0]);  
  SetY(trk[1]); 
  SetZ(trk[2]);
  //
}

//______________________________________________________________________________
void Cluster::GetTrackingXYZ(Float_t xyz[3]) const
{
  // convert to tracking frame
  if (IsFrameTrk()) {
    xyz[0] = GetX(); xyz[1] = GetY(); xyz[2] = GetZ();
    return;
  }
  //
  double loc[3],trk[3];
  if (IsFrameGlo()) {
    double glo[3]={GetX(),GetY(),GetZ()};
    GetMatrix()->MasterToLocal(glo,loc);
  }
  else {
    loc[0] = GetX(); loc[1] = GetY(); loc[2] = GetZ();
  }
  // now in local frame
  GetTracking2LocalMatrix()->MasterToLocal(loc,trk);
  for (int i=3;i--;) xyz[i] = trk[i];
  //
}

//______________________________________________________________________________
Int_t Cluster::Compare(const TObject* obj)  const
{
  // compare clusters accodring to specific mode
  const Cluster* px = (const Cluster*)obj;
  float xyz[3],xyz1[3];
  if (fgMode & kSortIdLocXZ) { // sorting in local frame
    if (GetVolumeId()==px->GetVolumeId()) {
      GetLocalXYZ(xyz);
      px->GetLocalXYZ(xyz1);
      if (xyz[0]<xyz1[0]) return -1; // sort in X
      if (xyz[0]>xyz1[0]) return  1;
      if (xyz[2]<xyz1[2]) return -1; // then in Z
      if (xyz[2]>xyz1[2]) return  1;
      return 0;
    }
    return int(GetVolumeId())-int(px->GetVolumeId());
  }
  if (fgMode & kSortIdTrkYZ) { // sorting in tracking frame
    if (GetVolumeId()==px->GetVolumeId()) {
      GetTrackingXYZ(xyz);
      px->GetTrackingXYZ(xyz1);
      if (xyz[1]<xyz1[1]) return -1; // sort in Y
      if (xyz[1]>xyz1[1]) return  1;
      if (xyz[2]<xyz1[2]) return -1; // then in Z
      if (xyz[2]>xyz1[2]) return  1;
      return 0;    
    }
    return int(GetVolumeId())-int(px->GetVolumeId());    
  }
  LOG(FATAL)<<"Unknown mode for sorting: "<<fgMode<<FairLogger::endl;
  return 0;
}

//______________________________________________________________________________
Bool_t Cluster::IsEqual(const TObject* obj)  const
{
  // compare clusters accodring to specific mode
  const Cluster* px = (const Cluster*)obj;
  const Float_t kTol = 1e-5;
  float xyz[3],xyz1[3];
  if (fgMode & kSortIdLocXZ) { // sorting in local frame
    if (GetVolumeId()!=px->GetVolumeId()) return kFALSE;
    GetLocalXYZ(xyz);
    px->GetLocalXYZ(xyz1);
    return (Abs(xyz[0]-xyz1[0])<kTol && Abs(xyz[2]-xyz1[2])<kTol) ? kTRUE : kFALSE;
  }
  if (fgMode & kSortIdTrkYZ) { // sorting in tracking frame
    if (GetVolumeId()!=px->GetVolumeId()) return kFALSE;
    GetTrackingXYZ(xyz);
    px->GetTrackingXYZ(xyz1);
    return (Abs(xyz[1]-xyz1[1])<kTol && Abs(xyz[2]-xyz1[2])<kTol) ? kTRUE : kFALSE;
  }
  LOG(FATAL)<<"Unknown mode for sorting: "<<fgMode<<FairLogger::endl;
  return kFALSE;
}

//______________________________________________________________________________
Bool_t Cluster::HasCommonTrack(const Cluster* cl) const
{
  // check if clusters have common tracks
  int lbi,lbj;
  for (int i=0;i<3;i++) {
    if ((lbi=GetLabel(i))<0) break;
    for (int j=0;j<3;j++) {
      if ((lbj=cl->GetLabel(j))<0) break;
      if (lbi==lbj) return kTRUE;
    }
  }
  return kFALSE;
}
