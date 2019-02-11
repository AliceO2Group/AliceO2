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

//////////////////////////////////////////////////////////////////////////////
//                          Class AliTrackPointArray                        //
//   This class contains the ESD track space-points which are used during   //
//   the alignment procedures. Each space-point consist of 3 coordinates    //
//   (and their errors) and the index of the sub-detector which contains    //
//   the space-point.                                                       //
//   cvetan.cheshkov@cern.ch 3/11/2005                                      //
//////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TGeoMatrix.h>
#include <TMatrixDSymEigen.h>

#include "AliTrackPointArray.h"

ClassImp(AliTrackPointArray)

//______________________________________________________________________________
AliTrackPointArray::AliTrackPointArray() :
  TObject(),
  fSorted(kFALSE),
  fNPoints(0),
  fX(0),
  fY(0),
  fZ(0),
  fCharge(0),
  fDriftTime(0),
  fChargeRatio(0),
  fClusterType(0),
  fIsExtra(0),
  fSize(0),
  fCov(0),
  fVolumeID(0)
{
}

//______________________________________________________________________________
AliTrackPointArray::AliTrackPointArray(Int_t npoints):
  TObject(),
  fSorted(kFALSE),
  fNPoints(npoints),
  fX(new Float_t[npoints]),
  fY(new Float_t[npoints]),
  fZ(new Float_t[npoints]),
  fCharge(new Float_t[npoints]),
  fDriftTime(new Float_t[npoints]),
  fChargeRatio(new Float_t[npoints]),
  fClusterType(new Int_t[npoints]),
  fIsExtra(new Bool_t[npoints]),
  fSize(6*npoints),
  fCov(new Float_t[fSize]),
  fVolumeID(new UShort_t[npoints])
{
  // Constructor
  //
  for (Int_t ip=0; ip<npoints;ip++){
    fX[ip]=0;
    fY[ip]=0;
    fZ[ip]=0;
    fCharge[ip]=0;
    fDriftTime[ip]=0;
    fChargeRatio[ip]=0;
    fClusterType[ip]=0;
    fIsExtra[ip]=kFALSE;
    fVolumeID[ip]=0;
    for (Int_t icov=0;icov<6; icov++)
      fCov[6*ip+icov]=0;
  }
}

//______________________________________________________________________________
AliTrackPointArray::AliTrackPointArray(const AliTrackPointArray &array):
  TObject(array),
  fSorted(array.fSorted),
  fNPoints(array.fNPoints),
  fX(new Float_t[fNPoints]),
  fY(new Float_t[fNPoints]),
  fZ(new Float_t[fNPoints]),
  fCharge(new Float_t[fNPoints]),
  fDriftTime(new Float_t[fNPoints]),
  fChargeRatio(new Float_t[fNPoints]),
  fClusterType(new Int_t[fNPoints]),
  fIsExtra(new Bool_t[fNPoints]),
  fSize(array.fSize),
  fCov(new Float_t[fSize]),
  fVolumeID(new UShort_t[fNPoints])
{
  // Copy constructor
  //
  memcpy(fX,array.fX,fNPoints*sizeof(Float_t));
  memcpy(fY,array.fY,fNPoints*sizeof(Float_t));
  memcpy(fZ,array.fZ,fNPoints*sizeof(Float_t));
  if (array.fCharge) {
    memcpy(fCharge,array.fCharge,fNPoints*sizeof(Float_t));
  } else {
    memset(fCharge, 0, fNPoints*sizeof(Float_t));
  }
  if (array.fDriftTime) {
    memcpy(fDriftTime,array.fDriftTime,fNPoints*sizeof(Float_t));
  } else {
    memset(fDriftTime, 0, fNPoints*sizeof(Float_t));
  }
  if (array.fChargeRatio) {
    memcpy(fChargeRatio,array.fChargeRatio,fNPoints*sizeof(Float_t));
  } else {
    memset(fChargeRatio, 0, fNPoints*sizeof(Float_t));
  }
  if (array.fClusterType) {
    memcpy(fClusterType,array.fClusterType,fNPoints*sizeof(Int_t));
  } else {
    memset(fClusterType, 0, fNPoints*sizeof(Int_t));
  }
  if (array.fIsExtra) {
    memcpy(fIsExtra,array.fIsExtra,fNPoints*sizeof(Bool_t));
  } else {
    memset(fIsExtra, 0, fNPoints*sizeof(Bool_t));
  }
  memcpy(fVolumeID,array.fVolumeID,fNPoints*sizeof(UShort_t));
  memcpy(fCov,array.fCov,fSize*sizeof(Float_t));
}

//_____________________________________________________________________________
AliTrackPointArray &AliTrackPointArray::operator =(const AliTrackPointArray& array)
{
  // assignment operator
  //
  if(this==&array) return *this;
  ((TObject *)this)->operator=(array);

  fSorted = array.fSorted;
  fNPoints = array.fNPoints;
  fSize = array.fSize;
  delete [] fX;
  fX = new Float_t[fNPoints];
  delete [] fY;
  fY = new Float_t[fNPoints];
  delete [] fZ;
  fZ = new Float_t[fNPoints];
  delete [] fCharge;
  fCharge = new Float_t[fNPoints];
  delete [] fDriftTime;
  fDriftTime = new Float_t[fNPoints];
  delete [] fChargeRatio;
  fChargeRatio = new Float_t[fNPoints];
  delete [] fClusterType;
  fClusterType = new Int_t[fNPoints];
  delete [] fIsExtra;
  fIsExtra = new Bool_t[fNPoints];
  delete [] fVolumeID;
  fVolumeID = new UShort_t[fNPoints];
  delete [] fCov;
  fCov = new Float_t[fSize];
  memcpy(fX,array.fX,fNPoints*sizeof(Float_t));
  memcpy(fY,array.fY,fNPoints*sizeof(Float_t));
  memcpy(fZ,array.fZ,fNPoints*sizeof(Float_t));
  memcpy(fCharge,array.fCharge,fNPoints*sizeof(Float_t));
  memcpy(fDriftTime,array.fDriftTime,fNPoints*sizeof(Float_t));
  memcpy(fChargeRatio,array.fChargeRatio,fNPoints*sizeof(Float_t));
  memcpy(fClusterType,array.fClusterType,fNPoints*sizeof(Int_t));
  memcpy(fIsExtra,array.fIsExtra,fNPoints*sizeof(Bool_t));
  memcpy(fVolumeID,array.fVolumeID,fNPoints*sizeof(UShort_t));
  memcpy(fCov,array.fCov,fSize*sizeof(Float_t));

  return *this;
}

//______________________________________________________________________________
AliTrackPointArray::~AliTrackPointArray()
{
  // Destructor
  //
  delete [] fX;
  delete [] fY;
  delete [] fZ;
  delete [] fCharge;
  delete [] fDriftTime;
  delete [] fChargeRatio;
  delete [] fClusterType;
  delete [] fIsExtra;
  delete [] fVolumeID;
  delete [] fCov;
}


//______________________________________________________________________________
Bool_t AliTrackPointArray::AddPoint(Int_t i, const AliTrackPoint *p)
{
  // Add a point to the array at position i
  //
  if (i >= fNPoints) return kFALSE;
  fX[i] = p->GetX();
  fY[i] = p->GetY();
  fZ[i] = p->GetZ();
  fCharge[i] = p->GetCharge();
  fDriftTime[i] = p->GetDriftTime();
  fChargeRatio[i]=p->GetChargeRatio();
  fClusterType[i]=p->GetClusterType();
  fIsExtra[i] = p->IsExtra();
  fVolumeID[i] = p->GetVolumeID();
  memcpy(&fCov[6*i],p->GetCov(),6*sizeof(Float_t));
  return kTRUE;
}


//______________________________________________________________________________
Bool_t AliTrackPointArray::GetPoint(AliTrackPoint &p, Int_t i) const
{
  // Get the point at position i
  //
  if (i >= fNPoints) return kFALSE;
  p.SetXYZ(fX[i],fY[i],fZ[i],&fCov[6*i]);
  p.SetVolumeID(fVolumeID[i]);
  p.SetCharge(fCharge ? fCharge[i] : 0);
  p.SetDriftTime(fDriftTime ? fDriftTime[i] : 0);
  p.SetChargeRatio(fChargeRatio ? fChargeRatio[i] : 0);
  p.SetClusterType(fClusterType ? fClusterType[i] : 0);
  p.SetExtra(fIsExtra ? fIsExtra[i] : kFALSE);
  return kTRUE;
}

//______________________________________________________________________________
Bool_t AliTrackPointArray::HasVolumeID(UShort_t volid) const
{
  // This method checks if the array
  // has at least one hit in the detector
  // volume defined by volid
  Bool_t check = kFALSE;
  for (Int_t ipoint = 0; ipoint < fNPoints; ipoint++)
    if (fVolumeID[ipoint] == volid) check = kTRUE;

  return check;
}

//______________________________________________________________________________
void AliTrackPointArray::Sort(Bool_t down)
{
  // Sort the array by the values of Y-coordinate of the track points.
  // The order is given by "down".
  // Optimized more for maintenance rather than for speed.
 
  if (fSorted) return;

  Int_t *index=new Int_t[fNPoints];
  AliTrackPointArray a(*this);
  TMath::Sort(fNPoints,a.GetY(),index,down);
 
  AliTrackPoint p;
  for (Int_t i = 0; i < fNPoints; i++) {
    a.GetPoint(p,index[i]);
    AddPoint(i,&p);
  }

  delete[] index;
  fSorted=kTRUE;
}

//_____________________________________________________________________________
void AliTrackPointArray::Print(Option_t *) const
{
  // Print the space-point coordinates and modules info
  for (int i=0;i<fNPoints;i++) {
    printf("#%3d VID %5d XYZ:%+9.3f/%+9.3f/%+9.3f |q: %+7.2f |DT: %+8.1f| ChR: %+.2e |Cl: %d %s\n",
	   i,fVolumeID[i],fX[i],fY[i],fZ[i],fCharge[i],fDriftTime[i],
	   fChargeRatio[i],fClusterType[i],fIsExtra[i] ? "|E":"");
  }
}


ClassImp(AliTrackPoint)

//______________________________________________________________________________
AliTrackPoint::AliTrackPoint() :
  TObject(),
  fX(0),
  fY(0),
  fZ(0),
  fCharge(0),
  fDriftTime(0),
  fChargeRatio(0),
  fClusterType(0),
  fIsExtra(kFALSE),
  fVolumeID(0)
{
  // Default constructor
  //
  memset(fCov,0,6*sizeof(Float_t));
}


//______________________________________________________________________________
AliTrackPoint::AliTrackPoint(Float_t x, Float_t y, Float_t z, const Float_t *cov, UShort_t volid, Float_t charge, Float_t drifttime,Float_t chargeratio, Int_t clutyp) :
  TObject(),
  fX(0),
  fY(0),
  fZ(0),
  fCharge(0),
  fDriftTime(0),
  fChargeRatio(0),
  fClusterType(0),
  fIsExtra(kFALSE),
  fVolumeID(0)
{
  // Constructor
  //
  SetXYZ(x,y,z,cov);
  SetCharge(charge);
  SetDriftTime(drifttime);
  SetChargeRatio(chargeratio);
  SetClusterType(clutyp);
  SetVolumeID(volid);
}

//______________________________________________________________________________
AliTrackPoint::AliTrackPoint(const Float_t *xyz, const Float_t *cov, UShort_t volid, Float_t charge, Float_t drifttime,Float_t chargeratio, Int_t clutyp) :
  TObject(),
  fX(0),
  fY(0),
  fZ(0),
  fCharge(0),
  fDriftTime(0),
  fChargeRatio(0),
  fClusterType(0),
  fIsExtra(kFALSE),
  fVolumeID(0)
{
  // Constructor
  //
  SetXYZ(xyz[0],xyz[1],xyz[2],cov);
  SetCharge(charge);
  SetDriftTime(drifttime);
  SetChargeRatio(chargeratio);
  SetVolumeID(volid);
  SetClusterType(clutyp);
}

//______________________________________________________________________________
AliTrackPoint::AliTrackPoint(const AliTrackPoint &p):
  TObject(p),
  fX(0),
  fY(0),
  fZ(0),
  fCharge(0),
  fDriftTime(0),
  fChargeRatio(0),
  fClusterType(0),
  fIsExtra(kFALSE),
  fVolumeID(0)
{
  // Copy constructor
  //
  SetXYZ(p.fX,p.fY,p.fZ,&(p.fCov[0]));
  SetCharge(p.fCharge);
  SetDriftTime(p.fDriftTime);
  SetChargeRatio(p.fChargeRatio);
  SetClusterType(p.fClusterType);
  SetExtra(p.fIsExtra);
  SetVolumeID(p.fVolumeID);
}

//_____________________________________________________________________________
AliTrackPoint &AliTrackPoint::operator =(const AliTrackPoint& p)
{
  // assignment operator
  //
  if(this==&p) return *this;
  ((TObject *)this)->operator=(p);

  SetXYZ(p.fX,p.fY,p.fZ,&(p.fCov[0]));
  SetCharge(p.fCharge);
  SetDriftTime(p.fDriftTime);
  SetChargeRatio(p.fChargeRatio);
  SetClusterType(p.fClusterType);
  SetExtra(p.fIsExtra);
  SetVolumeID(p.fVolumeID);

  return *this;
}

//______________________________________________________________________________
void AliTrackPoint::SetXYZ(Float_t x, Float_t y, Float_t z, const Float_t *cov)
{
  // Set XYZ coordinates and their cov matrix
  //
  fX = x;
  fY = y;
  fZ = z;
  if (cov)
    memcpy(fCov,cov,6*sizeof(Float_t));
}

//______________________________________________________________________________
void AliTrackPoint::SetXYZ(const Float_t *xyz, const Float_t *cov)
{
  // Set XYZ coordinates and their cov matrix
  //
  SetXYZ(xyz[0],xyz[1],xyz[2],cov);
}

//______________________________________________________________________________
void AliTrackPoint::SetCov(const Float_t *cov)
{
  // Set XYZ cov matrix
  //
  if (cov)
    memcpy(fCov,cov,6*sizeof(Float_t));
}

//______________________________________________________________________________
void AliTrackPoint::GetXYZ(Float_t *xyz, Float_t *cov) const
{
  xyz[0] = fX;
  xyz[1] = fY;
  xyz[2] = fZ;
  if (cov)
    memcpy(cov,fCov,6*sizeof(Float_t));
}

//______________________________________________________________________________
Float_t AliTrackPoint::GetResidual(const AliTrackPoint &p, Bool_t weighted) const
{
  // This method calculates the track to space-point residuals. The track
  // interpolation is also stored as AliTrackPoint. Using the option
  // 'weighted' one can calculate the residual either with or without
  // taking into account the covariance matrix of the space-point and
  // track interpolation. The second case the residual becomes a pull.

  Float_t res = 0;

  if (!weighted) {
    Float_t xyz[3],xyzp[3];
    GetXYZ(xyz);
    p.GetXYZ(xyzp);
    res = (xyz[0]-xyzp[0])*(xyz[0]-xyzp[0])+
          (xyz[1]-xyzp[1])*(xyz[1]-xyzp[1])+
          (xyz[2]-xyzp[2])*(xyz[2]-xyzp[2]);
  }
  else {
    Float_t xyz[3],xyzp[3];
    Float_t cov[6],covp[6];
    GetXYZ(xyz,cov);
    TMatrixDSym mcov(3);
    mcov(0,0) = cov[0]; mcov(0,1) = cov[1]; mcov(0,2) = cov[2];
    mcov(1,0) = cov[1]; mcov(1,1) = cov[3]; mcov(1,2) = cov[4];
    mcov(2,0) = cov[2]; mcov(2,1) = cov[4]; mcov(2,2) = cov[5];
    p.GetXYZ(xyzp,covp);
    TMatrixDSym mcovp(3);
    mcovp(0,0) = covp[0]; mcovp(0,1) = covp[1]; mcovp(0,2) = covp[2];
    mcovp(1,0) = covp[1]; mcovp(1,1) = covp[3]; mcovp(1,2) = covp[4];
    mcovp(2,0) = covp[2]; mcovp(2,1) = covp[4]; mcovp(2,2) = covp[5];
    TMatrixDSym msum = mcov + mcovp;
    msum.Invert();
    //    mcov.Print(); mcovp.Print(); msum.Print();
    if (msum.IsValid()) {
      for (Int_t i = 0; i < 3; i++)
	for (Int_t j = 0; j < 3; j++)
	  res += (xyz[i]-xyzp[i])*(xyz[j]-xyzp[j])*msum(i,j);
    }
  }

  return res;
}

//_____________________________________________________________________________
Bool_t AliTrackPoint::GetPCA(const AliTrackPoint &p, AliTrackPoint &out) const
{
  //
  // Get the intersection point between this point and
  // the point "p" belongs to.
  // The result is stored as a point 'out'
  // return kFALSE in case of failure.
  out.SetXYZ(0,0,0);

  TMatrixD t(3,1);
  t(0,0)=GetX();
  t(1,0)=GetY();
  t(2,0)=GetZ();
 
  TMatrixDSym tC(3);
  {
  const Float_t *cv=GetCov();
  tC(0,0)=cv[0]; tC(0,1)=cv[1]; tC(0,2)=cv[2];
  tC(1,0)=cv[1]; tC(1,1)=cv[3]; tC(1,2)=cv[4];
  tC(2,0)=cv[2]; tC(2,1)=cv[4]; tC(2,2)=cv[5];
  }

  TMatrixD m(3,1);
  m(0,0)=p.GetX();
  m(1,0)=p.GetY();
  m(2,0)=p.GetZ();
 
  TMatrixDSym mC(3);
  {
  const Float_t *cv=p.GetCov();
  mC(0,0)=cv[0]; mC(0,1)=cv[1]; mC(0,2)=cv[2];
  mC(1,0)=cv[1]; mC(1,1)=cv[3]; mC(1,2)=cv[4];
  mC(2,0)=cv[2]; mC(2,1)=cv[4]; mC(2,2)=cv[5];
  }

  TMatrixDSym tmW(tC);
  tmW+=mC;
  tmW.Invert();
  if (!tmW.IsValid()) return kFALSE; 

  TMatrixD mW(tC,TMatrixD::kMult,tmW);
  TMatrixD tW(mC,TMatrixD::kMult,tmW);

  TMatrixD mi(mW,TMatrixD::kMult,m);
  TMatrixD ti(tW,TMatrixD::kMult,t);
  ti+=mi;

  TMatrixD iC(tC,TMatrixD::kMult,tmW);
  iC*=mC;

  out.SetXYZ(ti(0,0),ti(1,0),ti(2,0));
  UShort_t id=p.GetVolumeID();
  out.SetVolumeID(id);

  return kTRUE;
}

//______________________________________________________________________________
Float_t AliTrackPoint::GetAngle() const
{
  // The method uses the covariance matrix of
  // the space-point in order to extract the
  // orientation of the detector plane.
  // The rotation in XY plane only is calculated.

  Float_t phi= TMath::ATan2(TMath::Sqrt(fCov[0]),TMath::Sqrt(fCov[3]));
  if (fCov[1] > 0) {
    phi = TMath::Pi() - phi;
    if ((fY-fX) < 0) phi += TMath::Pi();
  }
  else {
    if ((fX+fY) < 0) phi += TMath::Pi();
  }

  return phi;

}

//______________________________________________________________________________
Bool_t AliTrackPoint::GetRotMatrix(TGeoRotation& rot) const
{
  // Returns the orientation of the
  // sensitive layer (using cluster
  // covariance matrix).
  // Assumes that cluster has errors only in the layer's plane.
  // Return value is kTRUE in case of success.

  TMatrixDSym mcov(3);
  {
    const Float_t *cov=GetCov();
    mcov(0,0)=cov[0]; mcov(0,1)=cov[1]; mcov(0,2)=cov[2];
    mcov(1,0)=cov[1]; mcov(1,1)=cov[3]; mcov(1,2)=cov[4];
    mcov(2,0)=cov[2]; mcov(2,1)=cov[4]; mcov(2,2)=cov[5];
  }

  TMatrixDSymEigen eigen(mcov);
  TMatrixD eigenMatrix = eigen.GetEigenVectors();

  rot.SetMatrix(eigenMatrix.GetMatrixArray());

  return kTRUE;
}


//_____________________________________________________________________________
AliTrackPoint& AliTrackPoint::Rotate(Float_t alpha) const
{
  // Transform the space-point coordinates
  // and covariance matrix from global to
  // local (detector plane) coordinate system
  // XY plane rotation only

  static AliTrackPoint p;
  p = *this;

  Float_t xyz[3],cov[6];
  GetXYZ(xyz,cov);

  Float_t sin = TMath::Sin(alpha), cos = TMath::Cos(alpha);

  Float_t newxyz[3],newcov[6];
  newxyz[0] = cos*xyz[0] + sin*xyz[1];
  newxyz[1] = cos*xyz[1] - sin*xyz[0];
  newxyz[2] = xyz[2];

  newcov[0] = cov[0]*cos*cos+
            2*cov[1]*sin*cos+
              cov[3]*sin*sin;
  newcov[1] = cov[1]*(cos*cos-sin*sin)+
             (cov[3]-cov[0])*sin*cos;
  newcov[2] = cov[2]*cos+
              cov[4]*sin;
  newcov[3] = cov[0]*sin*sin-
            2*cov[1]*sin*cos+
              cov[3]*cos*cos;
  newcov[4] = cov[4]*cos-
              cov[2]*sin;
  newcov[5] = cov[5];

  p.SetXYZ(newxyz,newcov);
  p.SetVolumeID(GetVolumeID());

  return p;
}

//_____________________________________________________________________________
AliTrackPoint& AliTrackPoint::MasterToLocal() const
{
  // Transform the space-point coordinates
  // and the covariance matrix from the
  // (master) to the local (tracking)
  // coordinate system

  Float_t alpha = GetAngle();
  return Rotate(alpha);
}

//_____________________________________________________________________________
void AliTrackPoint::Print(Option_t *) const
{
  // Print the space-point coordinates and
  // covariance matrix

  printf("VolumeID=%d\n", GetVolumeID());
  printf("X = %12.6f    Tx = %12.6f%12.6f%12.6f\n", fX, fCov[0], fCov[1], fCov[2]);
  printf("Y = %12.6f    Ty = %12.6f%12.6f%12.6f\n", fY, fCov[1], fCov[3], fCov[4]);
  printf("Z = %12.6f    Tz = %12.6f%12.6f%12.6f\n", fZ, fCov[2], fCov[4], fCov[5]);
  printf("Charge = %f\n", fCharge);
  printf("Drift Time = %f\n", fDriftTime);
  printf("Charge Ratio  = %f\n", fChargeRatio);
  printf("Cluster Type  = %d\n", fClusterType);
  if(fIsExtra) printf("This is an extra point\n");

}


//________________________________
void AliTrackPoint::SetAlignCovMatrix(const TMatrixDSym& alignparmtrx){
  // Add the uncertainty on the cluster position due to alignment
  // (using the 6x6 AliAlignObj Cov. Matrix alignparmtrx) to the already
  // present Cov. Matrix 

  TMatrixDSym cov(3);
  TMatrixD coval(3,3);
  TMatrixD jacob(3,6);
  Float_t newcov[6];

  cov(0,0)=fCov[0];
  cov(1,0)=cov(0,1)=fCov[1];
  cov(2,0)=cov(0,2)=fCov[2];
  cov(1,1)=fCov[3];
  cov(2,1)=cov(1,2)=fCov[4];
  cov(2,2)=fCov[5];
  
  jacob(0,0) = 1;      jacob(1,0) = 0;       jacob(2,0) = 0;
  jacob(0,1) = 0;      jacob(1,1) = 1;       jacob(2,1) = 0;
  jacob(0,2) = 0;      jacob(1,2) = 0;       jacob(2,2) = 1;
  jacob(0,3) = 0;      jacob(1,3) =-fZ;      jacob(2,3) = fY;
  jacob(0,4) = fZ;     jacob(1,4) = 0;       jacob(2,4) =-fX;
  jacob(0,5) = -fY;    jacob(1,5) = fX;      jacob(2,5) = 0;
  
  TMatrixD jacobT=jacob.T();jacob.T();
  
  coval=jacob*alignparmtrx*jacobT+cov;


  newcov[0]=coval(0,0);
  newcov[1]=coval(1,0);
  newcov[2]=coval(2,0);
  newcov[3]=coval(1,1);
  newcov[4]=coval(2,1);
  newcov[5]=coval(2,2);
 
  SetXYZ(fX,fY,fZ,newcov);

}


