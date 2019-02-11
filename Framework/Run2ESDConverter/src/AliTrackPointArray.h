#ifndef ALITRACKPOINTARRAY_H
#define ALITRACKPOINTARRAY_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////////
//                          Class AliTrackPoint                             //
//   This class represent a single track space-point.                       //
//   It is used to access the points array defined in AliTrackPointArray.   //
//   Note that the space point coordinates are given in the global frame.   //
//                                                                          //
//   cvetan.cheshkov@cern.ch 3/11/2005                                      //
//////////////////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <TMatrixDSym.h>
#include "Rtypes.h"

class TGeoRotation;

class AliTrackPoint : public TObject {

 public:

  AliTrackPoint();
  AliTrackPoint(Float_t x, Float_t y, Float_t z, const Float_t *cov, UShort_t volid, Float_t charge = 0, Float_t drifttime = 0,Float_t chargeratio = 0, Int_t clutype = 0);
  AliTrackPoint(const Float_t *xyz, const Float_t *cov, UShort_t volid, Float_t charge = 0, Float_t drifttime = 0,Float_t chargeratio = 0, Int_t clutype=0);
  AliTrackPoint(const AliTrackPoint &p);
  AliTrackPoint& operator= (const AliTrackPoint& p);
  virtual ~AliTrackPoint() {}

  // Multiplication with TGeoMatrix and distance between points (chi2) to be implemented

  void     SetXYZ(Float_t x, Float_t y, Float_t z, const Float_t *cov = 0);
  void     SetXYZ(const Float_t *xyz, const Float_t *cov = 0);
  void     SetCov(const Float_t *cov);
  void     SetVolumeID(UShort_t volid) { fVolumeID = volid; }
  void     SetCharge(Float_t charge) { fCharge = charge; }
  void     SetDriftTime(Float_t time) { fDriftTime = time; }
  void     SetChargeRatio(Float_t ratio) {  fChargeRatio= ratio; }
  void     SetClusterType(Int_t clutype) {  fClusterType= clutype; }
  void     SetExtra(Bool_t flag=kTRUE) { fIsExtra = flag; }

  Float_t  GetX() const { return fX; }
  Float_t  GetY() const { return fY; }
  Float_t  GetZ() const { return fZ; }
  void     GetXYZ(Float_t *xyz, Float_t *cov = 0) const;
  const Float_t *GetCov() const { return &fCov[0]; }
  UShort_t GetVolumeID() const { return fVolumeID; }
  Float_t  GetCharge() const { return fCharge; }
  Float_t  GetDriftTime() const { return fDriftTime;}
  Float_t  GetChargeRatio() const { return fChargeRatio;}
  Int_t    GetClusterType() const { return fClusterType;}
  Bool_t   IsExtra() const { return fIsExtra;}

  Float_t  GetResidual(const AliTrackPoint &p, Bool_t weighted = kFALSE) const;
  Bool_t   GetPCA(const AliTrackPoint &p, AliTrackPoint &out) const;

  Float_t  GetAngle() const;
  Bool_t   GetRotMatrix(TGeoRotation& rot) const;
  void SetAlignCovMatrix(const TMatrixDSym& alignparmtrx);

  AliTrackPoint& Rotate(Float_t alpha) const;
  AliTrackPoint& MasterToLocal() const;

  void     Print(Option_t *) const;

 private:

  Float_t  fX;        // X coordinate
  Float_t  fY;        // Y coordinate
  Float_t  fZ;        // Z coordinate
  Float_t  fCharge;   // Cluster charge in arbitrary units
  Float_t  fDriftTime;// Drift time in SDD (in ns)
  Float_t  fChargeRatio; // Charge ratio in SSD 
  Int_t    fClusterType; // Cluster Type (encoded info on size and shape)
  Float_t  fCov[6];   // Cov matrix
  Bool_t   fIsExtra;  // attached by tracker but not used in fit
  UShort_t fVolumeID; // Volume ID

  ClassDef(AliTrackPoint,7)
};

//////////////////////////////////////////////////////////////////////////////
//                          Class AliTrackPointArray                        //
//   This class contains the ESD track space-points which are used during   //
//   the alignment procedures. Each space-point consist of 3 coordinates    //
//   (and their errors) and the index of the sub-detector which contains    //
//   the space-point.                                                       //
//   cvetan.cheshkov@cern.ch 3/11/2005                                      //
//////////////////////////////////////////////////////////////////////////////

class AliTrackPointArray : public TObject {

 public:

  enum {kTOFBugFixed=BIT(14)};

  AliTrackPointArray();
  AliTrackPointArray(Int_t npoints);
  AliTrackPointArray(const AliTrackPointArray &array);
  AliTrackPointArray& operator= (const AliTrackPointArray& array);
  virtual ~AliTrackPointArray();

  //  Bool_t    AddPoint(Int_t i, AliCluster *cl, UShort_t volid);
  Bool_t    AddPoint(Int_t i, const AliTrackPoint *p);

  Int_t     GetNPoints() const { return fNPoints; }
  Int_t     GetCovSize() const { return fSize; }
  Bool_t    GetPoint(AliTrackPoint &p, Int_t i) const;
  // Getters for fast access to the coordinate arrays
  const Float_t*  GetX() const { return &fX[0]; }
  const Float_t*  GetY() const { return &fY[0]; }
  const Float_t*  GetZ() const { return &fZ[0]; }
  const Float_t*  GetCharge() const { return &fCharge[0]; }
  const Float_t*  GetDriftTime() const { return &fDriftTime[0]; }
  const Float_t*  GetChargeRatio() const { return &fChargeRatio[0]; }
  const Int_t*    GetClusterType() const { return &fClusterType[0]; }
  const Bool_t*   GetExtra() const { return &fIsExtra[0]; }
  const Float_t*  GetCov() const { return &fCov[0]; }
  const UShort_t* GetVolumeID() const { return &fVolumeID[0]; }

  Bool_t    HasVolumeID(UShort_t volid) const;
  void      Print(Option_t *) const;

  void Sort(Bool_t down=kTRUE);

 private:
  Bool_t fSorted;        // Sorted flag

  Int_t     fNPoints;    // Number of space points
  Float_t   *fX;         //[fNPoints] Array with space point X coordinates
  Float_t   *fY;         //[fNPoints] Array with space point Y coordinates
  Float_t   *fZ;         //[fNPoints] Array with space point Z coordinates
  Float_t   *fCharge;    //[fNPoints] Array with clusters charge
  Float_t   *fDriftTime; //[fNPoints] Array with drift times
  Float_t   *fChargeRatio; //[fNPoints] Array with charge ratio
  Int_t     *fClusterType; //[fNPoints] Array with cluster type
  Bool_t    *fIsExtra;   //[fNPoints] Array with extra flags
  Int_t     fSize;       // Size of array with cov matrices = 6*N of points
  Float_t   *fCov;       //[fSize] Array with space point coordinates cov matrix
  UShort_t  *fVolumeID;  //[fNPoints] Array of space point volume IDs

  ClassDef(AliTrackPointArray,7)
};

#endif

