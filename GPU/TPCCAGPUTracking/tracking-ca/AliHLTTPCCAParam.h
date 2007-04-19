//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAPARAM_H
#define ALIHLTTPCCAPARAM_H

#include "Rtypes.h"
#include "Riostream.h"

/**
 * @class ALIHLTTPCCAParam
 * parameters of the AliHLTTPCCATracker, including geometry information
 * and some reconstructon constants.
 *
 * The class is under construction.
 *
 */
class AliHLTTPCCAParam
{
 public:

  AliHLTTPCCAParam();
  virtual ~AliHLTTPCCAParam(){;}

  void Initialize( Int_t iSlice, Int_t nRows, Float_t rowX[],
		   Float_t alpha, Float_t dAlpha,
		   Float_t rMin, Float_t rMax, Float_t zMin, Float_t zMax,
		   Float_t padPitch, Float_t zSigma, Float_t bz );
  void Update();
  
  void Slice2Global( Float_t x, Float_t y,  Float_t z, 
		     Float_t *X, Float_t *Y,  Float_t *Z ) const;
  void Global2Slice( Float_t x, Float_t y,  Float_t z, 
		     Float_t *X, Float_t *Y,  Float_t *Z ) const;
  Int_t &ISlice(){ return fISlice;}
  Int_t &NRows(){ return fNRows;}
  
  Float_t &RowX( Int_t iRow ){ return fRowX[iRow]; }
  
  Float_t &Alpha(){ return fAlpha;}
  Float_t &DAlpha(){ return fDAlpha;}
  Float_t &CosAlpha(){ return fCosAlpha;}
  Float_t &SinAlpha(){ return fSinAlpha;}
  Float_t &AngleMin(){ return fAngleMin;}
  Float_t &AngleMax(){ return fAngleMax;}
  Float_t &RMin(){ return fRMin;}
  Float_t &RMax(){ return fRMax;}
  Float_t &ZMin(){ return fZMin;}
  Float_t &ZMax(){ return fZMax;}
  Float_t &ErrZ(){ return fErrZ;}
  Float_t &ErrX(){ return fErrX;}
  Float_t &ErrY(){ return fErrY;}
  Float_t &Bz(){ return fBz;}

  Float_t &TrackConnectionFactor(){ return fTrackConnectionFactor; }
  Float_t &TrackChiCut() { return fTrackChiCut; }
  Float_t &TrackChi2Cut(){ return fTrackChi2Cut; }
  Int_t   &MaxTrackMatchDRow(){ return fMaxTrackMatchDRow; }
  Float_t &YErrorCorrection(){ return fYErrorCorrection; }
  Float_t &ZErrorCorrection(){ return fZErrorCorrection; }
  Float_t &CellConnectionAngleXY(){ return fCellConnectionAngleXY; }
  Float_t &CellConnectionAngleXZ(){ return fCellConnectionAngleXZ; }

  Float_t GetClusterError2(Int_t yz, Int_t type, Float_t z, Float_t angle );

  void WriteSettings( ostream &out );
  void ReadSettings( istream &in );

 protected:

  Int_t fISlice; // slice number
  Int_t fNRows; // number of rows

  Float_t fAlpha, fDAlpha; // slice angle and angular size
  Float_t fCosAlpha, fSinAlpha;// sign and cosine of the slice angle
  Float_t fAngleMin, fAngleMax; // minimal and maximal angle
  Float_t fRMin, fRMax;// slice R range
  Float_t fZMin, fZMax;// slice Z range
  Float_t fErrX, fErrY, fErrZ;// default cluster errors
  Float_t fPadPitch; // pad pitch 
  Float_t fBz;       // magnetic field value (only constant field can be used)

  Float_t fYErrorCorrection;// correction factor for Y error of input clusters
  Float_t fZErrorCorrection;// correction factor for Z error of input clusters

  Float_t fCellConnectionAngleXY; // max phi angle between connected cells
  Float_t fCellConnectionAngleXZ; // max psi angle between connected cells
  Int_t   fMaxTrackMatchDRow;// maximal jump in TPC row for connecting track segments
  Float_t fTrackConnectionFactor; // allowed distance in Chi^2/3.5 for neighbouring tracks
  Float_t fTrackChiCut; // cut for track Sqrt(Chi2/NDF);
  Float_t fTrackChi2Cut;// cut for track Chi^2/NDF

  Float_t fRowX[200];// X-coordinate of rows
  Float_t fParamS0Par[2][3][7];    // cluster error parameterization coeficients

  ClassDef(AliHLTTPCCAParam,1);
};


#endif
