//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAPARAM_H
#define ALIHLTTPCCAPARAM_H

#include "Rtypes.h"

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

  void Initialize( Int_t iSlice, Int_t nRows, Double_t rowX[],
		   Double_t alpha, Double_t dAlpha,
		   Double_t rMin, Double_t rMax, Double_t zMin, Double_t zMax,
		   Double_t padPitch, Double_t zSigma, Double_t bz );
  void Update();
  
  void Slice2Global( Double_t x, Double_t y,  Double_t z, 
		     Double_t *X, Double_t *Y,  Double_t *Z ) const;
  void Global2Slice( Double_t x, Double_t y,  Double_t z, 
		     Double_t *X, Double_t *Y,  Double_t *Z ) const;
  Int_t &ISlice(){ return fISlice;}
  Int_t &NRows(){ return fNRows;}
  
  Double_t &RowX( Int_t iRow ){ return fRowX[iRow]; }
  
  Double_t &Alpha(){ return fAlpha;}
  Double_t &DAlpha(){ return fDAlpha;}
  Double_t &CosAlpha(){ return fCosAlpha;}
  Double_t &SinAlpha(){ return fSinAlpha;}
  Double_t &AngleMin(){ return fAngleMin;}
  Double_t &AngleMax(){ return fAngleMax;}
  Double_t &RMin(){ return fRMin;}
  Double_t &RMax(){ return fRMax;}
  Double_t &ZMin(){ return fZMin;}
  Double_t &ZMax(){ return fZMax;}
  Double_t &ErrZ(){ return fErrZ;}
  Double_t &ErrX(){ return fErrX;}
  Double_t &ErrY(){ return fErrY;}
  Double_t &Bz(){ return fBz;}

  Double_t &TrackConnectionFactor(){ return fTrackConnectionFactor; }
  Double_t &TrackChiCut() { return fTrackChiCut; }
  Double_t &TrackChi2Cut(){ return fTrackChi2Cut; }
  Int_t    &MaxTrackMatchDRow(){ return fMaxTrackMatchDRow; }
  Double_t &YErrorCorrection(){ return fYErrorCorrection; }
  Double_t &ZErrorCorrection(){ return fZErrorCorrection; }
  Double_t &CellConnectionAngleXY(){ return fCellConnectionAngleXY; }
  Double_t &CellConnectionAngleXZ(){ return fCellConnectionAngleXZ; }

 protected:

  Int_t fISlice; // slice number
  Int_t fNRows; // number of rows

  Double_t fAlpha, fDAlpha; // slice angle and angular size
  Double_t fCosAlpha, fSinAlpha;// sign and cosine of the slice angle
  Double_t fAngleMin, fAngleMax; // minimal and maximal angle
  Double_t fRMin, fRMax;// slice R range
  Double_t fZMin, fZMax;// slice Z range
  Double_t fErrX, fErrY, fErrZ;// default cluster errors
  Double_t fPadPitch; // pad pitch 
  Double_t fBz;       // magnetic field value (only constant field can be used)

  Double_t fYErrorCorrection;// correction factor for Y error of input clusters
  Double_t fZErrorCorrection;// correction factor for Z error of input clusters

  Double_t fCellConnectionAngleXY; // max phi angle between connected cells
  Double_t fCellConnectionAngleXZ; // max psi angle between connected cells
  Int_t    fMaxTrackMatchDRow;// maximal jump in TPC row for connecting track segments
  Double_t fTrackConnectionFactor; // allowed distance in Chi^2/3.5 for neighbouring tracks
  Double_t fTrackChiCut; // cut for track Sqrt(Chi2/NDF);
  Double_t fTrackChi2Cut;// cut for track Chi^2/NDF

  Double_t fRowX[200];// X-coordinate of rows

  ClassDef(AliHLTTPCCAParam,1);
};


#endif
