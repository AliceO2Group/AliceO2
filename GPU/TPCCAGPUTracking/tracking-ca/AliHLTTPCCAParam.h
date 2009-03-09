//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAPARAM_H
#define ALIHLTTPCCAPARAM_H

#include "AliHLTTPCCADef.h"

#include <iostream>


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

#if !defined(HLTCA_GPUCODE)  
  GPUd() AliHLTTPCCAParam();
#endif

  ~AliHLTTPCCAParam(){;}

  GPUd() void Initialize( Int_t iSlice, Int_t nRows, Float_t rowX[],
		   Float_t alpha, Float_t dAlpha,
		   Float_t rMin, Float_t rMax, Float_t zMin, Float_t zMax,
		   Float_t padPitch, Float_t zSigma, Float_t bz );
  GPUd() void Update();
  
  GPUd() void Slice2Global( Float_t x, Float_t y,  Float_t z, 
		     Float_t *X, Float_t *Y,  Float_t *Z ) const;
  GPUd() GPUd() void Global2Slice( Float_t x, Float_t y,  Float_t z, 
		     Float_t *X, Float_t *Y,  Float_t *Z ) const;


  GPUhd() Int_t ISlice() const { return fISlice;}
  GPUhd() Int_t NRows() const { return fNRows;}

  GPUhd() Float_t RowX( Int_t iRow ) const { return fRowX[iRow]; }  

  GPUd() Float_t Alpha() const { return fAlpha;}
  GPUd() Float_t DAlpha() const { return fDAlpha;}
  GPUd() Float_t CosAlpha() const { return fCosAlpha;}
  GPUd() Float_t SinAlpha() const { return fSinAlpha;}
  GPUd() Float_t AngleMin() const { return fAngleMin;}
  GPUd() Float_t AngleMax() const { return fAngleMax;}
  GPUd() Float_t RMin() const { return fRMin;}
  GPUd() Float_t RMax() const { return fRMax;}
  GPUd() Float_t ZMin() const { return fZMin;}
  GPUd() Float_t ZMax() const { return fZMax;}
  GPUd() Float_t ErrZ() const { return fErrZ;}
  GPUd() Float_t ErrX() const { return fErrX;}
  GPUd() Float_t ErrY() const { return fErrY;}
  GPUd() Float_t Bz() const { return fBz;}

  GPUd() Float_t TrackConnectionFactor() const { return fTrackConnectionFactor; }
  GPUd() Float_t TrackChiCut()  const { return fTrackChiCut; }
  GPUd() Float_t TrackChi2Cut() const { return fTrackChi2Cut; }
  GPUd() Int_t   MaxTrackMatchDRow() const { return fMaxTrackMatchDRow; }
  GPUd() Float_t HitPickUpFactor() const { return fHitPickUpFactor; }



  GPUhd() void SetISlice( Int_t v ){  fISlice = v;}
  GPUhd() void SetNRows( Int_t v ){  fNRows = v;}  
  GPUhd() void SetRowX( Int_t iRow, Float_t v ){  fRowX[iRow] = v; }
  GPUd() void SetAlpha( Float_t v ){  fAlpha = v;}
  GPUd() void SetDAlpha( Float_t v ){  fDAlpha = v;}
  GPUd() void SetCosAlpha( Float_t v ){  fCosAlpha = v;}
  GPUd() void SetSinAlpha( Float_t v ){  fSinAlpha = v;}
  GPUd() void SetAngleMin( Float_t v ){  fAngleMin = v;}
  GPUd() void SetAngleMax( Float_t v ){  fAngleMax = v;}
  GPUd() void SetRMin( Float_t v ){  fRMin = v;}
  GPUd() void SetRMax( Float_t v ){  fRMax = v;}
  GPUd() void SetZMin( Float_t v ){  fZMin = v;}
  GPUd() void SetZMax( Float_t v ){  fZMax = v;}
  GPUd() void SetErrZ( Float_t v ){  fErrZ = v;}
  GPUd() void SetErrX( Float_t v ){  fErrX = v;}
  GPUd() void SetErrY( Float_t v ){  fErrY = v;}
  GPUd() void SetBz( Float_t v ){  fBz = v;}
  GPUd() void SetTrackConnectionFactor( Float_t v ){ fTrackConnectionFactor = v;}
  GPUd() void SetTrackChiCut( Float_t v ) {  fTrackChiCut = v; }
  GPUd() void SetTrackChi2Cut( Float_t v ){  fTrackChi2Cut = v; }
  GPUd() void SetMaxTrackMatchDRow( Int_t v ){  fMaxTrackMatchDRow = v; }
  GPUd() void SetHitPickUpFactor( Float_t v ){  fHitPickUpFactor = v; }


  GPUd() Float_t GetClusterError2(Int_t yz, Int_t type, Float_t z, Float_t angle ) const;

  void WriteSettings( std::ostream &out ) const;
  void ReadSettings( std::istream &in );
  
  GPUd() void SetParamS0Par(Int_t i, Int_t j, Int_t k, Float_t val ){
    fParamS0Par[i][j][k] = val;
  }

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

  Float_t fHitPickUpFactor;// multiplier for the chi2 window for hit pick up procedure

  Int_t   fMaxTrackMatchDRow;// maximal jump in TPC row for connecting track segments
  Float_t fTrackConnectionFactor; // allowed distance in Chi^2/3.5 for neighbouring tracks
  Float_t fTrackChiCut; // cut for track Sqrt(Chi2/NDF);
  Float_t fTrackChi2Cut;// cut for track Chi^2/NDF

  Float_t fRowX[200];// X-coordinate of rows
  Float_t fParamS0Par[2][3][7];    // cluster error parameterization coeficients

};


#endif
