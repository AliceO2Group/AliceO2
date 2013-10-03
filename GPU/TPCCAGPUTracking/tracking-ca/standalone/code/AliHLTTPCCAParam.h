//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCAParam.h 49685 2011-05-04 09:08:19Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAPARAM_H
#define ALIHLTTPCCAPARAM_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCATrackParam.h"

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
#include <iostream>
#endif

/**
 * @class ALIHLTTPCCAParam
 * parameters of the AliHLTTPCCATracker, including geometry information
 * and some reconstructon constants.
 *
 * The class is under construction.
 *
 */
MEM_CLASS_PRE class AliHLTTPCCAParam
{
  public:
	AliHLTTPCCAParam();
    ~AliHLTTPCCAParam() {}

#if !defined(HLTCA_GPUCODE)
    void Initialize( int iSlice, int nRows, float rowX[],
                            float alpha, float dAlpha,
                            float rMin, float rMax, float zMin, float zMax,
                            float padPitch, float zSigma, float bz );
    void Update();
#endif //!HLTCA_GPUCODE

	GPUd() void Slice2Global( float x, float y,  float z,
                              float *X, float *Y,  float *Z ) const;

    GPUd() void Global2Slice( float x, float y,  float z,
                              float *X, float *Y,  float *Z ) const;


    GPUhd() int ISlice() const { return fISlice;}
    GPUhd() int NRows() const { return fNRows;}

    GPUhd() float RowX( int iRow ) const { return fRowX[iRow]; }

    GPUd() float Alpha() const { return fAlpha;}
    GPUd() float Alpha( int iSlice ) const { return 0.174533 + DAlpha()*iSlice;}
    GPUd() float DAlpha() const { return fDAlpha;}
    GPUd() float CosAlpha() const { return fCosAlpha;}
    GPUd() float SinAlpha() const { return fSinAlpha;}
    GPUd() float AngleMin() const { return fAngleMin;}
    GPUd() float AngleMax() const { return fAngleMax;}
    GPUd() float RMin() const { return fRMin;}
    GPUd() float RMax() const { return fRMax;}
    GPUd() float ZMin() const { return fZMin;}
    GPUd() float ZMax() const { return fZMax;}
    GPUd() float ErrZ() const { return fErrZ;}
    GPUd() float ErrX() const { return fErrX;}
    GPUd() float ErrY() const { return fErrY;}
    GPUd() float BzkG() const { return fBzkG;}
    GPUd() float ConstBz() const { return fConstBz;}

    GPUd() float NeighboursSearchArea() const { return fNeighboursSearchArea; }
    GPUd() float TrackConnectionFactor() const { return fTrackConnectionFactor; }
    GPUd() float TrackChiCut()  const { return fTrackChiCut; }
    GPUd() float TrackChi2Cut() const { return fTrackChi2Cut; }
    GPUd() int   MaxTrackMatchDRow() const { return fMaxTrackMatchDRow; }
    GPUd() float HitPickUpFactor() const { return fHitPickUpFactor; }
  GPUd() float ClusterError2CorrectionY() const { return fClusterError2CorrectionY; }
  GPUd() float ClusterError2CorrectionZ() const { return fClusterError2CorrectionZ; }
  GPUd() int MinNTrackClusters() const { return fMinNTrackClusters; }
  GPUd() float MaxTrackQPt() const { return fMaxTrackQPt; }



    GPUhd() void SetISlice( int v ) {  fISlice = v;}
    GPUhd() void SetNRows( int v ) {  fNRows = v;}
    GPUhd() void SetRowX( int iRow, float v ) {  fRowX[iRow] = v; }
    GPUd() void SetAlpha( float v ) {  fAlpha = v;}
    GPUd() void SetDAlpha( float v ) {  fDAlpha = v;}
    GPUd() void SetCosAlpha( float v ) {  fCosAlpha = v;}
    GPUd() void SetSinAlpha( float v ) {  fSinAlpha = v;}
    GPUd() void SetAngleMin( float v ) {  fAngleMin = v;}
    GPUd() void SetAngleMax( float v ) {  fAngleMax = v;}
    GPUd() void SetRMin( float v ) {  fRMin = v;}
    GPUd() void SetRMax( float v ) {  fRMax = v;}
    GPUd() void SetZMin( float v ) {  fZMin = v;}
    GPUd() void SetZMax( float v ) {  fZMax = v;}
    GPUd() void SetErrZ( float v ) {  fErrZ = v;}
    GPUd() void SetErrX( float v ) {  fErrX = v;}
    GPUd() void SetErrY( float v ) {  fErrY = v;}
    GPUd() void SetBzkG( float v ) {  fBzkG = v;}

  GPUd() void SetNeighboursSearchArea( float v ) { fNeighboursSearchArea = v;}
    GPUd() void SetTrackConnectionFactor( float v ) { fTrackConnectionFactor = v;}
    GPUd() void SetTrackChiCut( float v ) {  fTrackChiCut = v; }
  GPUd() void SetTrackChi2Cut( float v ) {  fTrackChi2Cut = v; }
    GPUd() void SetMaxTrackMatchDRow( int v ) {  fMaxTrackMatchDRow = v; }
    GPUd() void SetHitPickUpFactor( float v ) {  fHitPickUpFactor = v; }
    GPUd() void SetClusterError2CorrectionY( float v ) { fClusterError2CorrectionY = v; }
    GPUd() void SetClusterError2CorrectionZ( float v ) { fClusterError2CorrectionZ = v; }

  GPUd() void SetMinNTrackClusters( int v ){ fMinNTrackClusters = v; }
  GPUd() void SetMinTrackPt( float v ){ fMaxTrackQPt = CAMath::Abs(v)>0.02 ?1./CAMath::Abs(v) :1./0.02; }

    GPUd() float GetClusterError2( int yz, int type, float z, float angle ) const;
    GPUd() void GetClusterErrors2( int row, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;
    GPUd() void GetClusterErrors2v1( int rowType, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
    void WriteSettings( std::ostream &out ) const;
    void ReadSettings( std::istream &in );
#endif

    GPUd() void SetParamS0Par( int i, int j, int k, float val ) {
      fParamS0Par[i][j][k] = val;
    }
  
    GPUd() const MakeType(float*) GetParamS0Par(int i, int j) const { return fParamS0Par[i][j]; }
 
    GPUd() float GetBzkG() const { return fBzkG;}
    GPUd() float GetConstBz() const { return fConstBz;}
    GPUd() float GetBz( float x, float y, float z ) const;
	MEM_CLASS_PRE2 GPUd() float GetBz( const AliHLTTPCCATrackParam MEM_LG2 &t ) const {return GetBz( t.X(), t.Y(), t.Z() );}

  protected:
    int fISlice; // slice number
    int fNRows; // number of rows

    float fAlpha, fDAlpha; // slice angle and angular size
    float fCosAlpha, fSinAlpha;// sign and cosine of the slice angle
    float fAngleMin, fAngleMax; // minimal and maximal angle
    float fRMin, fRMax;// slice R range
    float fZMin, fZMax;// slice Z range
    float fErrX, fErrY, fErrZ;// default cluster errors
    float fPadPitch; // pad pitch
    float fBzkG;       // constant magnetic field value in kG
    float fConstBz;       // constant magnetic field value in kG*clight

    float fHitPickUpFactor;// multiplier for the chi2 window for hit pick up procedure

    int   fMaxTrackMatchDRow;// maximal jump in TPC row for connecting track segments

  float fNeighboursSearchArea; // area in cm for the search of neighbours

    float fTrackConnectionFactor; // allowed distance in Chi^2/3.5 for neighbouring tracks
    float fTrackChiCut; // cut for track Sqrt(Chi2/NDF);
    float fTrackChi2Cut;// cut for track Chi^2/NDF
  float fClusterError2CorrectionY; // correction for the squared cluster error during tracking
  float fClusterError2CorrectionZ; // correction for the squared cluster error during tracking
    int fMinNTrackClusters; //* required min number of clusters on the track
    float fMaxTrackQPt;    //* required max Q/Pt (==min Pt) of tracks

    float fRowX[200];// X-coordinate of rows
    float fParamS0Par[2][3][7];    // cluster error parameterization coeficients
    float fPolinomialFieldBz[6];   // field coefficients

};



MEM_CLASS_PRE GPUd() inline float AliHLTTPCCAParam MEM_LG::GetBz( float x, float y, float z ) const
{
  float r2 = x * x + y * y;
  float r  = CAMath::Sqrt( r2 );
  const float *c = fPolinomialFieldBz;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

#endif //ALIHLTTPCCAPARAM_H
