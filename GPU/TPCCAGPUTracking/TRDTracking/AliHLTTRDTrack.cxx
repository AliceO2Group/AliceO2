#include "AliHLTTRDTrack.h"
#include "AliESDtrack.h"
#include "AliHLTTRDTrackData.h"

ClassImp(AliHLTTRDTrack);

AliHLTTRDTrack::AliHLTTRDTrack() :
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  //Default constructor
  //------------------------------------------------------------------
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}


AliHLTTRDTrack::AliHLTTRDTrack(const AliHLTTRDTrack& t) :
  AliKalmanTrack(t),
  fTPCtrackId( t.fTPCtrackId),
  fNtracklets( t.fNtracklets),
  fNmissingConsecLayers( t.fNmissingConsecLayers),
  fNtrackletsOffline( t.fNtrackletsOffline),
  fIsStopped( t.fIsStopped)
{
  //------------------------------------------------------------------
  //Copy constructor
  //------------------------------------------------------------------
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
}


AliHLTTRDTrack &AliHLTTRDTrack::operator=(const AliHLTTRDTrack& t)
{
  //------------------------------------------------------------------
  //Assignment operator
  //------------------------------------------------------------------
  if( &t==this ) return *this;
  *(AliKalmanTrack*)this = t;
  fTPCtrackId = t.fTPCtrackId;
  fNtracklets = t.fNtracklets;
  fNmissingConsecLayers = t.fNmissingConsecLayers;
  fNtrackletsOffline = t.fNtrackletsOffline;
  fIsStopped = t.fIsStopped;
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
  return *this;
}


AliHLTTRDTrack::AliHLTTRDTrack(AliESDtrack& t,Bool_t c) throw (const Char_t *) :
  AliKalmanTrack(),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // Conversion ESD track -> TRD HLT track.
  // If c==kTRUE, create the TRD track out of the constrained params.
  //------------------------------------------------------------------
  const AliExternalTrackParam *par=&t;
  if (c) {
    par=t.GetConstrainedParam();
    if (!par) throw "AliHLTTRDTrack: conversion failed !\n";
  }
  Set(par->GetX(),par->GetAlpha(),par->GetParameter(),par->GetCovariance());
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}

AliHLTTRDTrack::AliHLTTRDTrack(AliExternalTrackParam& t ) throw (const Char_t *) :
  AliKalmanTrack(),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // Conversion ESD track -> TRD track.
  // If c==kTRUE, create the TRD track out of the constrained params.
  //------------------------------------------------------------------
  const AliExternalTrackParam *par=&t;
  Set(par->GetX(),par->GetAlpha(),par->GetParameter(),par->GetCovariance());
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}

Int_t AliHLTTRDTrack::GetNlayers() const
{
  Int_t res = 0;
  for (Int_t iLy=0; iLy<6; iLy++) {
    if (fIsFindable[iLy]) {
      ++res;
    }
  }
  return res;
}


Int_t AliHLTTRDTrack::GetTracklet(Int_t iLayer) const
{
  if (iLayer < 0 || iLayer > 5) {
    //Error("GetTracklet", "illegal layer number %i", iLayer);
    return -1;
  }
  return fAttachedTracklets[iLayer];
}


Int_t AliHLTTRDTrack::GetNmissingConsecLayers(Int_t iLayer) const
{
  Int_t res = 0;
  while (!fIsFindable[iLayer]) {
    ++res;
    --iLayer;
  }
  return res;
}


void AliHLTTRDTrack::ConvertTo( AliHLTTRDTrackDataRecord &t ) const
{
  // convert to HLT structure

  t.fAlpha = GetAlpha();
  t.fX = GetX();
  t.fY = GetY();
  t.fZ = GetZ();
  t.fq1Pt = GetSigned1Pt();
  t.fSinPhi = GetSnp();
  t.fTgl = GetTgl();
  for( int i=0; i<15; i++ ) t.fC[i] = GetCovariance()[i];
  t.fTPCTrackID = GetTPCtrackId();
  for ( int i = 0; i <6; i++ ){
    t.fAttachedTracklets[ i ] = GetTracklet( i );
  }
}

void AliHLTTRDTrack::ConvertFrom( const AliHLTTRDTrackDataRecord &t )
{
  // convert from HLT structure

  Set(t.fX, t.fAlpha, &(t.fY), t.fC);
  SetTPCtrackId( t.fTPCTrackID );
  fNtracklets = 0;
  fNmissingConsecLayers = 0;
  fNtrackletsOffline = 0;
  fIsStopped = false;
  for ( int iLayer=0; iLayer <6; iLayer++ ){
    fAttachedTracklets[iLayer] = t.fAttachedTracklets[ iLayer ];
    fIsFindable[iLayer] = 0;
    if( fAttachedTracklets[iLayer]>=0 ) fNtracklets++;
  }
}
