#include "AliHLTTRDtrack.h"
#include "AliESDtrack.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "AliExternalTrackParam.h"

ClassImp(AliHLTTRDtrack);

AliHLTTRDtrack::AliHLTTRDtrack() :
  fTPCtrackId(0),
  fNtracklets(0)
{
  //------------------------------------------------------------------
  //Default constructor
  //------------------------------------------------------------------
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
  }
}


AliHLTTRDtrack::AliHLTTRDtrack(const AliHLTTRDtrack& t) :
  AliKalmanTrack(t),
  fTPCtrackId( t.fTPCtrackId),
  fNtracklets( t.fNtracklets)
{
  //------------------------------------------------------------------
  //Copy constructor
  //------------------------------------------------------------------
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
  }
}


AliHLTTRDtrack &AliHLTTRDtrack::operator=(const AliHLTTRDtrack& t)
{
  //------------------------------------------------------------------
  //Assignment operator
  //------------------------------------------------------------------
  if( &t==this ) return *this;
  *(AliKalmanTrack*)this = t;
  fTPCtrackId = t.fTPCtrackId;
  fNtracklets = t.fNtracklets;
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
  }
  return *this;
}


AliHLTTRDtrack::AliHLTTRDtrack(AliESDtrack& t,Bool_t c) throw (const Char_t *) :
  AliKalmanTrack(),
  fTPCtrackId(0),
  fNtracklets(0)
{
  //------------------------------------------------------------------
  // Conversion ESD track -> TRD HLT track.
  // If c==kTRUE, create the TRD track out of the constrained params.
  //------------------------------------------------------------------
  const AliExternalTrackParam *par=&t;
  if (c) {
    par=t.GetConstrainedParam();
    if (!par) throw "AliHLTTRDtrack: conversion failed !\n";
  }
  Set(par->GetX(),par->GetAlpha(),par->GetParameter(),par->GetCovariance());
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
  }
}

AliHLTTRDtrack::AliHLTTRDtrack(AliExternalTrackParam& t ) throw (const Char_t *) :
  AliKalmanTrack(),
  fTPCtrackId(0),
  fNtracklets(0)
{
  //------------------------------------------------------------------
  // Conversion ESD track -> TRD track.
  // If c==kTRUE, create the TRD track out of the constrained params.
  //------------------------------------------------------------------
  const AliExternalTrackParam *par=&t;
  Set(par->GetX(),par->GetAlpha(),par->GetParameter(),par->GetCovariance());
  for (Int_t i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
  }
}

AliHLTTRDtrack::AliHLTTRDtrack(const AliHLTGlobalBarrelTrack& t) :
  AliKalmanTrack(t),
  fTPCtrackId(0),
  fNtracklets(0)
{
  fTPCtrackId = 0;
  fNtracklets = 0;
  for (Int_t i=0; i<6; ++i) {
    fAttachedTracklets[i] = -2;
  }
  for( Int_t i=0; i<t.GetNumberOfPoints() && i<6; ++i) {
    fAttachedTracklets[i] = t.GetClusterIndex(i);
    if( fAttachedTracklets[i]>=0 ) fNtracklets++;
  }
}

Int_t AliHLTTRDtrack::GetTracklet(Int_t iLayer) const
{
  if (fAttachedTracklets[iLayer] == -1) {
    return -1;
  }
  else {
    return fAttachedTracklets[iLayer];
  }
}


size_t AliHLTTRDtrack::ConvertTo( AliHLTExternalTrackParam* t ) const
{
  // convert to HLT structure
  
  t->fAlpha = GetAlpha();
  t->fX = GetX();
  t->fY = GetY();
  t->fZ = GetZ();
  t->fLastX = 0;
  t->fLastY = 0;
  t->fLastZ = 0;
  t->fq1Pt = GetSigned1Pt();
  t->fSinPsi = GetSnp();
  t->fTgl = GetTgl();
  for( int i=0; i<15; i++ ) t->fC[i] = GetCovariance()[i];
  t->fTrackID = GetTPCtrackId();
  t->fFlags = 0;
  t->fNPoints = 6;
  for ( int i = 0; i <6; i++ ){
    t->fPointIDs[ i ] = GetTracklet( i );
  }  
  return sizeof(AliHLTExternalTrackParam) + 6 * sizeof( unsigned int );
}

void AliHLTTRDtrack::ConvertFrom( const AliHLTExternalTrackParam* t )
{
  // convert from HLT structure
  Set(t->fX, t->fAlpha, &(t->fY), t->fC);
  SetTPCtrackId( t->fTrackID );
  fNtracklets = 0;
  int iLayer=0;
  for ( ; iLayer <6 && iLayer<t->fNPoints; iLayer++ ){
    fAttachedTracklets[iLayer] = t->fPointIDs[ iLayer ];
    if( fAttachedTracklets[iLayer]>=0 ) fNtracklets++;
  }
  for ( ; iLayer <6; iLayer++ ){
    fAttachedTracklets[iLayer] = -1;
  }
}
