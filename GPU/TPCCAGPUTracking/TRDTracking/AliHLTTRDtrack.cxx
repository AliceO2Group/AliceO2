#include "AliHLTTRDtrack.h"
#include "AliESDtrack.h"
#include "AliHLTGlobalBarrelTrack.h"

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
