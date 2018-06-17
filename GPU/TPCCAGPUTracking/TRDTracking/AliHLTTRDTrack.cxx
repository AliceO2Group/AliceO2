#include "AliHLTTRDTrack.h"
#include "AliESDtrack.h"
#include "AliHLTTRDTrackData.h"
#include "AliHLTTRDInterfaces.h"


template <typename T>
AliHLTTRDTrack<T>::AliHLTTRDTrack() :
  fChi2(0),
  fMass(0),
  fLabel(-1),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fLabelOffline(0),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  //Default constructor
  //------------------------------------------------------------------
  for (int i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}


template <typename T>
AliHLTTRDTrack<T>::AliHLTTRDTrack(const AliHLTTRDTrack<T>& t) :
  T(t),
  fChi2( t.fChi2 ),
  fMass( t.fMass ),
  fLabel( t.fLabel ),
  fTPCtrackId( t.fTPCtrackId ),
  fNtracklets( t.fNtracklets ),
  fNmissingConsecLayers( t.fNmissingConsecLayers ),
  fNtrackletsOffline( t.fNtrackletsOffline ),
  fLabelOffline( t.fLabelOffline ),
  fIsStopped( t.fIsStopped )
{
  //------------------------------------------------------------------
  //Copy constructor
  //------------------------------------------------------------------
  for (int i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
}


template <typename T>
AliHLTTRDTrack<T> &AliHLTTRDTrack<T>::operator=(const AliHLTTRDTrack<T>& t)
{
  //------------------------------------------------------------------
  //Assignment operator
  //------------------------------------------------------------------
  if( &t==this ) return *this;
  *(T*)this = t;
  fChi2 = t.fChi2;
  fMass = t.fMass;
  fLabel = t.fLabel;
  fTPCtrackId = t.fTPCtrackId;
  fNtracklets = t.fNtracklets;
  fNmissingConsecLayers = t.fNmissingConsecLayers;
  fNtrackletsOffline = t.fNtrackletsOffline;
  fLabelOffline = t.fLabelOffline;
  fIsStopped = t.fIsStopped;
  for (int i=0; i<=5; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
  return *this;
}


template <typename T>
AliHLTTRDTrack<T>::AliHLTTRDTrack(AliESDtrack& t,bool c) throw (const char *) :
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fLabelOffline(-1),
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
  //Set(par->GetX(),par->GetAlpha(),par->GetParameter(),par->GetCovariance());
  for (int i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}

template <typename T>
AliHLTTRDTrack<T>::AliHLTTRDTrack(const typename T::baseClass &t ) throw (const char *) :
  T(t),
  fChi2(0),
  fMass(0),
  fLabel(-1),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fNtrackletsOffline(0),
  fLabelOffline(-1),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // Conversion template track -> TRD track.
  //------------------------------------------------------------------
  for (int i=0; i<=5; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
}

template <typename T>
int AliHLTTRDTrack<T>::GetNlayers() const
{
  int res = 0;
  for (int iLy=0; iLy<6; iLy++) {
    if (fIsFindable[iLy]) {
      ++res;
    }
  }
  return res;
}


template <typename T>
int AliHLTTRDTrack<T>::GetTracklet(int iLayer) const
{
  if (iLayer < 0 || iLayer > 5) {
    //Error("GetTracklet", "illegal layer number %i", iLayer);
    return -1;
  }
  return fAttachedTracklets[iLayer];
}


template <typename T>
int AliHLTTRDTrack<T>::GetNmissingConsecLayers(int iLayer) const
{
  int res = 0;
  while (!fIsFindable[iLayer]) {
    ++res;
    --iLayer;
  }
  return res;
}


template <typename T>
void AliHLTTRDTrack<T>::ConvertTo( AliHLTTRDTrackDataRecord &t ) const
{
  // convert to HLT structure

  t.fAlpha = T::getAlpha();
  t.fX = T::getX();
  t.fY = T::getY();
  t.fZ = T::getZ();
  t.fq1Pt = T::getQ2Pt();
  t.fSinPhi = T::getSnp();
  t.fTgl = T::GetTgl();
  for( int i=0; i<15; i++ ) t.fC[i] = T::getCov()[i];
  t.fTPCTrackID = GetTPCtrackId();
  for ( int i = 0; i <6; i++ ){
    t.fAttachedTracklets[ i ] = GetTracklet( i );
  }
}

template <typename T>
void AliHLTTRDTrack<T>::ConvertFrom( const AliHLTTRDTrackDataRecord &t )
{
  // convert from HLT structure

  T::set(t.fX, t.fAlpha, &(t.fY), t.fC);
  SetTPCtrackId( t.fTPCTrackID );
  fChi2 = 0;
  fMass = 0.13957;
  fLabel = -1;
  fNtracklets = 0;
  fNmissingConsecLayers = 0;
  fLabelOffline = -1;
  fNtrackletsOffline = 0;
  fIsStopped = false;
  for ( int iLayer=0; iLayer <6; iLayer++ ){
    fAttachedTracklets[iLayer] = t.fAttachedTracklets[ iLayer ];
    fIsFindable[iLayer] = 0;
    if( fAttachedTracklets[iLayer]>=0 ) fNtracklets++;
  }
}

template class AliHLTTRDTrack<trackInterface<AliExternalTrackParam>>;
//template class AliHLTTRDTrack<trackInterface<AliHLTTPCGMTrackParam>>;
