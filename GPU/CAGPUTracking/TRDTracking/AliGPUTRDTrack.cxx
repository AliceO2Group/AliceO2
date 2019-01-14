#include "AliGPUTRDTrack.h"
#include "AliGPUTRDTrackData.h"

#ifdef GPUCA_ALIROOT_LIB
#include "AliHLTExternalTrackParam.h"

template <typename T>
GPUd() AliGPUTRDTrack<T>::AliGPUTRDTrack(const AliHLTExternalTrackParam &t) :
  T(t),
  fChi2(0),
  fMass(0),
  fLabel(-1),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fLabelOffline(-1),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // copy constructor from AliHLTExternalTrackParam struct
  //------------------------------------------------------------------
  for (int i=0; i<kNLayers; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = 0;
  }
}
#endif

template <typename T>
GPUd() AliGPUTRDTrack<T>::AliGPUTRDTrack() :
  fChi2(0),
  fMass(0),
  fLabel(-1),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fLabelOffline(0),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // default constructor
  //------------------------------------------------------------------
  for (int i=0; i<kNLayers; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = 0;
  }
}


template <typename T>
GPUd() AliGPUTRDTrack<T>::AliGPUTRDTrack(const AliGPUTRDTrack<T>& t) :
  T(t),
  fChi2( t.fChi2 ),
  fMass( t.fMass ),
  fLabel( t.fLabel ),
  fTPCtrackId( t.fTPCtrackId ),
  fNtracklets( t.fNtracklets ),
  fNmissingConsecLayers( t.fNmissingConsecLayers ),
  fLabelOffline( t.fLabelOffline ),
  fIsStopped( t.fIsStopped )
{
  //------------------------------------------------------------------
  // copy constructor
  //------------------------------------------------------------------
  for (int i=0; i<kNLayers; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = t.fNtrackletsOffline[j];
  }
}

template <typename T>
GPUd() AliGPUTRDTrack<T>::AliGPUTRDTrack(const T& t) :
  T(t),
  fChi2(0),
  fMass(0),
  fLabel(-1),
  fTPCtrackId(0),
  fNtracklets(0),
  fNmissingConsecLayers(0),
  fLabelOffline(-1),
  fIsStopped(false)
{
  //------------------------------------------------------------------
  // copy constructor from anything
  //------------------------------------------------------------------
  for (int i=0; i<kNLayers; ++i) {
    fAttachedTracklets[i] = -1;
    fIsFindable[i] = 0;
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = 0;
  }
}

template <typename T>
GPUd() AliGPUTRDTrack<T> &AliGPUTRDTrack<T>::operator=(const AliGPUTRDTrack<T>& t)
{
  //------------------------------------------------------------------
  // assignment operator
  //------------------------------------------------------------------
  if( &t==this ) return *this;
  *(T*)this = t;
  fChi2 = t.fChi2;
  fMass = t.fMass;
  fLabel = t.fLabel;
  fTPCtrackId = t.fTPCtrackId;
  fNtracklets = t.fNtracklets;
  fNmissingConsecLayers = t.fNmissingConsecLayers;
  fLabelOffline = t.fLabelOffline;
  fIsStopped = t.fIsStopped;
  for (int i=0; i<kNLayers; ++i) {
    fAttachedTracklets[i] = t.fAttachedTracklets[i];
    fIsFindable[i] = t.fIsFindable[i];
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = t.fNtrackletsOffline[j];
  }
  return *this;
}


template <typename T>
GPUd() int AliGPUTRDTrack<T>::GetNlayers() const
{
  //------------------------------------------------------------------
  // returns number of layers in which the track is in active area of TRD
  //------------------------------------------------------------------
  int res = 0;
  for (int iLy=0; iLy<kNLayers; iLy++) {
    if (fIsFindable[iLy]) {
      ++res;
    }
  }
  return res;
}


template <typename T>
GPUd() int AliGPUTRDTrack<T>::GetTracklet(int iLayer) const
{
  //------------------------------------------------------------------
  // returns index of attached tracklet in given layer
  //------------------------------------------------------------------
  if (iLayer < 0 || iLayer >= kNLayers) {
    return -1;
  }
  return fAttachedTracklets[iLayer];
}


template <typename T>
GPUd() int AliGPUTRDTrack<T>::GetNmissingConsecLayers(int iLayer) const
{
  //------------------------------------------------------------------
  // returns number of consecutive layers in which the track was
  // inside the deadzone up to (and including) the given layer
  //------------------------------------------------------------------
  int res = 0;
  while (!fIsFindable[iLayer]) {
    ++res;
    --iLayer;
    if (iLayer < 0) {
      break;
    }
  }
  return res;
}


template <typename T>
GPUd() void AliGPUTRDTrack<T>::ConvertTo( AliGPUTRDTrackDataRecord &t ) const
{
  //------------------------------------------------------------------
  // convert to GPU structure
  //------------------------------------------------------------------
  t.fAlpha = T::getAlpha();
  t.fX = T::getX();
  t.fY = T::getY();
  t.fZ = T::getZ();
  t.fq1Pt = T::getQ2Pt();
  t.fSinPhi = T::getSnp();
  t.fTgl = T::getTgl();
  for( int i=0; i<15; i++ ) {
    t.fC[i] = T::getCov()[i];
  }
  t.fTPCTrackID = GetTPCtrackId();
  for ( int i = 0; i < kNLayers; i++ ) {
    t.fAttachedTracklets[ i ] = GetTracklet( i );
  }
}

template <typename T>
GPUd() void AliGPUTRDTrack<T>::ConvertFrom( const AliGPUTRDTrackDataRecord &t )
{
  //------------------------------------------------------------------
  // convert from GPU structure
  //------------------------------------------------------------------
  T::set(t.fX, t.fAlpha, &(t.fY), t.fC);
  SetTPCtrackId( t.fTPCTrackID );
  fChi2 = 0;
  fMass = 0.13957;
  fLabel = -1;
  fNtracklets = 0;
  fNmissingConsecLayers = 0;
  fLabelOffline = -1;
  fIsStopped = false;
  for ( int iLayer=0; iLayer < kNLayers; iLayer++ ){
    fAttachedTracklets[iLayer] = t.fAttachedTracklets[ iLayer ];
    fIsFindable[iLayer] = 0;
    if( fAttachedTracklets[iLayer]>=0 ) fNtracklets++;
  }
  for (int j=0; j<4; ++j) {
    fNtrackletsOffline[j] = 0;
  }
}

#ifdef GPUCA_ALIROOT_LIB //Instantiate AliRoot track version
template class AliGPUTRDTrack<trackInterface<AliExternalTrackParam>>;
#endif
#ifdef GPUCA_O2_LIB //Instantiate O2 track version
//Not yet existing
#endif
template class AliGPUTRDTrack<trackInterface<AliGPUTPCGMTrackParam>>; //Always instatiate GPU track version
