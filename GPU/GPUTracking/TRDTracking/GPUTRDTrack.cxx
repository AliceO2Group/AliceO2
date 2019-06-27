// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrack.cxx
/// \author Ole Schmidt, Sergey Gorbunov

#include "GPUTRDTrack.h"
#include "GPUTRDTrackData.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_ALIROOT_LIB
#include "AliHLTExternalTrackParam.h"

template <typename T>
GPUd() GPUTRDTrack_t<T>::GPUTRDTrack_t(const AliHLTExternalTrackParam& t) : T(t), mChi2(0.f), mMass(0.f), mLabel(-1), mTPCTrackId(0), mNTracklets(0), mNMissingConsecLayers(0), mLabelOffline(-1), mIsStopped(false)
{
  //------------------------------------------------------------------
  // copy constructor from AliHLTExternalTrackParam struct
  //------------------------------------------------------------------
  for (int i = 0; i < kNLayers; ++i) {
    mAttachedTracklets[i] = -1;
    mIsFindable[i] = 0;
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = 0;
  }
}
#endif

template <typename T>
GPUd() GPUTRDTrack_t<T>::GPUTRDTrack_t() : mChi2(0.f), mMass(0.f), mLabel(-1), mTPCTrackId(0), mNTracklets(0), mNMissingConsecLayers(0), mLabelOffline(-1), mIsStopped(false)
{
  //------------------------------------------------------------------
  // default constructor
  //------------------------------------------------------------------
  for (int i = 0; i < kNLayers; ++i) {
    mAttachedTracklets[i] = -1;
    mIsFindable[i] = 0;
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = 0;
  }
}

template <typename T>
GPUd() GPUTRDTrack_t<T>::GPUTRDTrack_t(const GPUTRDTrack_t<T>& t)
  : T(t), mChi2(t.mChi2), mMass(t.mMass), mLabel(t.mLabel), mTPCTrackId(t.mTPCTrackId), mNTracklets(t.mNTracklets), mNMissingConsecLayers(t.mNMissingConsecLayers), mLabelOffline(t.mLabelOffline), mIsStopped(t.mIsStopped)
{
  //------------------------------------------------------------------
  // copy constructor
  //------------------------------------------------------------------
  for (int i = 0; i < kNLayers; ++i) {
    mAttachedTracklets[i] = t.mAttachedTracklets[i];
    mIsFindable[i] = t.mIsFindable[i];
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = t.mNTrackletsOffline[j];
  }
}

template <typename T>
GPUd() GPUTRDTrack_t<T>::GPUTRDTrack_t(const T& t) : T(t), mChi2(0.f), mMass(0.f), mLabel(-1), mTPCTrackId(0), mNTracklets(0), mNMissingConsecLayers(0), mLabelOffline(-1), mIsStopped(false)
{
  //------------------------------------------------------------------
  // copy constructor from anything
  //------------------------------------------------------------------
  for (int i = 0; i < kNLayers; ++i) {
    mAttachedTracklets[i] = -1;
    mIsFindable[i] = 0;
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = 0;
  }
}

template <typename T>
GPUd() GPUTRDTrack_t<T>& GPUTRDTrack_t<T>::operator=(const GPUTRDTrack_t<T>& t)
{
  //------------------------------------------------------------------
  // assignment operator
  //------------------------------------------------------------------
  if (&t == this) {
    return *this;
  }
  *(T*)this = t;
  mChi2 = t.mChi2;
  mMass = t.mMass;
  mLabel = t.mLabel;
  mTPCTrackId = t.mTPCTrackId;
  mNTracklets = t.mNTracklets;
  mNMissingConsecLayers = t.mNMissingConsecLayers;
  mLabelOffline = t.mLabelOffline;
  mIsStopped = t.mIsStopped;
  for (int i = 0; i < kNLayers; ++i) {
    mAttachedTracklets[i] = t.mAttachedTracklets[i];
    mIsFindable[i] = t.mIsFindable[i];
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = t.mNTrackletsOffline[j];
  }
  return *this;
}

template <typename T>
GPUd() int GPUTRDTrack_t<T>::GetNlayers() const
{
  //------------------------------------------------------------------
  // returns number of layers in which the track is in active area of TRD
  //------------------------------------------------------------------
  int res = 0;
  for (int iLy = 0; iLy < kNLayers; iLy++) {
    if (mIsFindable[iLy]) {
      ++res;
    }
  }
  return res;
}

template <typename T>
GPUd() int GPUTRDTrack_t<T>::GetTracklet(int iLayer) const
{
  //------------------------------------------------------------------
  // returns index of attached tracklet in given layer
  //------------------------------------------------------------------
  if (iLayer < 0 || iLayer >= kNLayers) {
    return -1;
  }
  return mAttachedTracklets[iLayer];
}

template <typename T>
GPUd() int GPUTRDTrack_t<T>::GetNmissingConsecLayers(int iLayer) const
{
  //------------------------------------------------------------------
  // returns number of consecutive layers in which the track was
  // inside the deadzone up to (and including) the given layer
  //------------------------------------------------------------------
  int res = 0;
  while (!mIsFindable[iLayer]) {
    ++res;
    --iLayer;
    if (iLayer < 0) {
      break;
    }
  }
  return res;
}

template <typename T>
GPUd() void GPUTRDTrack_t<T>::ConvertTo(GPUTRDTrackDataRecord& t) const
{
  //------------------------------------------------------------------
  // convert to GPU structure
  //------------------------------------------------------------------
  t.mAlpha = T::getAlpha();
  t.fX = T::getX();
  t.fY = T::getY();
  t.fZ = T::getZ();
  t.fq1Pt = T::getQ2Pt();
  t.mSinPhi = T::getSnp();
  t.fTgl = T::getTgl();
  for (int i = 0; i < 15; i++) {
    t.fC[i] = T::getCov()[i];
  }
  t.fTPCTrackID = GetTPCtrackId();
  for (int i = 0; i < kNLayers; i++) {
    t.fAttachedTracklets[i] = GetTracklet(i);
  }
}

template <typename T>
GPUd() void GPUTRDTrack_t<T>::ConvertFrom(const GPUTRDTrackDataRecord& t)
{
  //------------------------------------------------------------------
  // convert from GPU structure
  //------------------------------------------------------------------
  T::set(t.fX, t.mAlpha, &(t.fY), t.fC);
  SetTPCtrackId(t.fTPCTrackID);
  mChi2 = 0.f;
  mMass = 0.13957f;
  mLabel = -1;
  mNTracklets = 0;
  mNMissingConsecLayers = 0;
  mLabelOffline = -1;
  mIsStopped = false;
  for (int iLayer = 0; iLayer < kNLayers; iLayer++) {
    mAttachedTracklets[iLayer] = t.fAttachedTracklets[iLayer];
    mIsFindable[iLayer] = 0;
    if (mAttachedTracklets[iLayer] >= 0) {
      mNTracklets++;
    }
  }
  for (int j = 0; j < 4; ++j) {
    mNTrackletsOffline[j] = 0;
  }
}

#ifndef GPUCA_GPUCODE
namespace GPUCA_NAMESPACE
{
namespace gpu
{
#ifdef GPUCA_ALIROOT_LIB // Instantiate AliRoot track version
template class GPUTRDTrack_t<trackInterface<AliExternalTrackParam>>;
#endif
#ifdef GPUCA_O2_LIB // Instantiate O2 track version
// Not yet existing
#endif
template class GPUTRDTrack_t<trackInterface<GPUTPCGMTrackParam>>; // Always instatiate GM track version
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif
