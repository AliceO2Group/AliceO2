// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackend.cxx
/// \author David Rohr

#include "GPUDisplayBackend.h"
#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

void* GPUDisplayBackend::OpenGLWrapper(void* ptr)
{
  GPUDisplayBackend* me = reinterpret_cast<GPUDisplayBackend*>(ptr);
  int retVal = me->OpenGLMain();
  if (retVal == -1) {
    me->InitGL(true);
  }
  return ((void*)(size_t)retVal);
}

void GPUDisplayBackend::HandleSendKey()
{
  if (mSendKey) {
    mDisplay->HandleSendKey(mSendKey);
    mSendKey = 0;
  }
}

void GPUDisplayBackend::HandleKeyRelease(unsigned char key) { mDisplay->HandleKeyRelease(key); }
int GPUDisplayBackend::DrawGLScene(bool mixAnimation, float animateTime) { return mDisplay->DrawGLScene(mixAnimation, animateTime); }
void GPUDisplayBackend::ReSizeGLScene(int width, int height)
{
  mDisplayHeight = height;
  mDisplayWidth = width;
  mDisplay->ReSizeGLScene(width, height);
}
int GPUDisplayBackend::InitGL(bool initFailure) { return mDisplay->InitGL(initFailure); }
void GPUDisplayBackend::ExitGL() { return mDisplay->ExitGL(); }
bool GPUDisplayBackend::EnableSendKey() { return true; }
