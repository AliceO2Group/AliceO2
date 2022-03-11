// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackend.cxx
/// \author David Rohr

#include "GPUDisplayBackend.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#include "GPUDisplayBackendOpenGL.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#include "GPUDisplayBackendVulkan.h"
#endif
#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayBackend* GPUDisplayBackend::getBackend(const char* type)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY
  if (strcmp(type, "opengl") == 0) {
    return new GPUDisplayBackendOpenGL;
  }
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  if (strcmp(type, "vulkan") == 0) {
    return new GPUDisplayBackendVulkan;
  }
#endif
  return nullptr;
}

void GPUDisplayBackend::fillIndirectCmdBuffer()
{
  mCmdBuffer.clear();
  mIndirectSliceOffset.resize(GPUCA_NSLICES);
  // TODO: Check if this can be parallelized
  for (int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
    mIndirectSliceOffset[iSlice] = mCmdBuffer.size();
    for (unsigned int k = 0; k < mDisplay->vertexBufferStart()[iSlice].size(); k++) {
      mCmdBuffer.emplace_back(mDisplay->vertexBufferCount()[iSlice][k], 1, mDisplay->vertexBufferStart()[iSlice][k], 0);
    }
  }
}
