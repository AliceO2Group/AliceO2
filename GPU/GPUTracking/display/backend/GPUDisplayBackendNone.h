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

/// \file GPUDisplayBackendNone.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDNONE_H
#define GPUDISPLAYBACKENDNONE_H

#include "GPUDisplayBackend.h"

namespace o2::gpu
{
class GPUDisplayBackendNone : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendNone();
  ~GPUDisplayBackendNone() override = default;

 protected:
  uint32_t DepthBits() override { return 32; };
  uint32_t drawVertices(const vboList& v, const drawType t) override { return 0; }
  void ActivateColor(std::array<float, 4>& color) override {}
  void setDepthBuffer() override {}
  int32_t InitBackendA() override;
  void ExitBackendA() override {}
  void loadDataToGPU(size_t totalVertizes) override {}
  void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) override {}
  void finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage) override {}
  void finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) override {}
  void prepareText() override {}
  void finishText() override {}
  void pointSizeFactor(float factor) override {}
  void lineWidthFactor(float factor) override {}
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override {}
  void addFontSymbol(int32_t symbol, int32_t sizex, int32_t sizey, int32_t offsetx, int32_t offsety, int32_t advance, void* data) override {}
  void initializeTextDrawing() override {}
};
} // namespace o2::gpu

#endif
