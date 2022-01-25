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

/// \file GPUDisplayBackendOpenGL.h
/// \author David Rohr

#ifndef GPUDISPLAYEXT_H
#define GPUDISPLAYEXT_H
#ifdef GPUCA_BUILD_EVENT_DISPLAY

#include "GPUDisplayBackend.h"

#include <vector>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackendOpenGL : public GPUDisplayBackend
{
  int ExtInit() override;
  bool CoreProfile() override;
  unsigned int DepthBits() override;

 protected:
  void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa) override;
  void deleteFB(GLfb& fb) override;

  unsigned int drawVertices(const vboList& v, const drawType t) override;
  void ActivateColor(std::array<float, 3>& color) override;
  void setQuality() override;
  void setDepthBuffer() override;
  void setFrameBuffer(int updateCurrent, unsigned int newID) override;
  int InitBackend() override;
  void ExitBackend() override;
  void clearScreen(bool colorOnly = false) override;
  void updateSettings() override;
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw() override;
  void finishDraw() override;
  void prepareText() override;
  void setMatrices(const hmm_mat4& proj, const hmm_mat4& view) override;
  void mixImages(GLfb& mixBuffer, float mixSlaveImage) override;
  void renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer) override;
  void readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height) override;
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;

  struct DrawArraysIndirectCommand {
    DrawArraysIndirectCommand(unsigned int a = 0, unsigned int b = 0, unsigned int c = 0, unsigned int d = 0) : count(a), instanceCount(b), first(c), baseInstance(d) {}
    unsigned int count;
    unsigned int instanceCount;

    unsigned int first;
    unsigned int baseInstance;
  };
  vecpod<DrawArraysIndirectCommand> mCmdBuffer;

  unsigned int mIndirectId;
  std::vector<unsigned int> mVBOId;
  std::vector<int> mIndirectSliceOffset;
  int mModelViewProjId;
  int mColorId;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
