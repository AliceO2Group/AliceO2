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

#ifndef GPUDISPLAYBACKENDOPENGL_H
#define GPUDISPLAYBACKENDOPENGL_H

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
  void ActivateColor(std::array<float, 4>& color) override;
  void setQuality() override;
  void setDepthBuffer() override;
  void setFrameBuffer(int updateCurrent, unsigned int newID) override;
  int InitBackendA() override;
  void ExitBackendA() override;
  void clearScreen(bool colorOnly = false) override;
  void updateSettings() override;
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw() override;
  void finishDraw() override;
  void finishFrame() override;
  void prepareText() override;
  void finishText() override;
  void setMatrices(const hmm_mat4& proj, const hmm_mat4& view) override;
  void mixImages(GLfb& mixBuffer, float mixSlaveImage) override;
  void renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer) override;
  void readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height) override;
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;
  backendTypes backendType() const override { return TYPE_OPENGL; }
  size_t needMultiVBO() override { return 0x100000000ll; }

  void addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) override;
  void initializeTextDrawing() override;
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override;

  struct FontSymbolOpenGL : public FontSymbol {
    unsigned int texId;
  };

  unsigned int mVertexShader;
  unsigned int mFragmentShader;
  unsigned int mVertexShaderText;
  unsigned int mFragmentShaderText;
  unsigned int mShaderProgram;
  unsigned int mShaderProgramText;
  unsigned int mVertexArray;

  unsigned int mIndirectId;
  std::vector<unsigned int> mVBOId;
  std::vector<FontSymbolOpenGL> mFontSymbols;
  int mModelViewProjId;
  int mColorId;
  int mModelViewProjIdText;
  int mColorIdText;
  unsigned int mSPIRVModelViewBuffer;
  unsigned int mSPIRVColorBuffer;

  unsigned int VAO_text, VBO_text;
  bool mSPIRVShaders = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
