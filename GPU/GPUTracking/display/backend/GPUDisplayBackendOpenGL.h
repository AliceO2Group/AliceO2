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

namespace GPUCA_NAMESPACE::gpu
{
struct GLfb {
  uint32_t fb_id = 0, fbCol_id = 0, fbDepth_id = 0;
  bool tex = false;
  bool msaa = false;
  bool depth = false;
  bool created = false;
};
class GPUDisplayBackendOpenGL : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendOpenGL();
  ~GPUDisplayBackendOpenGL() override = default;
  int32_t ExtInit() override;
  bool CoreProfile() override;
  uint32_t DepthBits() override;

 protected:
  void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa, uint32_t width, uint32_t height);
  void deleteFB(GLfb& fb);

  uint32_t drawVertices(const vboList& v, const drawType t) override;
  uint32_t drawField() override;
  void ActivateColor(std::array<float, 4>& color) override;
  void setQuality() override;
  void setDepthBuffer() override;
  void setFrameBuffer(uint32_t newID = 0);
  int32_t InitBackendA() override;
  int32_t InitMagFieldVisualization();
  void ExitBackendA() override;
  void ExitMagFieldVisualization();
  static int32_t checkShaderStatus(uint32_t shader);
  static int32_t checkProgramStatus(uint32_t program);
  void clearScreen(bool alphaOnly = false);
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) override;
  void resizeScene(uint32_t width, uint32_t height) override;
  void updateRenderer(bool withScreenshot);
  void ClearOffscreenBuffers();
  void finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void prepareText() override;
  void finishText() override;
  void mixImages(float mixSlaveImage);
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;
  size_t needMultiVBO() override { return 0x100000000ll; }
  void readImageToPixels();

  void addFontSymbol(int32_t symbol, int32_t sizex, int32_t sizey, int32_t offsetx, int32_t offsety, int32_t advance, void* data) override;
  void initializeTextDrawing() override;
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override;

  struct FontSymbolOpenGL : public FontSymbol {
    uint32_t texId;
  };

  uint32_t mVertexShader;
  uint32_t mFragmentShader;
  uint32_t mVertexShaderTexture;
  uint32_t mVertexShaderPassthrough;
  uint32_t mFragmentShaderTexture;
  uint32_t mFragmentShaderText;
  uint32_t mGeometryShader;
  uint32_t mShaderProgram;
  uint32_t mShaderProgramText;
  uint32_t mShaderProgramTexture;
  uint32_t mShaderProgramField;
  uint32_t mVertexArray;

  uint32_t mIndirectId;
  std::vector<uint32_t> mVBOId;
  std::vector<FontSymbolOpenGL> mFontSymbols;
  int32_t mModelViewProjId;
  int32_t mColorId;
  int32_t mModelViewProjIdTexture;
  int32_t mAlphaIdTexture;
  int32_t mModelViewProjIdText;
  int32_t mColorIdText;
  uint32_t mSPIRVModelViewBuffer;
  uint32_t mSPIRVColorBuffer;

  uint32_t mFieldModelViewBuffer;
  uint32_t mFieldModelConstantsBuffer;
  uint32_t mSolenoidSegmentsBuffer;
  uint32_t mSolenoidParameterizationBuffer;
  uint32_t mDipoleSegmentsBuffer;
  uint32_t mDipoleParameterizationBuffer;

  uint32_t VAO_text, VBO_text;

  uint32_t VAO_texture, VBO_texture;

  uint32_t VAO_field, VBO_field;

  bool mSPIRVShaders = false;

  GLfb mMixBuffer;
  GLfb mOffscreenBufferMSAA, mOffscreenBuffer;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
