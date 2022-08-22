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
  unsigned int fb_id = 0, fbCol_id = 0, fbDepth_id = 0;
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
  int ExtInit() override;
  bool CoreProfile() override;
  unsigned int DepthBits() override;

 protected:
  void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa, unsigned int width, unsigned int height);
  void deleteFB(GLfb& fb);

  unsigned int drawVertices(const vboList& v, const drawType t) override;
  unsigned int drawField() override;
  void ActivateColor(std::array<float, 4>& color) override;
  void setQuality() override;
  void setDepthBuffer() override;
  void setFrameBuffer(unsigned int newID = 0);
  int InitBackendA() override;
  int InitMagField();
  void ExitBackendA() override;
  void ExitMagField();
  static int checkShaderStatus(unsigned int shader);
  static int checkProgramStatus(unsigned int program);
  void clearScreen(bool alphaOnly = false);
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) override;
  void resizeScene(unsigned int width, unsigned int height) override;
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

  void addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) override;
  void initializeTextDrawing() override;
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override;

  struct FontSymbolOpenGL : public FontSymbol {
    unsigned int texId;
  };

  unsigned int mVertexShader;
  unsigned int mFragmentShader;
  unsigned int mVertexShaderTexture;
  unsigned int mVertexShaderPassthrough;
  unsigned int mFragmentShaderTexture;
  unsigned int mFragmentShaderText;
  unsigned int mGeometryShader;
  unsigned int mShaderProgram;
  unsigned int mShaderProgramText;
  unsigned int mShaderProgramTexture;
  unsigned int mShaderProgramField;
  unsigned int mVertexArray;

  unsigned int mIndirectId;
  std::vector<unsigned int> mVBOId;
  std::vector<FontSymbolOpenGL> mFontSymbols;
  int mModelViewProjId;
  int mColorId;
  int mModelViewProjIdTexture;
  int mAlphaIdTexture;
  int mModelViewProjIdText;
  int mColorIdText;
  unsigned int mSPIRVModelViewBuffer;
  unsigned int mSPIRVColorBuffer;

  unsigned int mFieldModelViewBuffer;
  unsigned int mFieldModelConstantsBuffer;
  unsigned int mSolenoidSegmentsBuffer;
  unsigned int mSolenoidParameterizationBuffer;
  unsigned int mDipoleSegmentsBuffer;
  unsigned int mDipoleParameterizationBuffer;

  unsigned int VAO_text, VBO_text;

  unsigned int VAO_texture, VBO_texture;

  unsigned int VAO_field, VBO_field;

  bool mSPIRVShaders = false;

  GLfb mMixBuffer;
  GLfb mOffscreenBufferMSAA, mOffscreenBuffer;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
