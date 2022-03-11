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

/// \file GPUDisplayBackend.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKEND_H
#define GPUDISPLAYBACKEND_H

#include "GPUCommonDef.h"
#include "../utils/vecpod.h"
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#if defined(GPUCA_DISPLAY_GL3W) && !defined(GPUCA_DISPLAY_OPENGL_CORE)
#define GPUCA_DISPLAY_OPENGL_CORE
#endif

union hmm_mat4;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplay;
class GPUDisplayFrontend;
class GPUDisplayBackend
{
  friend GPUDisplay;

 public:
  GPUDisplayBackend();
  virtual ~GPUDisplayBackend();

  virtual int ExtInit() = 0;
  virtual bool CoreProfile() = 0;
  virtual unsigned int DepthBits() = 0;

  typedef std::tuple<unsigned int, unsigned int, int> vboList;

  enum drawType {
    POINTS = 0,
    LINES = 1,
    LINE_STRIP = 2
  };

  struct GLfb {
    unsigned int fb_id = 0, fbCol_id = 0, fbDepth_id = 0;
    bool tex = false;
    bool msaa = false;
    bool depth = false;
    bool created = false;
  };

  enum backendTypes {
    TYPE_OPENGL = 0,
    TYPE_VULKAN = 1
  };

  struct DrawArraysIndirectCommand {
    DrawArraysIndirectCommand(unsigned int a = 0, unsigned int b = 0, unsigned int c = 0, unsigned int d = 0) : count(a), instanceCount(b), first(c), baseInstance(d) {}
    unsigned int count;
    unsigned int instanceCount;

    unsigned int first;
    unsigned int baseInstance;
  };

  struct FontSymbol {
    int size[2];
    int offset[2];
    int advance;
  };

  virtual void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa) = 0;
  virtual void deleteFB(GLfb& fb) = 0;

  virtual unsigned int drawVertices(const vboList& v, const drawType t) = 0;
  virtual void ActivateColor(std::array<float, 4>& color) = 0;
  virtual void setQuality() = 0;
  virtual void setDepthBuffer() = 0;
  virtual void setFrameBuffer(int updateCurrent, unsigned int newID) = 0;
  virtual int InitBackendA() = 0;
  virtual void ExitBackendA() = 0;
  int InitBackend();
  void ExitBackend();
  virtual void clearScreen(bool colorOnly = false) = 0;
  virtual void updateSettings() = 0;
  virtual void loadDataToGPU(size_t totalVertizes) = 0;
  virtual void prepareDraw() = 0;
  virtual void finishDraw() = 0;
  virtual void finishFrame() = 0;
  virtual void prepareText() = 0;
  virtual void finishText() = 0;
  virtual void setMatrices(const hmm_mat4& proj, const hmm_mat4& view) = 0;
  virtual void mixImages(GLfb& mixBuffer, float mixSlaveImage) = 0;
  virtual void renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer) = 0;
  virtual void readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height) = 0;
  virtual void pointSizeFactor(float factor) = 0;
  virtual void lineWidthFactor(float factor) = 0;
  virtual backendTypes backendType() const = 0;
  virtual void resizeScene(unsigned int width, unsigned int height) {}
  virtual size_t needMultiVBO() { return 0; }
  virtual void OpenGLPrint(const char* s, float x, float y, float* color, float scale) = 0;
  static GPUDisplayBackend* getBackend(const char* type);

 protected:
  GPUDisplay* mDisplay = nullptr;
  std::vector<int> mIndirectSliceOffset;
  vecpod<DrawArraysIndirectCommand> mCmdBuffer;
  void fillIndirectCmdBuffer();
  virtual void addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) = 0;
  virtual void initializeTextDrawing() = 0;
  bool mFreetypeInitialized = false;
  bool mFrontendCompatTextDraw = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
