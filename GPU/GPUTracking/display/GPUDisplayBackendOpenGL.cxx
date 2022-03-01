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

/// \file GPUDisplayBackendOpenGL.cxx
/// \author David Rohr

#ifdef GPUCA_DISPLAY_GL3W
#include "GL/gl3w.h"
#else
#include <GL/glew.h>
#endif
#include <GL/glu.h>

#include "GPUCommonDef.h"
#include "GPUDisplayBackendOpenGL.h"
#include "GPUDisplayShaders.h"
#include "GPUDisplay.h"

#define OPENGL_EMULATE_MULTI_DRAW 0

using namespace GPUCA_NAMESPACE::gpu;

// Runtime minimum version defined in GPUDisplayFrontend.h, keep in sync!
#if !defined(GL_VERSION_4_5) || GL_VERSION_4_5 != 1
#ifdef GPUCA_STANDALONE
#error Unsupported OpenGL version < 4.5
#elif defined(GPUCA_O2_LIB)
#pragma message "Unsupported OpenGL version < 4.5, disabling standalone event display"
#else
#warning Unsupported OpenGL version < 4.5, disabling standalone event display
#endif
#undef GPUCA_BUILD_EVENT_DISPLAY
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY

#ifdef GPUCA_DISPLAY_GL3W
int GPUDisplayBackendOpenGL::ExtInit()
{
  return gl3wInit();
}
#else
int GPUDisplayBackendOpenGL::ExtInit()
{
  return glewInit();
}
#endif
#ifdef GPUCA_DISPLAY_OPENGL_CORE
bool GPUDisplayBackendOpenGL::CoreProfile()
{
  return true;
}
#else
bool GPUDisplayBackendOpenGL::CoreProfile()
{
  return false;
}
#endif

//#define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                           \
  do {                                                                                        \
    (cmd);                                                                                    \
    GLenum err = glGetError();                                                                \
    while (err != GL_NO_ERROR) {                                                              \
      GPUError("OpenGL Error %d: %s (%s: %d)", err, gluErrorString(err), __FILE__, __LINE__); \
      throw std::runtime_error("OpenGL Failure");                                             \
    }                                                                                         \
  } while (false)

unsigned int GPUDisplayBackendOpenGL::DepthBits()
{
  GLint depthBits = 0;
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);
#endif
  return depthBits;
}

void GPUDisplayBackendOpenGL::createFB(GLfb& fb, bool tex, bool withDepth, bool msaa)
{
  fb.tex = tex;
  fb.depth = withDepth;
  fb.msaa = msaa;
  GLint drawFboId = 0, readFboId = 0;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFboId);
  CHKERR(glGenFramebuffers(1, &fb.fb_id));
  CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, fb.fb_id));

  auto createFB_texture = [&](GLuint& id, GLenum storage, GLenum attachment) {
    GLenum textureType = fb.msaa ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
    CHKERR(glGenTextures(1, &id));
    CHKERR(glBindTexture(textureType, id));
    if (fb.msaa) {
      CHKERR(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, mDisplay->cfgR().drawQualityMSAA, storage, mDisplay->renderWidth(), mDisplay->renderHeight(), false));
    } else {
      CHKERR(glTexImage2D(GL_TEXTURE_2D, 0, storage, mDisplay->renderWidth(), mDisplay->renderHeight(), 0, storage, GL_UNSIGNED_BYTE, nullptr));
      CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
      CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    }
    CHKERR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textureType, id, 0));
  };

  auto createFB_renderbuffer = [&](GLuint& id, GLenum storage, GLenum attachment) {
    CHKERR(glGenRenderbuffers(1, &id));
    CHKERR(glBindRenderbuffer(GL_RENDERBUFFER, id));
    if (fb.msaa) {
      CHKERR(glRenderbufferStorageMultisample(GL_RENDERBUFFER, mDisplay->cfgR().drawQualityMSAA, storage, mDisplay->renderWidth(), mDisplay->renderHeight()));
    } else {
      CHKERR(glRenderbufferStorage(GL_RENDERBUFFER, storage, mDisplay->renderWidth(), mDisplay->renderHeight()));
    }
    CHKERR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, id));
  };

  if (tex) {
    createFB_texture(fb.fbCol_id, GL_RGBA, GL_COLOR_ATTACHMENT0);
  } else {
    createFB_renderbuffer(fb.fbCol_id, GL_RGBA, GL_COLOR_ATTACHMENT0);
  }

  if (withDepth) {
    if (tex && fb.msaa) {
      createFB_texture(fb.fbDepth_id, GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
    } else {
      createFB_renderbuffer(fb.fbDepth_id, GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
    }
  }

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    GPUError("Error creating framebuffer (tex %d) - incomplete (%d)", (int)tex, status);
    exit(1);
  }
  CHKERR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFboId));
  CHKERR(glBindFramebuffer(GL_READ_FRAMEBUFFER, readFboId));
  fb.created = true;
}

void GPUDisplayBackendOpenGL::deleteFB(GLfb& fb)
{
  if (fb.tex) {
    CHKERR(glDeleteTextures(1, &fb.fbCol_id));
  } else {
    CHKERR(glDeleteRenderbuffers(1, &fb.fbCol_id));
  }
  if (fb.depth) {
    if (fb.tex && fb.msaa) {
      CHKERR(glDeleteTextures(1, &fb.fbDepth_id));
    } else {
      CHKERR(glDeleteRenderbuffers(1, &fb.fbDepth_id));
    }
  }
  CHKERR(glDeleteFramebuffers(1, &fb.fb_id));
  fb.created = false;
}

unsigned int GPUDisplayBackendOpenGL::drawVertices(const vboList& v, const drawType tt)
{
  static constexpr GLenum types[3] = {GL_POINTS, GL_LINES, GL_LINE_STRIP};
  GLenum t = types[tt];
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSlice = std::get<2>(v);
  if (count == 0) {
    return 0;
  }

  if (mDisplay->useMultiVBO()) {
    if (mDisplay->cfgR().openGLCore) {
      CHKERR(glBindVertexArray(mVertexArray));
    }
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[iSlice]));
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    if (!mDisplay->cfgR().openGLCore) {
      CHKERR(glVertexPointer(3, GL_FLOAT, 0, nullptr));
    } else
#endif
    {
      CHKERR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
      glEnableVertexAttribArray(0);
    }
  }

  if (mDisplay->cfgR().useGLIndirectDraw) {
    CHKERR(glMultiDrawArraysIndirect(t, (void*)(size_t)((mIndirectSliceOffset[iSlice] + first) * sizeof(DrawArraysIndirectCommand)), count, 0));
  } else if (OPENGL_EMULATE_MULTI_DRAW) {
    for (unsigned int k = 0; k < count; k++) {
      CHKERR(glDrawArrays(t, mDisplay->vertexBufferStart()[iSlice][first + k], mDisplay->vertexBufferCount()[iSlice][first + k]));
    }
  } else {
    static_assert(sizeof(GLsizei) == sizeof(*mDisplay->vertexBufferCount()[iSlice].data()), "Invalid counter size does not match GLsizei");
    CHKERR(glMultiDrawArrays(t, mDisplay->vertexBufferStart()[iSlice].data() + first, ((const GLsizei*)mDisplay->vertexBufferCount()[iSlice].data()) + first, count));
  }
  return count;
}

void GPUDisplayBackendOpenGL::ActivateColor(std::array<float, 3>& color)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glColor3f(color[0], color[1], color[2]);
  } else
#endif
  {
    glUniform3fv(mColorId, 1, &color[0]);
  }
}

void GPUDisplayBackendOpenGL::setQuality()
{
  // Doesn't seem to make a difference in this applicattion
  if (mDisplay->cfgR().drawQualityMSAA > 1) {
    CHKERR(glEnable(GL_MULTISAMPLE));
  } else {
    CHKERR(glDisable(GL_MULTISAMPLE));
  }
}

void GPUDisplayBackendOpenGL::setDepthBuffer()
{
  if (mDisplay->cfgL().depthBuffer) {
    CHKERR(glEnable(GL_DEPTH_TEST)); // Enables Depth Testing
    CHKERR(glDepthFunc(GL_LEQUAL));  // The Type Of Depth Testing To Do
  } else {
    CHKERR(glDisable(GL_DEPTH_TEST));
  }
}

void GPUDisplayBackendOpenGL::setFrameBuffer(int updateCurrent, unsigned int newID)
{
  if (newID == 0) {
    CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glDrawBuffer(GL_BACK);
  } else {
    CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, newID));
    GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
  }
}

int GPUDisplayBackendOpenGL::InitBackend()
{
  int glVersion[2] = {0, 0};
  glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
  glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
  if (glVersion[0] < GPUDisplayFrontend::GL_MIN_VERSION_MAJOR || (glVersion[0] == GPUDisplayFrontend::GL_MIN_VERSION_MAJOR && glVersion[1] < GPUDisplayFrontend::GL_MIN_VERSION_MINOR)) {
    GPUError("Unsupported OpenGL runtime %d.%d < %d.%d", glVersion[0], glVersion[1], GPUDisplayFrontend::GL_MIN_VERSION_MAJOR, GPUDisplayFrontend::GL_MIN_VERSION_MINOR);
    return (1);
  }
  mIndirectSliceOffset.resize(GPUCA_NSLICES);
  mVBOId.resize(GPUCA_NSLICES);
  CHKERR(glCreateBuffers(mVBOId.size(), mVBOId.data()));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0]));
  CHKERR(glGenBuffers(1, &mIndirectId));
  CHKERR(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, mIndirectId));
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  CHKERR(glShadeModel(GL_SMOOTH)); // Enable Smooth Shading
#endif
  setDepthBuffer();
  setQuality();
  CHKERR(mVertexShader = glCreateShader(GL_VERTEX_SHADER));
  CHKERR(glShaderSource(mVertexShader, 1, &GPUDisplayShaders::vertexShader, nullptr));
  CHKERR(glCompileShader(mVertexShader));
  CHKERR(mFragmentShader = glCreateShader(GL_FRAGMENT_SHADER));
  CHKERR(glShaderSource(mFragmentShader, 1, &GPUDisplayShaders::fragmentShader, nullptr));
  CHKERR(glCompileShader(mFragmentShader));
  CHKERR(mShaderProgram = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgram, mVertexShader));
  CHKERR(glAttachShader(mShaderProgram, mFragmentShader));
  CHKERR(glLinkProgram(mShaderProgram));
  CHKERR(glGenVertexArrays(1, &mVertexArray));
  CHKERR(mModelViewProjId = glGetUniformLocation(mShaderProgram, "ModelViewProj"));
  CHKERR(mColorId = glGetUniformLocation(mShaderProgram, "color"));
  return (0); // Initialization Went OK
}

void GPUDisplayBackendOpenGL::ExitBackend()
{
  CHKERR(glDeleteBuffers(mVBOId.size(), mVBOId.data()));
  CHKERR(glDeleteBuffers(1, &mIndirectId));
  CHKERR(glDeleteProgram(mShaderProgram));
  CHKERR(glDeleteShader(mVertexShader));
  CHKERR(glDeleteShader(mFragmentShader));
}

void GPUDisplayBackendOpenGL::clearScreen(bool colorOnly)
{
  if (mDisplay->cfgL().invertColors && !colorOnly) {
    CHKERR(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
  } else {
    CHKERR(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
  }
  if (colorOnly) {
    glColorMask(false, false, false, true);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(true, true, true, true);
  } else {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
  }
}

void GPUDisplayBackendOpenGL::updateSettings()
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (mDisplay->cfgL().smoothPoints && !mDisplay->cfgR().openGLCore) {
    CHKERR(glEnable(GL_POINT_SMOOTH));
  } else {
    CHKERR(glDisable(GL_POINT_SMOOTH));
  }
  if (mDisplay->cfgL().smoothLines && !mDisplay->cfgR().openGLCore) {
    CHKERR(glEnable(GL_LINE_SMOOTH));
  } else {
    CHKERR(glDisable(GL_LINE_SMOOTH));
  }
#endif
  CHKERR(glEnable(GL_BLEND));
  CHKERR(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  pointSizeFactor(1);
  lineWidthFactor(1);
}

void GPUDisplayBackendOpenGL::loadDataToGPU(size_t totalVertizes)
{
  // TODO: Check if this can be parallelized
  if (mDisplay->useMultiVBO()) {
    for (int i = 0; i < GPUCA_NSLICES; i++) {
      CHKERR(glNamedBufferData(mVBOId[i], mDisplay->vertexBuffer()[i].size() * sizeof(mDisplay->vertexBuffer()[i][0]), mDisplay->vertexBuffer()[i].data(), GL_STATIC_DRAW));
    }
  } else {
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0])); // Bind ahead of time, since it is not going to change
    CHKERR(glNamedBufferData(mVBOId[0], totalVertizes * sizeof(mDisplay->vertexBuffer()[0][0]), mDisplay->vertexBuffer()[0].data(), GL_STATIC_DRAW));
  }

  if (mDisplay->cfgR().useGLIndirectDraw) {
    mCmdBuffer.clear();
    // TODO: Check if this can be parallelized
    for (int iSlice = 0; iSlice < GPUCA_NSLICES; iSlice++) {
      mIndirectSliceOffset[iSlice] = mCmdBuffer.size();
      for (unsigned int k = 0; k < mDisplay->vertexBufferStart()[iSlice].size(); k++) {
        mCmdBuffer.emplace_back(mDisplay->vertexBufferCount()[iSlice][k], 1, mDisplay->vertexBufferStart()[iSlice][k], 0);
      }
    }
    CHKERR(glBufferData(GL_DRAW_INDIRECT_BUFFER, mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), GL_STATIC_DRAW));
    mCmdBuffer.clear();
  }
}

void GPUDisplayBackendOpenGL::prepareDraw()
{
  glViewport(0, 0, mDisplay->renderWidth(), mDisplay->renderHeight());
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    CHKERR(glEnableClientState(GL_VERTEX_ARRAY));
    CHKERR(glVertexPointer(3, GL_FLOAT, 0, nullptr));
  } else
#endif
  {
    CHKERR(glBindVertexArray(mVertexArray));
    CHKERR(glUseProgram(mShaderProgram));
    CHKERR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
    CHKERR(glEnableVertexAttribArray(0));
  }
}

void GPUDisplayBackendOpenGL::finishDraw()
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    CHKERR(glDisableClientState(GL_VERTEX_ARRAY));
  } else
#endif
  {
    CHKERR(glDisableVertexAttribArray(0));
    CHKERR(glUseProgram(0));
  }
}

void GPUDisplayBackendOpenGL::prepareText()
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    hmm_mat4 proj = HMM_Orthographic(0.f, mDisplay->screenWidth(), 0.f, mDisplay->screenHeight(), -1, 1);
    glLoadMatrixf(&proj.Elements[0][0]);
    glViewport(0, 0, mDisplay->screenWidth(), mDisplay->screenHeight());
  }
#endif
}

void GPUDisplayBackendOpenGL::renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer)
{
  GLuint srcid = buffer.fb_id;
  if (mDisplay->cfgR().drawQualityMSAA > 1 && mDisplay->cfgR().drawQualityDownsampleFSAA > 1) {
    CHKERR(glBlitNamedFramebuffer(srcid, bufferNoMSAA.fb_id, 0, 0, mDisplay->renderWidth(), mDisplay->renderHeight(), 0, 0, mDisplay->renderWidth(), mDisplay->renderHeight(), GL_COLOR_BUFFER_BIT, GL_LINEAR));
    srcid = bufferNoMSAA.fb_id;
  }
  CHKERR(glBlitNamedFramebuffer(srcid, mainBuffer, 0, 0, mDisplay->renderWidth(), mDisplay->renderHeight(), 0, 0, mDisplay->screenWidth(), mDisplay->screenHeight(), GL_COLOR_BUFFER_BIT, GL_LINEAR));
}

void GPUDisplayBackendOpenGL::setMatrices(const hmm_mat4& proj, const hmm_mat4& view)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    CHKERR(glMatrixMode(GL_PROJECTION));
    CHKERR(glLoadMatrixf(&proj.Elements[0][0]));
    CHKERR(glMatrixMode(GL_MODELVIEW));
    CHKERR(glLoadMatrixf(&view.Elements[0][0]));
  } else
#endif
  {
    const hmm_mat4 modelViewProj = proj * view;
    CHKERR(glUniformMatrix4fv(mModelViewProjId, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
  }
}

void GPUDisplayBackendOpenGL::mixImages(GLfb& mixBuffer, float mixSlaveImage)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    hmm_mat4 proj = HMM_Orthographic(0.f, mDisplay->renderWidth(), 0.f, mDisplay->renderHeight(), -1.f, 1.f);
    glLoadMatrixf(&proj.Elements[0][0]);
    CHKERR(glEnable(GL_TEXTURE_2D));
    glDisable(GL_DEPTH_TEST);
    CHKERR(glBindTexture(GL_TEXTURE_2D, mixBuffer.fbCol_id));
    glColor4f(1, 1, 1, mixSlaveImage);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, 1);
    glVertex3f(0, mDisplay->renderHeight(), 0);
    glTexCoord2f(1, 1);
    glVertex3f(mDisplay->renderWidth(), mDisplay->renderHeight(), 0);
    glTexCoord2f(1, 0);
    glVertex3f(mDisplay->renderWidth(), 0, 0);
    glEnd();
    glColor4f(1, 1, 1, 0);
    CHKERR(glDisable(GL_TEXTURE_2D));
    setDepthBuffer();
  } else
#endif
  {
    GPUWarning("Image mixing unsupported in OpenGL CORE profile");
  }
}

void GPUDisplayBackendOpenGL::readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height)
{
  CHKERR(glPixelStorei(GL_PACK_ALIGNMENT, 1));
  CHKERR(glReadBuffer(needBuffer ? GL_COLOR_ATTACHMENT0 : GL_BACK));
  CHKERR(glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixels));
}

void GPUDisplayBackendOpenGL::pointSizeFactor(float factor)
{
  CHKERR(glPointSize(mDisplay->cfgL().pointSize * (mDisplay->cfgR().drawQualityDownsampleFSAA > 1 ? mDisplay->cfgR().drawQualityDownsampleFSAA : 1) * factor));
}

void GPUDisplayBackendOpenGL::lineWidthFactor(float factor)
{
  CHKERR(glLineWidth(mDisplay->cfgL().lineWidth * (mDisplay->cfgR().drawQualityDownsampleFSAA > 1 ? mDisplay->cfgR().drawQualityDownsampleFSAA : 1) * factor));
}

#endif // GPUCA_BUILD_EVENT_DISPLAY
