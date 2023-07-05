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
#if __has_include(<GL/glu.h>)
#include <GL/glu.h>
#else
#define gluErrorString(err) ""
#endif

#include "GPUCommonDef.h"
#include "GPUDisplayBackendOpenGL.h"
#include "GPUDisplayShaders.h"
#include "GPUDisplay.h"

#define OPENGL_EMULATE_MULTI_DRAW 0

using namespace GPUCA_NAMESPACE::gpu;

#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(shaders_shaders_vertex_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_fragmentUniform_frag_spv);
#endif

// Runtime minimum version defined in GPUDisplayFrontend.h, keep in sync!
#define GPUCA_BUILD_EVENT_DISPLAY_OPENGL
#if !defined(GL_VERSION_4_5) || GL_VERSION_4_5 != 1
#ifdef GPUCA_STANDALONE
//#error Unsupported OpenGL version < 4.5
#elif defined(GPUCA_O2_LIB)
#pragma message "Unsupported OpenGL version < 4.5, disabling standalone event display"
#else
#warning Unsupported OpenGL version < 4.5, disabling standalone event display
#endif
#undef GPUCA_BUILD_EVENT_DISPLAY_OPENGL
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY_OPENGL

GPUDisplayBackendOpenGL::GPUDisplayBackendOpenGL()
{
  mBackendType = TYPE_OPENGL;
  mBackendName = "OpenGL";
}

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
#define CHKERR(cmd)                                                                                             \
  do {                                                                                                          \
    (cmd);                                                                                                      \
    GLenum err = glGetError();                                                                                  \
    while (err != GL_NO_ERROR) {                                                                                \
      GPUError("OpenGL Error %d: %s (%s: %d)", (int)err, (const char*)gluErrorString(err), __FILE__, __LINE__); \
      throw std::runtime_error("OpenGL Failure");                                                               \
    }                                                                                                           \
  } while (false)

unsigned int GPUDisplayBackendOpenGL::DepthBits()
{
  GLint depthBits = 0;
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  glGetIntegerv(GL_DEPTH_BITS, &depthBits);
#endif
  return depthBits;
}

void GPUDisplayBackendOpenGL::createFB(GLfb& fb, bool tex, bool withDepth, bool msaa, unsigned int width, unsigned int height)
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
      CHKERR(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, mDisplay->cfgR().drawQualityMSAA, storage, width, height, false));
    } else {
      CHKERR(glTexImage2D(GL_TEXTURE_2D, 0, storage, width, height, 0, storage, GL_UNSIGNED_BYTE, nullptr));
      CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
      CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    }
    CHKERR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textureType, id, 0));
  };

  auto createFB_renderbuffer = [&](GLuint& id, GLenum storage, GLenum attachment) {
    CHKERR(glGenRenderbuffers(1, &id));
    CHKERR(glBindRenderbuffer(GL_RENDERBUFFER, id));
    if (fb.msaa) {
      CHKERR(glRenderbufferStorageMultisample(GL_RENDERBUFFER, mDisplay->cfgR().drawQualityMSAA, storage, width, height));
    } else {
      CHKERR(glRenderbufferStorage(GL_RENDERBUFFER, storage, width, height));
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

void GPUDisplayBackendOpenGL::ActivateColor(std::array<float, 4>& color)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glColor4f(color[0], color[1], color[2], color[3]);
  } else
#endif
  {
    if (mSPIRVShaders) {
      CHKERR(glBindBuffer(GL_UNIFORM_BUFFER, mSPIRVColorBuffer));
      CHKERR(glBufferData(GL_UNIFORM_BUFFER, sizeof(color), &color[0], GL_STATIC_DRAW));
      CHKERR(glBindBufferBase(GL_UNIFORM_BUFFER, 1, mSPIRVColorBuffer));
    } else {
      CHKERR(glUniform4fv(mColorId, 1, &color[0]));
    }
  }
}

void GPUDisplayBackendOpenGL::setQuality()
{
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

void GPUDisplayBackendOpenGL::setFrameBuffer(unsigned int newID)
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

int GPUDisplayBackendOpenGL::checkShaderStatus(unsigned int shader)
{
  int status, loglen;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status) {
    printf("failed to compile shader\n");
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &loglen);
    std::vector<char> buf(loglen + 1);
    glGetShaderInfoLog(shader, loglen, nullptr, buf.data());
    buf[loglen] = 0;
    printf("%s\n", buf.data());
    return 1;
  }
  return 0;
}

int GPUDisplayBackendOpenGL::checkProgramStatus(unsigned int program)
{
  int status, loglen;
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (!status) {
    printf("failed to link program\n");
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &loglen);
    std::vector<char> buf(loglen + 1);
    glGetProgramInfoLog(program, loglen, nullptr, buf.data());
    buf[loglen] = 0;
    printf("%s\n", buf.data());
    return 1;
  }
  return 0;
}

int GPUDisplayBackendOpenGL::InitBackendA()
{
  if (mDisplay->param()->par.debugLevel >= 2) {
    auto renderer = glGetString(GL_RENDERER);
    GPUInfo("Renderer: %s", (const char*)renderer);
  }

  int glVersion[2] = {0, 0};
  glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
  glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
  if (glVersion[0] < GPUDisplayFrontend::GL_MIN_VERSION_MAJOR || (glVersion[0] == GPUDisplayFrontend::GL_MIN_VERSION_MAJOR && glVersion[1] < GPUDisplayFrontend::GL_MIN_VERSION_MINOR)) {
    GPUError("Unsupported OpenGL runtime %d.%d < %d.%d", glVersion[0], glVersion[1], GPUDisplayFrontend::GL_MIN_VERSION_MAJOR, GPUDisplayFrontend::GL_MIN_VERSION_MINOR);
    return (1);
  }
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
  CHKERR(mFragmentShader = glCreateShader(GL_FRAGMENT_SHADER));
  CHKERR(mVertexShaderTexture = glCreateShader(GL_VERTEX_SHADER));
  CHKERR(mFragmentShaderTexture = glCreateShader(GL_FRAGMENT_SHADER));
  CHKERR(mFragmentShaderText = glCreateShader(GL_FRAGMENT_SHADER));
#if defined(GL_VERSION_4_6) && GL_VERSION_4_6 == 1 && defined(GPUCA_BUILD_EVENT_DISPLAY_VULKAN)
  if (getenv("USE_SPIRV_SHADERS") && atoi(getenv("USE_SPIRV_SHADERS"))) {
    CHKERR(glShaderBinary(1, &mVertexShader, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, _binary_shaders_shaders_vertex_vert_spv_start, _binary_shaders_shaders_vertex_vert_spv_len));
    CHKERR(glSpecializeShader(mVertexShader, "main", 0, 0, 0));
    CHKERR(glShaderBinary(1, &mFragmentShader, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, _binary_shaders_shaders_fragmentUniform_frag_spv_start, _binary_shaders_shaders_fragmentUniform_frag_spv_len));
    CHKERR(glSpecializeShader(mFragmentShader, "main", 0, 0, 0));
    GPUInfo("Using SPIR-V shaders");
    mSPIRVShaders = true;
  } else
#endif
  {
    CHKERR(glShaderSource(mVertexShader, 1, &GPUDisplayShaders::vertexShader, nullptr));
    CHKERR(glCompileShader(mVertexShader));
    CHKERR(glShaderSource(mFragmentShader, 1, &GPUDisplayShaders::fragmentShader, nullptr));
    CHKERR(glCompileShader(mFragmentShader));
  }
  CHKERR(glShaderSource(mVertexShaderTexture, 1, &GPUDisplayShaders::vertexShaderTexture, nullptr));
  CHKERR(glCompileShader(mVertexShaderTexture));
  CHKERR(glShaderSource(mFragmentShaderTexture, 1, &GPUDisplayShaders::fragmentShaderTexture, nullptr));
  CHKERR(glCompileShader(mFragmentShaderTexture));
  CHKERR(glShaderSource(mFragmentShaderText, 1, &GPUDisplayShaders::fragmentShaderText, nullptr));
  CHKERR(glCompileShader(mFragmentShaderText));
  if (checkShaderStatus(mVertexShader) || checkShaderStatus(mFragmentShader) || checkShaderStatus(mVertexShaderTexture) || checkShaderStatus(mFragmentShaderTexture) || checkShaderStatus(mFragmentShaderText)) {
    return 1;
  }
  CHKERR(mShaderProgram = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgram, mVertexShader));
  CHKERR(glAttachShader(mShaderProgram, mFragmentShader));
  CHKERR(glLinkProgram(mShaderProgram));
  if (checkProgramStatus(mShaderProgram)) {
    return 1;
  }
  if (mSPIRVShaders) {
    CHKERR(glGenBuffers(1, &mSPIRVModelViewBuffer));
    CHKERR(glGenBuffers(1, &mSPIRVColorBuffer));
  } else {
    CHKERR(mModelViewProjId = glGetUniformLocation(mShaderProgram, "ModelViewProj"));
    CHKERR(mColorId = glGetUniformLocation(mShaderProgram, "color"));
  }
  CHKERR(mShaderProgramText = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramText, mVertexShaderTexture));
  CHKERR(glAttachShader(mShaderProgramText, mFragmentShaderText));
  CHKERR(glLinkProgram(mShaderProgramText));
  if (checkProgramStatus(mShaderProgramText)) {
    return 1;
  }
  CHKERR(mModelViewProjIdText = glGetUniformLocation(mShaderProgramText, "projection"));
  CHKERR(mColorIdText = glGetUniformLocation(mShaderProgramText, "textColor"));
  CHKERR(mShaderProgramTexture = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramTexture, mVertexShaderTexture));
  CHKERR(glAttachShader(mShaderProgramTexture, mFragmentShaderTexture));
  CHKERR(glLinkProgram(mShaderProgramTexture));
  if (checkProgramStatus(mShaderProgramTexture)) {
    return 1;
  }
  CHKERR(mModelViewProjIdTexture = glGetUniformLocation(mShaderProgramTexture, "projection"));
  CHKERR(mAlphaIdTexture = glGetUniformLocation(mShaderProgramTexture, "alpha"));
  CHKERR(glGenVertexArrays(1, &mVertexArray));

  CHKERR(glGenVertexArrays(1, &VAO_texture));
  CHKERR(glGenBuffers(1, &VBO_texture));
  CHKERR(glBindVertexArray(VAO_texture));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, VBO_texture));
  CHKERR(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW));
  CHKERR(glEnableVertexAttribArray(0));
  CHKERR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, 0));

  CHKERR(glBindVertexArray(0));
  return 0;
}

void GPUDisplayBackendOpenGL::ExitBackendA()
{
  ClearOffscreenBuffers();
  CHKERR(glDeleteBuffers(mVBOId.size(), mVBOId.data()));
  CHKERR(glDeleteBuffers(1, &mIndirectId));
  CHKERR(glDeleteProgram(mShaderProgram));
  CHKERR(glDeleteShader(mVertexShader));
  CHKERR(glDeleteShader(mFragmentShader));
  CHKERR(glDeleteProgram(mShaderProgramText));
  CHKERR(glDeleteShader(mFragmentShaderText));
  CHKERR(glDeleteProgram(mShaderProgramTexture));
  CHKERR(glDeleteShader(mVertexShaderTexture));
  CHKERR(glDeleteShader(mFragmentShaderTexture));

  CHKERR(glDeleteVertexArrays(1, &mVertexArray));

  CHKERR(glDeleteBuffers(1, &VBO_texture));
  CHKERR(glDeleteVertexArrays(1, &VAO_texture));

  if (mFreetypeInitialized) {
    CHKERR(glDeleteBuffers(1, &VBO_text));
    CHKERR(glDeleteVertexArrays(1, &VAO_text));
    for (auto& symbol : mFontSymbols) {
      CHKERR(glDeleteTextures(1, &symbol.texId));
    }
  }
  if (mSPIRVShaders) {
    CHKERR(glDeleteBuffers(1, &mSPIRVModelViewBuffer));
    CHKERR(glDeleteBuffers(1, &mSPIRVColorBuffer));
  }
  if (mMagneticField) {
    ExitMagField();
  }
}

void GPUDisplayBackendOpenGL::clearScreen(bool alphaOnly)
{
  if (mDisplay->cfgL().invertColors && !alphaOnly) {
    CHKERR(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
  } else {
    CHKERR(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
  }
  if (alphaOnly) {
    glColorMask(false, false, false, true);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(true, true, true, true);
  } else {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
  }
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
    fillIndirectCmdBuffer();
    CHKERR(glBufferData(GL_DRAW_INDIRECT_BUFFER, mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), GL_STATIC_DRAW));
    mCmdBuffer.clear();
  }
}

void GPUDisplayBackendOpenGL::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (mDisplay->updateRenderPipeline() || mDownsampleFactor != getDownsampleFactor(requestScreenshot)) {
    updateRenderer(requestScreenshot);
  }
  if (mOffscreenBufferMSAA.created) {
    setFrameBuffer(mOffscreenBufferMSAA.fb_id);
  } else if (toMixBuffer) {
    setFrameBuffer(mMixBuffer.fb_id);
  } else if (mOffscreenBuffer.created) {
    setFrameBuffer(mOffscreenBuffer.fb_id);
  } else {
    setFrameBuffer(0);
  }
  clearScreen();
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

  glViewport(0, 0, mRenderWidth, mRenderHeight);
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
    if (mSPIRVShaders) {
      glBindBuffer(GL_UNIFORM_BUFFER, mSPIRVModelViewBuffer);
      glBufferData(GL_UNIFORM_BUFFER, sizeof(modelViewProj), &modelViewProj, GL_STATIC_DRAW);
      glBindBufferBase(GL_UNIFORM_BUFFER, 0, mSPIRVModelViewBuffer);
    } else {
      CHKERR(glUniformMatrix4fv(mModelViewProjId, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
    }
    if (mMagneticField) {
      CHKERR(glNamedBufferSubData(mFieldModelViewBuffer, 0, sizeof(modelViewProj), &modelViewProj));
    }
  }
}

void GPUDisplayBackendOpenGL::finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    CHKERR(glDisableClientState(GL_VERTEX_ARRAY));
  } else
#endif
  {
    CHKERR(glBindVertexArray(0));
    CHKERR(glUseProgram(0));
  }

  if (mDisplay->cfgR().drawQualityMSAA) {
    unsigned int dstId = toMixBuffer ? mMixBuffer.fb_id : (mDownsampleFactor != 1 ? mOffscreenBuffer.fb_id : 0);
    CHKERR(glBlitNamedFramebuffer(mOffscreenBufferMSAA.fb_id, dstId, 0, 0, mRenderWidth, mRenderHeight, 0, 0, mRenderWidth, mRenderHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  }
  if (includeMixImage > 0) {
    setFrameBuffer(mDownsampleFactor != 1 ? mOffscreenBuffer.fb_id : 0);
    mixImages(includeMixImage);
  }
  if (mDownsampleFactor != 1 && !toMixBuffer) {
    CHKERR(glBlitNamedFramebuffer(mOffscreenBuffer.fb_id, 0, 0, 0, mRenderWidth, mRenderHeight, 0, 0, mScreenWidth, mScreenHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  }

  if (doScreenshot && !toMixBuffer) {
    readImageToPixels();
  }
}

void GPUDisplayBackendOpenGL::readImageToPixels()
{
  int scaleFactor = mDisplay->cfgR().screenshotScaleFactor;
  unsigned int width = mScreenWidth * scaleFactor;
  unsigned int height = mScreenHeight * scaleFactor;
  GLfb tmpBuffer;
  if (mDisplay->cfgR().drawQualityDownsampleFSAA && mDisplay->cfgR().screenshotScaleFactor != 1) {
    createFB(tmpBuffer, false, true, false, width, height);
    CHKERR(glBlitNamedFramebuffer(mOffscreenBuffer.fb_id, tmpBuffer.fb_id, 0, 0, mRenderWidth, mRenderHeight, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  }
  setFrameBuffer(tmpBuffer.created ? tmpBuffer.fb_id : (scaleFactor != 1 ? mOffscreenBuffer.fb_id : 0));
  CHKERR(glPixelStorei(GL_PACK_ALIGNMENT, 1));
  CHKERR(glReadBuffer(scaleFactor != 1 ? GL_COLOR_ATTACHMENT0 : GL_BACK));
  mScreenshotPixels.resize(width * height * 4);
  CHKERR(glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, mScreenshotPixels.data()));
  if (tmpBuffer.created) {
    deleteFB(tmpBuffer);
  }
}

void GPUDisplayBackendOpenGL::finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) {}

void GPUDisplayBackendOpenGL::prepareText()
{
  setFrameBuffer(0);
  glViewport(0, 0, mScreenWidth, mScreenHeight);
  hmm_mat4 proj = HMM_Orthographic(0.f, mScreenWidth, 0.f, mScreenHeight, -1, 1);
  if (mDisplay->drawTextInCompatMode()) {
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&proj.Elements[0][0]);
    glViewport(0, 0, mScreenWidth, mScreenHeight);
#endif
  } else if (mFreetypeInitialized) {
    CHKERR(glUseProgram(mShaderProgramText));
    CHKERR(glActiveTexture(GL_TEXTURE0));
    CHKERR(glBindVertexArray(VAO_text));
    CHKERR(glUniformMatrix4fv(mModelViewProjIdText, 1, GL_FALSE, &proj.Elements[0][0]));
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    CHKERR(glUniform4fv(mColorIdText, 1, &color[0]));
  }
}

void GPUDisplayBackendOpenGL::finishText()
{
  if (!mDisplay->drawTextInCompatMode() && mFreetypeInitialized) {
    CHKERR(glBindVertexArray(0));
    CHKERR(glBindTexture(GL_TEXTURE_2D, 0));
    CHKERR(glUseProgram(0));
  }
}

void GPUDisplayBackendOpenGL::mixImages(float mixSlaveImage)
{
  hmm_mat4 proj = HMM_Orthographic(0.f, mRenderWidth, 0.f, mRenderHeight, -1.f, 1.f);
  glDisable(GL_DEPTH_TEST);
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&proj.Elements[0][0]);
    CHKERR(glEnable(GL_TEXTURE_2D));
  } else
#endif
  {
    CHKERR(glUseProgram(mShaderProgramTexture));
    CHKERR(glActiveTexture(GL_TEXTURE0));
    CHKERR(glBindVertexArray(VAO_texture));
    CHKERR(glUniformMatrix4fv(mModelViewProjIdTexture, 1, GL_FALSE, &proj.Elements[0][0]));
    CHKERR(glUniform1f(mAlphaIdTexture, mixSlaveImage));
  }

  CHKERR(glBindTexture(GL_TEXTURE_2D, mMixBuffer.fbCol_id));

#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mDisplay->cfgR().openGLCore) {
    glColor4f(1, 1, 1, mixSlaveImage);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, 1);
    glVertex3f(0, mRenderHeight, 0);
    glTexCoord2f(1, 1);
    glVertex3f(mRenderWidth, mRenderHeight, 0);
    glTexCoord2f(1, 0);
    glVertex3f(mRenderWidth, 0, 0);
    glEnd();
    glColor4f(1, 1, 1, 0);
    CHKERR(glDisable(GL_TEXTURE_2D));
  } else
#endif
  {
    CHKERR(glDrawArrays(GL_TRIANGLES, 0, 6));
    CHKERR(glBindVertexArray(0));
    CHKERR(glUseProgram(0));
  }
  CHKERR(glBindTexture(GL_TEXTURE_2D, 0));
  setDepthBuffer();
}

void GPUDisplayBackendOpenGL::pointSizeFactor(float factor)
{
  CHKERR(glPointSize(mDisplay->cfgL().pointSize * mDownsampleFactor * factor));
}

void GPUDisplayBackendOpenGL::lineWidthFactor(float factor)
{
  CHKERR(glLineWidth(mDisplay->cfgL().lineWidth * mDownsampleFactor * factor));
}

void GPUDisplayBackendOpenGL::addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data)
{
  int oldAlign;
  CHKERR(glGetIntegerv(GL_UNPACK_ALIGNMENT, &oldAlign));
  CHKERR(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
  if (symbol != (int)mFontSymbols.size()) {
    throw std::runtime_error("Incorrect symbol ID");
  }
  unsigned int texId;
  glGenTextures(1, &texId);
  std::vector<unsigned char> tmp;
  if (!smoothFont()) {
    tmp.resize(sizex * sizey);
    unsigned char* src = (unsigned char*)data;
    for (int i = 0; i < sizex * sizey; i++) {
      tmp[i] = src[i] > 128 ? 255 : 0;
    }
    data = tmp.data();
  }
  mFontSymbols.emplace_back(FontSymbolOpenGL{{{sizex, sizey}, {offsetx, offsety}, advance}, texId});
  if (sizex == 0 || sizey == 0) {
    return;
  }
  glBindTexture(GL_TEXTURE_2D, texId);
  CHKERR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, sizex, sizey, 0, GL_RED, GL_UNSIGNED_BYTE, data));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  CHKERR(glPixelStorei(GL_UNPACK_ALIGNMENT, oldAlign));
}

void GPUDisplayBackendOpenGL::initializeTextDrawing()
{
  CHKERR(glGenVertexArrays(1, &VAO_text));
  CHKERR(glGenBuffers(1, &VBO_text));
  CHKERR(glBindVertexArray(VAO_text));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, VBO_text));
  CHKERR(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW));
  CHKERR(glEnableVertexAttribArray(0));
  CHKERR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, 0));
  CHKERR(glBindVertexArray(0));
}

void GPUDisplayBackendOpenGL::OpenGLPrint(const char* s, float x, float y, float* color, float scale)
{
  if (!mFreetypeInitialized || mDisplay->drawTextInCompatMode()) {
    return;
  }
  CHKERR(glUniform4fv(mColorIdText, 1, color));

  if (smoothFont()) {
    scale *= 0.25f; // Font size is 48 to have nice bitmap, scale to size 12
  }
  for (const char* c = s; *c; c++) {
    if ((int)*c > (int)mFontSymbols.size()) {
      GPUError("Trying to draw unsupported symbol: %d > %d\n", (int)*c, (int)mFontSymbols.size());
      continue;
    }
    FontSymbolOpenGL sym = mFontSymbols[*c];
    if (sym.size[0] && sym.size[1]) {
      float xpos = x + sym.offset[0] * scale;
      float ypos = y - (sym.size[1] - sym.offset[1]) * scale;
      float w = sym.size[0] * scale;
      float h = sym.size[1] * scale;
      float vertices[6][4] = {
        {xpos, ypos + h, 0.0f, 0.0f},
        {xpos, ypos, 0.0f, 1.0f},
        {xpos + w, ypos, 1.0f, 1.0f},
        {xpos, ypos + h, 0.0f, 0.0f},
        {xpos + w, ypos, 1.0f, 1.0f},
        {xpos + w, ypos + h, 1.0f, 0.0f}};
      CHKERR(glBindTexture(GL_TEXTURE_2D, sym.texId));
      CHKERR(glBindBuffer(GL_ARRAY_BUFFER, VBO_text));
      CHKERR(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices));
      CHKERR(glBindBuffer(GL_ARRAY_BUFFER, 0));
      CHKERR(glDrawArrays(GL_TRIANGLES, 0, 6));
    }
    x += (sym.advance >> 6) * scale; // shift is in 1/64th of a pixel
  }
  if (!mDisplay->useMultiVBO()) {
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0])); // If we don't use multiple buffers, we keep the default buffer bound
  }
}

void GPUDisplayBackendOpenGL::ClearOffscreenBuffers()
{
  if (mMixBuffer.created) {
    deleteFB(mMixBuffer);
  }
  if (mOffscreenBufferMSAA.created) {
    deleteFB(mOffscreenBufferMSAA);
  }
  if (mOffscreenBuffer.created) {
    deleteFB(mOffscreenBuffer);
  }
}

void GPUDisplayBackendOpenGL::resizeScene(unsigned int width, unsigned int height)
{
  mScreenWidth = width;
  mScreenHeight = height;

  updateRenderer(false);

  float vertices[6][4] = {
    {0, (float)mRenderHeight, 0.0f, 1.0f},
    {0, 0, 0.0f, 0.0f},
    {(float)mRenderWidth, 0, 1.0f, 0.0f},
    {0, (float)mRenderHeight, 0.0f, 1.0f},
    {(float)mRenderWidth, 0, 1.0f, 0.0f},
    {(float)mRenderWidth, (float)mRenderHeight, 1.0f, 1.0f}};
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, VBO_texture));
  CHKERR(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, 0));
  if (!mDisplay->useMultiVBO()) {
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0])); // If we don't use multiple buffers, we keep the default buffer bound
  }
}

void GPUDisplayBackendOpenGL::updateRenderer(bool withScreenshot)
{
  mDownsampleFactor = getDownsampleFactor(withScreenshot);
  mRenderWidth = mScreenWidth * mDownsampleFactor;
  mRenderHeight = mScreenHeight * mDownsampleFactor;
  ClearOffscreenBuffers();
  if (mDisplay->cfgR().drawQualityMSAA > 1) {
    createFB(mOffscreenBufferMSAA, false, true, true, mRenderWidth, mRenderHeight);
  }
  if (mDownsampleFactor != 1) {
    createFB(mOffscreenBuffer, false, true, false, mRenderWidth, mRenderHeight);
  }
  createFB(mMixBuffer, true, true, false, mRenderWidth, mRenderHeight);
  setQuality();
}

#else  // GPUCA_BUILD_EVENT_DISPLAY_OPENGL
GPUDisplayBackendOpenGL::GPUDisplayBackendOpenGL()
{
}
int GPUDisplayBackendOpenGL::checkShaderStatus(unsigned int shader) { return 0; }
int GPUDisplayBackendOpenGL::checkProgramStatus(unsigned int program) { return 0; }
int GPUDisplayBackendOpenGL::ExtInit() { throw std::runtime_error("Insufficnet OpenGL version"); }
bool GPUDisplayBackendOpenGL::CoreProfile() { return false; }
unsigned int GPUDisplayBackendOpenGL::DepthBits() { return 0; }
unsigned int GPUDisplayBackendOpenGL::drawVertices(const vboList& v, const drawType t) { return 0; }
void GPUDisplayBackendOpenGL::ActivateColor(std::array<float, 4>& color) {}
void GPUDisplayBackendOpenGL::setQuality() {}
void GPUDisplayBackendOpenGL::setDepthBuffer() {}
int GPUDisplayBackendOpenGL::InitBackendA() { throw std::runtime_error("Insufficnet OpenGL version"); }
void GPUDisplayBackendOpenGL::ExitBackendA() {}
void GPUDisplayBackendOpenGL::loadDataToGPU(size_t totalVertizes) {}
void GPUDisplayBackendOpenGL::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) {}
void GPUDisplayBackendOpenGL::resizeScene(unsigned int width, unsigned int height) {}
void GPUDisplayBackendOpenGL::finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage) {}
void GPUDisplayBackendOpenGL::finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) {}
void GPUDisplayBackendOpenGL::prepareText() {}
void GPUDisplayBackendOpenGL::finishText() {}
void GPUDisplayBackendOpenGL::pointSizeFactor(float factor) {}
void GPUDisplayBackendOpenGL::lineWidthFactor(float factor) {}
void GPUDisplayBackendOpenGL::addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) {}
void GPUDisplayBackendOpenGL::initializeTextDrawing() {}
void GPUDisplayBackendOpenGL::OpenGLPrint(const char* s, float x, float y, float* color, float scale) {}
#endif // GPUCA_BUILD_EVENT_DISPLAY_OPENGL
