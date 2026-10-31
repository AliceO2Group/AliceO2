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
#include "HandMadeMath.h"

#include "GPUCommonDef.h"
#include "GPUDisplayBackendOpenGL.h"
#include "GPUDisplay.h"
#include "GPULogging.h"
#include "GPUParam.h"
#include "GPUTPCGeometry.h"

#define OPENGL_EMULATE_MULTI_DRAW 0

using namespace o2::gpu;

#define GPUCA_DISPLAY_SHADER_TYPE_OPENGL
#include "GPUDisplayLoadShaders.h"

uint32_t GPUDisplayBackendOpenGL::loadShader(uint32_t type, const char* spirv, size_t size)
{
  uint32_t shader = glCreateShader(type);
  glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V, spirv, size);
  glSpecializeShader(shader, "main", 0, nullptr, nullptr);
  return shader;
}

GPUDisplayBackendOpenGL::GPUDisplayBackendOpenGL()
{
  mBackendType = TYPE_OPENGL;
  mBackendName = "OpenGL";
}

int32_t GPUDisplayBackendOpenGL::ExtInit()
{
#ifdef GPUCA_DISPLAY_GL3W
  return gl3wInit();
#else
  return glewInit();
#endif
}

bool GPUDisplayBackendOpenGL::canLimitToVSync() const
{
  return mDisplay->frontend()->canSetVSync();
}

// #define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                                                 \
  do {                                                                                                              \
    (cmd);                                                                                                          \
    GLenum err = glGetError();                                                                                      \
    while (err != GL_NO_ERROR) {                                                                                    \
      GPUError("OpenGL Error %d: %s (%s: %d)", (int32_t)err, (const char*)gluErrorString(err), __FILE__, __LINE__); \
      throw std::runtime_error("OpenGL Failure");                                                                   \
    }                                                                                                               \
  } while (false)

void GPUDisplayBackendOpenGL::createFB(GLfb& fb, bool tex, bool withDepth, bool msaa, uint32_t width, uint32_t height)
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
    GPUError("Error creating framebuffer (tex %d) - incomplete (%d)", (int32_t)tex, status);
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

uint32_t GPUDisplayBackendOpenGL::drawVertices(const vboList& v, const drawType tt)
{
  static constexpr GLenum types[3] = {GL_POINTS, GL_LINES, GL_LINE_STRIP};
  GLenum t = types[tt];
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSector = std::get<2>(v);
  if (count == 0) {
    return 0;
  }

  if (mDisplay->useMultiVBO()) {
    CHKERR(glBindVertexArray(mVertexArray));
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[iSector]));
    CHKERR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
    glEnableVertexAttribArray(0);
  }

  const uint32_t smoothProgram = tt == POINTS ? (mDisplay->cfgL().smoothPoints ? mShaderProgramSmoothPoint : 0) : (mDisplay->cfgL().smoothLines ? mShaderProgramSmoothLine : 0);
  if (smoothProgram) {
    CHKERR(glUseProgram(smoothProgram));
  }
  if (mDisplay->cfgR().useGLIndirectDraw) {
    CHKERR(glMultiDrawArraysIndirect(t, (void*)(size_t)((mIndirectSectorOffset[iSector] + first) * sizeof(DrawArraysIndirectCommand)), count, 0));
  } else if (OPENGL_EMULATE_MULTI_DRAW) {
    for (uint32_t k = 0; k < count; k++) {
      CHKERR(glDrawArrays(t, mDisplay->vertexBufferStart()[iSector][first + k], mDisplay->vertexBufferCount()[iSector][first + k]));
    }
  } else {
    static_assert(sizeof(GLsizei) == sizeof(*mDisplay->vertexBufferCount()[iSector].data()), "Invalid counter size does not match GLsizei");
    CHKERR(glMultiDrawArrays(t, mDisplay->vertexBufferStart()[iSector].data() + first, ((const GLsizei*)mDisplay->vertexBufferCount()[iSector].data()) + first, count));
  }
  if (smoothProgram) {
    CHKERR(glUseProgram(mShaderProgram));
  }
  return count;
}

void GPUDisplayBackendOpenGL::ActivateColor(std::array<float, 4>& color)
{
  CHKERR(glUniform4fv(UNIFORM_LOCATION_COLOR, 1, &color[0]));
  if (mDisplay->cfgL().smoothPoints) {
    CHKERR(glProgramUniform4fv(mShaderProgramSmoothPoint, UNIFORM_LOCATION_COLOR, 1, &color[0]));
  }
  if (mDisplay->cfgL().smoothLines) {
    CHKERR(glProgramUniform4fv(mShaderProgramSmoothLine, UNIFORM_LOCATION_COLOR, 1, &color[0]));
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

void GPUDisplayBackendOpenGL::setFrameBuffer(uint32_t newID)
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

int32_t GPUDisplayBackendOpenGL::checkShaderStatus(uint32_t shader)
{
  int32_t status, loglen;
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

int32_t GPUDisplayBackendOpenGL::checkProgramStatus(uint32_t program)
{
  int32_t status, loglen;
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

int32_t GPUDisplayBackendOpenGL::InitBackendA()
{
  if (mDisplay->GetProcessingSettings().debugLevel >= 2) {
    auto renderer = glGetString(GL_RENDERER);
    GPUInfo("Renderer: %s", (const char*)renderer);
  }

  int32_t glVersion[2] = {0, 0};
  glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
  glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
  if (glVersion[0] < GPUDisplayFrontend::GL_MIN_VERSION_MAJOR || (glVersion[0] == GPUDisplayFrontend::GL_MIN_VERSION_MAJOR && glVersion[1] < GPUDisplayFrontend::GL_MIN_VERSION_MINOR)) {
    GPUError("Unsupported OpenGL runtime %d.%d < %d.%d", glVersion[0], glVersion[1], GPUDisplayFrontend::GL_MIN_VERSION_MAJOR, GPUDisplayFrontend::GL_MIN_VERSION_MINOR);
    return (1);
  }
  for (GLenum limit : {GL_MAX_TEXTURE_SIZE, GL_MAX_RENDERBUFFER_SIZE, GL_MAX_FRAMEBUFFER_WIDTH, GL_MAX_FRAMEBUFFER_HEIGHT}) {
    GLint value = 0;
    glGetIntegerv(limit, &value);
    mMaxRenderSize = std::min<uint32_t>(mMaxRenderSize, value);
  }
  mVBOId.resize(GPUTPCGeometry::NSECTORS);
  CHKERR(glCreateBuffers(mVBOId.size(), mVBOId.data()));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0]));
  CHKERR(glGenBuffers(1, &mIndirectId));
  CHKERR(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, mIndirectId));
  setDepthBuffer();
  setQuality();
  CHKERR(mVertexShader = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertex_vert_spv_start, _binary_opengl_vertex_vert_spv_len));
  CHKERR(mFragmentShader = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragment_frag_spv_start, _binary_opengl_fragment_frag_spv_len));
  CHKERR(mVertexShaderTexture = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertexTexture_vert_spv_start, _binary_opengl_vertexTexture_vert_spv_len));
  CHKERR(mFragmentShaderTexture = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragmentTexture_frag_spv_start, _binary_opengl_fragmentTexture_frag_spv_len));
  CHKERR(mFragmentShaderText = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragmentText_frag_spv_start, _binary_opengl_fragmentText_frag_spv_len));
  CHKERR(mVertexShaderFullscreen = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertexFullscreen_vert_spv_start, _binary_opengl_vertexFullscreen_vert_spv_len));
  CHKERR(mFragmentShaderDownsample = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragmentDownsample_frag_spv_start, _binary_opengl_fragmentDownsample_frag_spv_len));
  CHKERR(mVertexShaderSmoothPoint = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertexSmoothPoint_vert_spv_start, _binary_opengl_vertexSmoothPoint_vert_spv_len));
  CHKERR(mFragmentShaderSmoothPoint = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragmentSmoothPoint_frag_spv_start, _binary_opengl_fragmentSmoothPoint_frag_spv_len));
  CHKERR(mVertexShaderSmoothLine = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertexSmoothLine_vert_spv_start, _binary_opengl_vertexSmoothLine_vert_spv_len));
  CHKERR(mFragmentShaderSmoothLine = loadShader(GL_FRAGMENT_SHADER, _binary_opengl_fragmentSmoothLine_frag_spv_start, _binary_opengl_fragmentSmoothLine_frag_spv_len));
  if (checkShaderStatus(mVertexShader) || checkShaderStatus(mFragmentShader) || checkShaderStatus(mVertexShaderSmoothPoint) || checkShaderStatus(mFragmentShaderSmoothPoint) || checkShaderStatus(mVertexShaderSmoothLine) || checkShaderStatus(mFragmentShaderSmoothLine) || checkShaderStatus(mVertexShaderTexture) || checkShaderStatus(mFragmentShaderTexture) || checkShaderStatus(mFragmentShaderText) || checkShaderStatus(mVertexShaderFullscreen) || checkShaderStatus(mFragmentShaderDownsample)) {
    return 1;
  }
  CHKERR(mShaderProgram = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgram, mVertexShader));
  CHKERR(glAttachShader(mShaderProgram, mFragmentShader));
  CHKERR(glLinkProgram(mShaderProgram));
  if (checkProgramStatus(mShaderProgram)) {
    return 1;
  }
  CHKERR(mShaderProgramSmoothPoint = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramSmoothPoint, mVertexShaderSmoothPoint));
  CHKERR(glAttachShader(mShaderProgramSmoothPoint, mFragmentShaderSmoothPoint));
  CHKERR(glLinkProgram(mShaderProgramSmoothPoint));
  if (checkProgramStatus(mShaderProgramSmoothPoint)) {
    return 1;
  }
  CHKERR(mShaderProgramSmoothLine = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramSmoothLine, mVertexShaderSmoothLine));
  CHKERR(glAttachShader(mShaderProgramSmoothLine, mFragmentShaderSmoothLine));
  CHKERR(glLinkProgram(mShaderProgramSmoothLine));
  if (checkProgramStatus(mShaderProgramSmoothLine)) {
    return 1;
  }
  CHKERR(mShaderProgramText = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramText, mVertexShaderTexture));
  CHKERR(glAttachShader(mShaderProgramText, mFragmentShaderText));
  CHKERR(glLinkProgram(mShaderProgramText));
  if (checkProgramStatus(mShaderProgramText)) {
    return 1;
  }
  CHKERR(mShaderProgramTexture = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramTexture, mVertexShaderTexture));
  CHKERR(glAttachShader(mShaderProgramTexture, mFragmentShaderTexture));
  CHKERR(glLinkProgram(mShaderProgramTexture));
  if (checkProgramStatus(mShaderProgramTexture)) {
    return 1;
  }
  CHKERR(mShaderProgramDownsample = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramDownsample, mVertexShaderFullscreen));
  CHKERR(glAttachShader(mShaderProgramDownsample, mFragmentShaderDownsample));
  CHKERR(glLinkProgram(mShaderProgramDownsample));
  if (checkProgramStatus(mShaderProgramDownsample)) {
    return 1;
  }
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
  CHKERR(glDeleteProgram(mShaderProgramSmoothPoint));
  CHKERR(glDeleteShader(mVertexShaderSmoothPoint));
  CHKERR(glDeleteShader(mFragmentShaderSmoothPoint));
  CHKERR(glDeleteProgram(mShaderProgramSmoothLine));
  CHKERR(glDeleteShader(mVertexShaderSmoothLine));
  CHKERR(glDeleteShader(mFragmentShaderSmoothLine));
  CHKERR(glDeleteProgram(mShaderProgramText));
  CHKERR(glDeleteShader(mFragmentShaderText));
  CHKERR(glDeleteProgram(mShaderProgramTexture));
  CHKERR(glDeleteProgram(mShaderProgramDownsample));
  CHKERR(glDeleteShader(mVertexShaderFullscreen));
  CHKERR(glDeleteShader(mFragmentShaderDownsample));
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
  if (mMagneticFieldVisualization) {
    ExitMagFieldVisualization();
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
  if (mDisplay->useMultiVBO()) {
    for (uint32_t i = 0; i < GPUTPCGeometry::NSECTORS; i++) {
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

bool GPUDisplayBackendOpenGL::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (mDisplay->updateRenderPipeline() || mDownsampleFactor != getDownsampleFactor(requestScreenshot)) {
    updateRenderer(requestScreenshot);
  }
  if (toMixBuffer && !mMixBuffer.created) { // Only allocated when mixing is used, freed with the other offscreen buffers
    createFB(mMixBuffer, true, true, false, mRenderWidth, mRenderHeight);
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
  CHKERR(glEnable(GL_BLEND));
  CHKERR(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

  glViewport(0, 0, mRenderWidth, mRenderHeight);
  CHKERR(glBindVertexArray(mVertexArray));
  CHKERR(glUseProgram(mShaderProgram));
  CHKERR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
  CHKERR(glEnableVertexAttribArray(0));

  const hmm_mat4 modelViewProj = proj * view;
  CHKERR(glUniformMatrix4fv(UNIFORM_LOCATION_MATRIX, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
  if (mDisplay->cfgL().smoothPoints) {
    CHKERR(glProgramUniformMatrix4fv(mShaderProgramSmoothPoint, UNIFORM_LOCATION_MATRIX, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
  }
  if (mDisplay->cfgL().smoothLines) {
    CHKERR(glProgramUniformMatrix4fv(mShaderProgramSmoothLine, UNIFORM_LOCATION_MATRIX, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
    CHKERR(glProgramUniform2f(mShaderProgramSmoothLine, UNIFORM_LOCATION_VIEWPORT, mRenderWidth, mRenderHeight));
  }
  if (mMagneticFieldVisualization) {
    CHKERR(glNamedBufferSubData(mFieldModelViewBuffer, 0, sizeof(modelViewProj), &modelViewProj));
  }
  return true;
}

void GPUDisplayBackendOpenGL::finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
  CHKERR(glBindVertexArray(0));
  CHKERR(glUseProgram(0));

  if (mOffscreenBufferMSAA.created) {
    uint32_t dstId = toMixBuffer ? mMixBuffer.fb_id : (mDownsampleFactor != 1 ? mOffscreenBuffer.fb_id : 0);
    CHKERR(glBlitNamedFramebuffer(mOffscreenBufferMSAA.fb_id, dstId, 0, 0, mRenderWidth, mRenderHeight, 0, 0, mRenderWidth, mRenderHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  }
  if (includeMixImage > 0) {
    setFrameBuffer(mDownsampleFactor != 1 ? mOffscreenBuffer.fb_id : 0);
    mixImages(includeMixImage);
  }
  if (mDownsampleFactor == 2 && !toMixBuffer) { // The linear blit is only exact for factor 2
    CHKERR(glBlitNamedFramebuffer(mOffscreenBuffer.fb_id, 0, 0, 0, mRenderWidth, mRenderHeight, 0, 0, mScreenWidth, mScreenHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  } else if (mDownsampleFactor != 1 && !toMixBuffer) {
    drawDownsample(mOffscreenBuffer.fbCol_id, 0, mScreenWidth, mScreenHeight, mDownsampleFactor);
  }

  if (doScreenshot && !toMixBuffer) {
    readImageToPixels();
  }
}

void GPUDisplayBackendOpenGL::readImageToPixels()
{
  int32_t scaleFactor = mDisplay->cfgR().screenshotScaleFactor;
  uint32_t width = mScreenWidth * scaleFactor;
  uint32_t height = mScreenHeight * scaleFactor;
  GLfb tmpBuffer;
  if (mDisplay->cfgR().drawQualityDownsampleFSAA && mDisplay->cfgR().screenshotScaleFactor != 1) {
    createFB(tmpBuffer, false, true, false, width, height);
    const int32_t factor = mRenderWidth / width;
    if (factor > 2) {
      drawDownsample(mOffscreenBuffer.fbCol_id, tmpBuffer.fb_id, width, height, factor);
    } else {
      CHKERR(glBlitNamedFramebuffer(mOffscreenBuffer.fb_id, tmpBuffer.fb_id, 0, 0, mRenderWidth, mRenderHeight, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR));
    }
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
  if (mFreetypeInitialized) {
    CHKERR(glUseProgram(mShaderProgramText));
    CHKERR(glActiveTexture(GL_TEXTURE0));
    CHKERR(glBindVertexArray(VAO_text));
    CHKERR(glUniformMatrix4fv(UNIFORM_LOCATION_MATRIX, 1, GL_FALSE, &proj.Elements[0][0]));
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    CHKERR(glUniform4fv(UNIFORM_LOCATION_COLOR, 1, &color[0]));
  }
}

void GPUDisplayBackendOpenGL::finishText()
{
  if (mFreetypeInitialized) {
    CHKERR(glBindVertexArray(0));
    CHKERR(glBindTexture(GL_TEXTURE_2D, 0));
    CHKERR(glUseProgram(0));
  }
}

void GPUDisplayBackendOpenGL::drawDownsample(uint32_t srcTexture, uint32_t dstFramebuffer, uint32_t width, uint32_t height, int32_t factor)
{
  setFrameBuffer(dstFramebuffer);
  CHKERR(glViewport(0, 0, width, height));
  CHKERR(glDisable(GL_DEPTH_TEST));
  CHKERR(glDisable(GL_BLEND));
  CHKERR(glUseProgram(mShaderProgramDownsample));
  CHKERR(glUniform1i(UNIFORM_LOCATION_COLOR, factor));
  CHKERR(glActiveTexture(GL_TEXTURE0));
  CHKERR(glBindTexture(GL_TEXTURE_2D, srcTexture));
  CHKERR(glBindVertexArray(VAO_texture)); // Any VAO, the full-screen triangle has no vertex input
  CHKERR(glDrawArrays(GL_TRIANGLES, 0, 3));
  CHKERR(glBindVertexArray(0));
  CHKERR(glBindTexture(GL_TEXTURE_2D, 0));
  CHKERR(glUseProgram(0));
  CHKERR(glEnable(GL_BLEND));
  setDepthBuffer();
}

void GPUDisplayBackendOpenGL::mixImages(float mixSlaveImage)
{
  hmm_mat4 proj = HMM_Orthographic(0.f, mRenderWidth, 0.f, mRenderHeight, -1.f, 1.f);
  glDisable(GL_DEPTH_TEST);
  CHKERR(glUseProgram(mShaderProgramTexture));
  CHKERR(glActiveTexture(GL_TEXTURE0));
  CHKERR(glBindVertexArray(VAO_texture));
  CHKERR(glUniformMatrix4fv(UNIFORM_LOCATION_MATRIX, 1, GL_FALSE, &proj.Elements[0][0]));
  CHKERR(glUniform1f(UNIFORM_LOCATION_COLOR, mixSlaveImage));

  CHKERR(glBindTexture(GL_TEXTURE_2D, mMixBuffer.fbCol_id));

  CHKERR(glDrawArrays(GL_TRIANGLES, 0, 6));
  CHKERR(glBindVertexArray(0));
  CHKERR(glUseProgram(0));
  CHKERR(glBindTexture(GL_TEXTURE_2D, 0));
  setDepthBuffer();
}

void GPUDisplayBackendOpenGL::pointSizeFactor(float factor)
{
  const float size = mDisplay->cfgL().pointSize * mDownsampleFactor * factor + (mDisplay->cfgL().smoothPoints ? SMOOTH_POINT_EXTRA_SIZE : 0.f);
  if (mDisplay->cfgL().smoothPoints) {
    CHKERR(glProgramUniform1f(mShaderProgramSmoothPoint, UNIFORM_LOCATION_POINT_SIZE, size));
  }
  CHKERR(glPointSize(size));
}

void GPUDisplayBackendOpenGL::lineWidthFactor(float factor)
{
  float width = mDisplay->cfgL().lineWidth * mDownsampleFactor * factor;
  if (mDisplay->cfgL().smoothLines) {
    width += SMOOTH_LINE_EXTRA_WIDTH;
    CHKERR(glProgramUniform1f(mShaderProgramSmoothLine, UNIFORM_LOCATION_LINE_WIDTH, width));
    width *= 1.4143f; // Wide lines are parallelograms in OpenGL, widen so that the perpendicular width is covered
  }
  CHKERR(glLineWidth(width));
}

void GPUDisplayBackendOpenGL::addFontSymbol(int32_t symbol, int32_t sizex, int32_t sizey, int32_t offsetx, int32_t offsety, int32_t advance, void* data)
{
  int32_t oldAlign;
  CHKERR(glGetIntegerv(GL_UNPACK_ALIGNMENT, &oldAlign));
  CHKERR(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
  if (symbol != (int32_t)mFontSymbols.size()) {
    throw std::runtime_error("Incorrect symbol ID");
  }
  uint32_t texId;
  glGenTextures(1, &texId);
  std::vector<uint8_t> tmp;
  if (!smoothFont()) {
    tmp.resize(sizex * sizey);
    uint8_t* src = (uint8_t*)data;
    for (int32_t i = 0; i < sizex * sizey; i++) {
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

void GPUDisplayBackendOpenGL::clearFontSymbols()
{
  CHKERR(glDeleteBuffers(1, &VBO_text));
  CHKERR(glDeleteVertexArrays(1, &VAO_text));
  for (auto& symbol : mFontSymbols) {
    CHKERR(glDeleteTextures(1, &symbol.texId));
  }
  mFontSymbols.clear();
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
  if (!mFreetypeInitialized) {
    return;
  }
  CHKERR(glUniform4fv(UNIFORM_LOCATION_COLOR, 1, color));

  if (smoothFont()) {
    scale *= 0.25f; // Font size is 48 to have nice bitmap, scale to size 12
  }
  for (const unsigned char* c = (const unsigned char*)s; *c; c++) {
    if ((int32_t)*c >= (int32_t)mFontSymbols.size()) {
      GPUError("Trying to draw unsupported symbol: %d > %d\n", (int32_t)*c, (int32_t)mFontSymbols.size());
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

void GPUDisplayBackendOpenGL::resizeScene(uint32_t width, uint32_t height)
{
  mScreenWidth = width;
  mScreenHeight = height;

  updateRenderer(false);
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
    createFB(mOffscreenBuffer, mDownsampleFactor != 2, true, false, mRenderWidth, mRenderHeight); // Texture for shader downsampling (factor != 2), screenshot downsampling uses the shader only if this factor is != 2 as well
  }
  setQuality();

  float vertices[6][4] = {// Mix quad at render size
                          {0, (float)mRenderHeight, 0.0f, 1.0f},
                          {0, 0, 0.0f, 0.0f},
                          {(float)mRenderWidth, 0, 1.0f, 0.0f},
                          {0, (float)mRenderHeight, 0.0f, 1.0f},
                          {(float)mRenderWidth, 0, 1.0f, 0.0f},
                          {(float)mRenderWidth, (float)mRenderHeight, 1.0f, 1.0f}};
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, VBO_texture));
  CHKERR(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mDisplay->useMultiVBO() ? 0 : mVBOId[0])); // If we don't use multiple buffers, we keep the default buffer bound (also called from prepareDraw before the vertex pointers are set)
}
