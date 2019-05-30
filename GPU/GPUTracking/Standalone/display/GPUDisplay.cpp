// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplay.cpp
/// \author David Rohr

#include "GPUDisplay.h"
#include "GPUTPCDef.h"

#include <GL/glu.h>
#include <vector>
#include <array>
#include <tuple>
#include <memory>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include "bitmapfile.h"
#include "../utils/linux_helpers.h"
#endif
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

#include "GPUTPCMCInfo.h"
#include "GPUChainTracking.h"
#include "GPUQA.h"
#include "GPUTPCSliceData.h"
#include "GPUChainTracking.h"
#include "GPUTPCTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTRDTracker.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCClusterData.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDGeometry.h"
#include "utils/qconfig.h"

using namespace GPUCA_NAMESPACE::gpu;

//#define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                           \
  do {                                                                                        \
    (cmd);                                                                                    \
    GLenum err = glGetError();                                                                \
    while (err != GL_NO_ERROR) {                                                              \
      printf("OpenGL Error %d: %s (%s: %d)\n", err, gluErrorString(err), __FILE__, __LINE__); \
      throw std::runtime_error("OpenGL Failure");                                             \
    }                                                                                         \
  } while (false)

#define OPENGL_EMULATE_MULTI_DRAW 0

#define GL_SCALE_FACTOR 100.f

#define SEPERATE_GLOBAL_TRACKS_LIMIT (mSeparateGlobalTracks ? tGLOBALTRACK : TRACK_TYPE_ID_LIMIT)

static const GPUDisplay::configDisplay& GPUDisplay_GetConfig(GPUChainTracking* rec)
{
#if !defined(GPUCA_STANDALONE)
  static GPUDisplay::configDisplay defaultConfig;
  if (rec->mConfigDisplay) {
    return *((const GPUDisplay::configDisplay*)rec->mConfigDisplay);
  } else {
    return defaultConfig;
  }

#else
  return configStandalone.configGL;
#endif
}

GPUDisplay::GPUDisplay(GPUDisplayBackend* backend, GPUChainTracking* rec, GPUQA* qa) : mBackend(backend), mChain(rec), mConfig(GPUDisplay_GetConfig(rec)), mQA(qa), mMerger(rec->GetTPCMerger()) { backend->mDisplay = this; }

const GPUParam& GPUDisplay::param() { return mChain->GetParam(); }
const GPUTPCTracker& GPUDisplay::sliceTracker(int iSlice) { return mChain->GetTPCSliceTrackers()[iSlice]; }
const GPUTRDTracker& GPUDisplay::trdTracker() { return *mChain->GetTRDTracker(); }
const GPUChainTracking::InOutPointers GPUDisplay::ioptrs() { return mChain->mIOPtrs; }

inline void GPUDisplay::drawVertices(const vboList& v, const GLenum t)
{
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSlice = std::get<2>(v);
  if (count == 0) {
    return;
  }
  mNDrawCalls += count;

  if (mUseMultiVBO) {
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[iSlice]));
    CHKERR(glVertexPointer(3, GL_FLOAT, 0, nullptr));
  }

  if (mUseGLIndirectDraw) {
    CHKERR(glMultiDrawArraysIndirect(t, (void*)(size_t)((mIndirectSliceOffset[iSlice] + first) * sizeof(DrawArraysIndirectCommand)), count, 0));
  } else if (OPENGL_EMULATE_MULTI_DRAW) {
    for (int k = 0; k < count; k++) {
      CHKERR(glDrawArrays(t, mVertexBufferStart[iSlice][first + k], mVertexBufferCount[iSlice][first + k]));
    }
  } else {
    CHKERR(glMultiDrawArrays(t, mVertexBufferStart[iSlice].data() + first, mVertexBufferCount[iSlice].data() + first, count));
  }
}
inline void GPUDisplay::insertVertexList(std::pair<vecpod<GLint>*, vecpod<GLsizei>*>& vBuf, size_t first, size_t last)
{
  if (first == last) {
    return;
  }
  vBuf.first->emplace_back(first);
  vBuf.second->emplace_back(last - first);
}
inline void GPUDisplay::insertVertexList(int iSlice, size_t first, size_t last)
{
  std::pair<vecpod<GLint>*, vecpod<GLsizei>*> vBuf(mVertexBufferStart + iSlice, mVertexBufferCount + iSlice);
  insertVertexList(vBuf, first, last);
}

void GPUDisplay::calcXYZ()
{
  mXYZ[0] = -(mCurrentMatrix[0] * mCurrentMatrix[12] + mCurrentMatrix[1] * mCurrentMatrix[13] + mCurrentMatrix[2] * mCurrentMatrix[14]);
  mXYZ[1] = -(mCurrentMatrix[4] * mCurrentMatrix[12] + mCurrentMatrix[5] * mCurrentMatrix[13] + mCurrentMatrix[6] * mCurrentMatrix[14]);
  mXYZ[2] = -(mCurrentMatrix[8] * mCurrentMatrix[12] + mCurrentMatrix[9] * mCurrentMatrix[13] + mCurrentMatrix[10] * mCurrentMatrix[14]);

  mAngle[0] = -asinf(mCurrentMatrix[6]); // Invert rotY*rotX*rotZ
  float A = cosf(mAngle[0]);
  if (fabsf(A) > 0.005) {
    mAngle[1] = atan2f(-mCurrentMatrix[2] / A, mCurrentMatrix[10] / A);
    mAngle[2] = atan2f(mCurrentMatrix[4] / A, mCurrentMatrix[5] / A);
  } else {
    mAngle[1] = 0;
    mAngle[2] = atan2f(-mCurrentMatrix[1], -mCurrentMatrix[0]);
  }

  mRPhiTheta[0] = sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[1] * mXYZ[1] + mXYZ[2] * mXYZ[2]);
  mRPhiTheta[1] = atan2f(mXYZ[0], mXYZ[2]);
  mRPhiTheta[2] = atan2f(mXYZ[1], sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[2] * mXYZ[2]));

  createQuaternionFromMatrix(mQuat, mCurrentMatrix);

  /*float mAngle[1] = -asinf(mCurrentMatrix[2]); //Calculate Y-axis angle - for rotX*rotY*rotZ
  float C = cosf( angle_y );
  if (fabsf(C) > 0.005) //Gimball lock?
  {
      mAngle[0]  = atan2f(-mCurrentMatrix[6] / C, mCurrentMatrix[10] / C);
      mAngle[2]  = atan2f(-mCurrentMatrix[1] / C, mCurrentMatrix[0] / C);
  }
  else
  {
      mAngle[0]  = 0; //set x-angle
      mAngle[2]  = atan2f(mCurrentMatrix[4], mCurrentMatrix[5]);
  }*/
}

void GPUDisplay::SetCollisionFirstCluster(unsigned int collision, int slice, int cluster)
{
  mNCollissions = collision + 1;
  mCollisionClusters.resize(mNCollissions);
  mCollisionClusters[collision][slice] = cluster;
}

void GPUDisplay::mAnimationCloseAngle(float& newangle, float lastAngle)
{
  const float delta = lastAngle > newangle ? (2 * M_PI) : (-2 * M_PI);
  while (fabsf(newangle + delta - lastAngle) < fabsf(newangle - lastAngle)) {
    newangle += delta;
  }
}
void GPUDisplay::mAnimateCloseQuaternion(float* v, float lastx, float lasty, float lastz, float lastw)
{
  float distPos2 = (lastx - v[0]) * (lastx - v[0]) + (lasty - v[1]) * (lasty - v[1]) + (lastz - v[2]) * (lastz - v[2]) + (lastw - v[3]) * (lastw - v[3]);
  float distNeg2 = (lastx + v[0]) * (lastx + v[0]) + (lasty + v[1]) * (lasty + v[1]) + (lastz + v[2]) * (lastz + v[2]) + (lastw + v[3]) * (lastw + v[3]);
  if (distPos2 > distNeg2) {
    for (int i = 0; i < 4; i++) {
      v[i] = -v[i];
    }
  }
}
void GPUDisplay::setAnimationPoint()
{
  if (mCfg.animationMode & 4) // Spherical
  {
    float rxy = sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[2] * mXYZ[2]);
    float anglePhi = atan2f(mXYZ[0], mXYZ[2]);
    float angleTheta = atan2f(mXYZ[1], rxy);
    if (mAnimateVectors[0].size()) {
      mAnimationCloseAngle(anglePhi, mAnimateVectors[2].back());
    }
    if (mAnimateVectors[0].size()) {
      mAnimationCloseAngle(angleTheta, mAnimateVectors[3].back());
    }
    mAnimateVectors[1].emplace_back(0);
    mAnimateVectors[2].emplace_back(anglePhi);
    mAnimateVectors[3].emplace_back(angleTheta);
  } else {
    for (int i = 0; i < 3; i++) {
      mAnimateVectors[i + 1].emplace_back(mXYZ[i]);
    }
    // Cartesian
  }
  float r = sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[1] * mXYZ[1] + mXYZ[2] * mXYZ[2]);
  mAnimateVectors[4].emplace_back(r);
  if (mCfg.animationMode & 1) // Euler-angles
  {
    for (int i = 0; i < 3; i++) {
      float newangle = mAngle[i];
      if (mAnimateVectors[0].size()) {
        mAnimationCloseAngle(newangle, mAnimateVectors[i + 5].back());
      }
      mAnimateVectors[i + 5].emplace_back(newangle);
    }
    mAnimateVectors[8].emplace_back(0);
  } else { // Quaternions
    float v[4];
    createQuaternionFromMatrix(v, mCurrentMatrix);
    if (mAnimateVectors[0].size()) {
      mAnimateCloseQuaternion(v, mAnimateVectors[5].back(), mAnimateVectors[6].back(), mAnimateVectors[7].back(), mAnimateVectors[8].back());
    }
    for (int i = 0; i < 4; i++) {
      mAnimateVectors[i + 5].emplace_back(v[i]);
    }
  }
  float delay = 0.f;
  if (mAnimateVectors[0].size()) {
    delay = mAnimateVectors[0].back() + ((int)(mAnimationDelay * 20)) / 20.f;
  }
  mAnimateVectors[0].emplace_back(delay);
  mAnimateConfig.emplace_back(mCfg);
}
void GPUDisplay::resetAnimation()
{
  for (int i = 0; i < 9; i++) {
    mAnimateVectors[i].clear();
  }
  mAnimateConfig.clear();
  mAnimate = 0;
}
void GPUDisplay::removeAnimationPoint()
{
  if (mAnimateVectors[0].size() == 0) {
    return;
  }
  for (int i = 0; i < 9; i++) {
    mAnimateVectors[i].pop_back();
  }
  mAnimateConfig.pop_back();
}
void GPUDisplay::startAnimation()
{
  for (int i = 0; i < 8; i++) {
    mAnimationSplines[i].create(mAnimateVectors[0], mAnimateVectors[i + 1]);
  }
  mAnimationTimer.ResetStart();
  mAnimationFrame = 0;
  mAnimate = 1;
  mAnimationLastBase = 0;
}

inline void GPUDisplay::SetColorClusters()
{
  if (mCfg.colorCollisions) {
    return;
  }
  if (mInvertColors) {
    glColor3f(0, 0.3, 0.7);
  } else {
    glColor3f(0, 0.7, 1.0);
  }
}
inline void GPUDisplay::SetColorTRD()
{
  if (mCfg.colorCollisions) {
    return;
  }
  if (mInvertColors) {
    glColor3f(0.7, 0.3, 0);
  } else {
    glColor3f(1.0, 0.7, 0);
  }
}
inline void GPUDisplay::SetColorInitLinks()
{
  if (mInvertColors) {
    glColor3f(0.42, 0.4, 0.1);
  } else {
    glColor3f(0.42, 0.4, 0.1);
  }
}
inline void GPUDisplay::SetColorLinks()
{
  if (mInvertColors) {
    glColor3f(0.6, 0.1, 0.1);
  } else {
    glColor3f(0.8, 0.2, 0.2);
  }
}
inline void GPUDisplay::SetColorSeeds()
{
  if (mInvertColors) {
    glColor3f(0.6, 0.0, 0.65);
  } else {
    glColor3f(0.8, 0.1, 0.85);
  }
}
inline void GPUDisplay::SetColorTracklets()
{
  if (mInvertColors) {
    glColor3f(0, 0, 0);
  } else {
    glColor3f(1, 1, 1);
  }
}
inline void GPUDisplay::SetColorTracks()
{
  if (mInvertColors) {
    glColor3f(0.6, 0, 0.1);
  } else {
    glColor3f(0.8, 1., 0.15);
  }
}
inline void GPUDisplay::SetColorGlobalTracks()
{
  if (mInvertColors) {
    glColor3f(0.8, 0.2, 0);
  } else {
    glColor3f(1.0, 0.4, 0);
  }
}
inline void GPUDisplay::SetColorFinal()
{
  if (mCfg.colorCollisions) {
    return;
  }
  if (mInvertColors) {
    glColor3f(0, 0.6, 0.1);
  } else {
    glColor3f(0, 0.7, 0.2);
  }
}
inline void GPUDisplay::SetColorGrid()
{
  if (mInvertColors) {
    glColor3f(0.5, 0.5, 0);
  } else {
    glColor3f(0.7, 0.7, 0.0);
  }
}
inline void GPUDisplay::SetColorMarked()
{
  if (mInvertColors) {
    glColor3f(0.8, 0, 0);
  } else {
    glColor3f(1.0, 0.0, 0.0);
  }
}
inline void GPUDisplay::SetCollisionColor(int col)
{
  int red = (col * 2) % 5;
  int blue = (2 + col * 3) % 7;
  int green = (4 + col * 5) % 6;
  if (mInvertColors && red == 4 && blue == 5 && green == 6) {
    red = 0;
  }
  if (!mInvertColors && red == 0 && blue == 0 && green == 0) {
    red = 4;
  }
  glColor3f(red / 4., green / 5., blue / 6.);
}

void GPUDisplay::setQuality()
{
  // Doesn't seem to make a difference in this applicattion
  if (mDrawQualityMSAA > 1) {
    CHKERR(glEnable(GL_MULTISAMPLE));
  } else {
    CHKERR(glDisable(GL_MULTISAMPLE));
  }
}

void GPUDisplay::setDepthBuffer()
{
  if (mCfg.depthBuffer) {
    CHKERR(glEnable(GL_DEPTH_TEST)); // Enables Depth Testing
    CHKERR(glDepthFunc(GL_LEQUAL));  // The Type Of Depth Testing To Do
  } else {
    CHKERR(glDisable(GL_DEPTH_TEST));
  }
}

void GPUDisplay::createFB_texture(GLuint& id, bool msaa, GLenum storage, GLenum attachment)
{
  GLenum textureType = msaa ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
  CHKERR(glGenTextures(1, &id));
  CHKERR(glBindTexture(textureType, id));
  if (msaa) {
    CHKERR(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, mDrawQualityMSAA, storage, mRenderwidth, mRenderheight, false));
  } else {
    CHKERR(glTexImage2D(GL_TEXTURE_2D, 0, storage, mRenderwidth, mRenderheight, 0, storage, GL_UNSIGNED_BYTE, nullptr));
    CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    CHKERR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  }
  CHKERR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textureType, id, 0));
}

void GPUDisplay::createFB_renderbuffer(GLuint& id, bool msaa, GLenum storage, GLenum attachment)
{
  CHKERR(glGenRenderbuffers(1, &id));
  CHKERR(glBindRenderbuffer(GL_RENDERBUFFER, id));
  if (msaa) {
    CHKERR(glRenderbufferStorageMultisample(GL_RENDERBUFFER, mDrawQualityMSAA, storage, mRenderwidth, mRenderheight));
  } else {
    CHKERR(glRenderbufferStorage(GL_RENDERBUFFER, storage, mRenderwidth, mRenderheight));
  }
  CHKERR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, id));
}

void GPUDisplay::createFB(GLfb& fb, bool tex, bool withDepth, bool msaa)
{
  fb.tex = tex;
  fb.depth = withDepth;
  fb.msaa = msaa;
  GLint drawFboId = 0, readFboId = 0;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFboId);
  CHKERR(glGenFramebuffers(1, &fb.fb_id));
  CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, fb.fb_id));

  if (tex) {
    createFB_texture(fb.fbCol_id, fb.msaa, GL_RGBA, GL_COLOR_ATTACHMENT0);
  } else {
    createFB_renderbuffer(fb.fbCol_id, fb.msaa, GL_RGBA, GL_COLOR_ATTACHMENT0);
  }

  if (withDepth) {
    if (tex && fb.msaa) {
      createFB_texture(fb.fbDepth_id, fb.msaa, GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
    } else {
      createFB_renderbuffer(fb.fbDepth_id, fb.msaa, GL_DEPTH_COMPONENT24, GL_DEPTH_ATTACHMENT);
    }
  }

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    printf("Error creating framebuffer (tex %d) - incomplete (%d)\n", (int)tex, status);
    exit(1);
  }
  CHKERR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFboId));
  CHKERR(glBindFramebuffer(GL_READ_FRAMEBUFFER, readFboId));
  fb.created = true;
}

void GPUDisplay::deleteFB(GLfb& fb)
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

void GPUDisplay::setFrameBuffer(int updateCurrent, GLuint newID)
{
  if (updateCurrent == 1) {
    mMainBufferStack.push_back(newID);
  } else if (updateCurrent == 2) {
    mMainBufferStack.back() = newID;
  } else if (updateCurrent == -2) {
    newID = mMainBufferStack.back();
  } else if (updateCurrent == -1) {
    mMainBufferStack.pop_back();
    newID = mMainBufferStack.back();
  }
  if (newID == 0) {
    CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glDrawBuffer(GL_BACK);
  } else {
    CHKERR(glBindFramebuffer(GL_FRAMEBUFFER, newID));
    GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
  }
}

void GPUDisplay::UpdateOffscreenBuffers(bool clean)
{
  if (mMixBuffer.created) {
    deleteFB(mMixBuffer);
  }
  if (mOffscreenBuffer.created) {
    deleteFB(mOffscreenBuffer);
  }
  if (mOffscreenBufferNoMSAA.created) {
    deleteFB(mOffscreenBufferNoMSAA);
  }
  if (clean) {
    return;
  }

  if (mDrawQualityDownsampleFSAA > 1) {
    mRenderwidth = mScreenwidth * mDrawQualityDownsampleFSAA;
    mRenderheight = mScreenheight * mDrawQualityDownsampleFSAA;
  } else {
    mRenderwidth = mScreenwidth;
    mRenderheight = mScreenheight;
  }
  if (mDrawQualityMSAA > 1 || mDrawQualityDownsampleFSAA > 1) {
    createFB(mOffscreenBuffer, false, true, mDrawQualityMSAA > 1);
    if (mDrawQualityMSAA > 1 && mDrawQualityDownsampleFSAA > 1) {
      createFB(mOffscreenBufferNoMSAA, false, true, false);
    }
  }
  createFB(mMixBuffer, true, true, false);
  glViewport(0, 0, mRenderwidth, mRenderheight); // Reset The Current Viewport
  setQuality();
}

void GPUDisplay::ReSizeGLScene(int width, int height, bool init) // Resize And Initialize The GL Window
{
  if (height == 0) { // Prevent A Divide By Zero By
    height = 1;      // Making Height Equal One
  }
  mScreenwidth = width;
  mScreenheight = height;
  UpdateOffscreenBuffers();

  glMatrixMode(GL_PROJECTION); // Select The Projection Matrix
  glLoadIdentity();
  gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 1000.0f);

  glMatrixMode(GL_MODELVIEW); // Select The Modelview Matrix
  if (init) {
    mResetScene = 1;
    glLoadIdentity();
  } else {
    glLoadMatrixf(mCurrentMatrix);
  }

  glGetFloatv(GL_MODELVIEW_MATRIX, mCurrentMatrix);
}

void GPUDisplay::updateConfig()
{
  setQuality();
  setDepthBuffer();
}

int GPUDisplay::InitGL(bool initFailure)
{
  int retVal = initFailure;
  try {
    if (!initFailure) {
      retVal = InitGL_internal();
    }
  } catch (const std::runtime_error& e) {
    retVal = 1;
  }
  mInitResult = retVal == 0 ? 1 : -1;
  return (retVal);
}

int GPUDisplay::InitGL_internal()
{
  int glVersion[2] = { 0, 0 };
  glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
  glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
  if (glVersion[0] < 4 || (glVersion[0] == 4 && glVersion[1] < 5)) {
    printf("Unsupported OpenGL runtime %d.%d < 4.5\n", glVersion[0], glVersion[1]);
    return (1);
  }

  CHKERR(glCreateBuffers(GPUChainTracking::NSLICES, mVBOId));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0]));
  CHKERR(glGenBuffers(1, &mIndirectId));
  CHKERR(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, mIndirectId));

  CHKERR(glShadeModel(GL_SMOOTH)); // Enable Smooth Shading
  setDepthBuffer();
  setQuality();
  ReSizeGLScene(GPUDisplayBackend::INIT_WIDTH, GPUDisplayBackend::INIT_HEIGHT, true);
#ifdef GPUCA_HAVE_OPENMP
  int maxThreads = mChain->GetDeviceProcessingSettings().nThreads > 1 ? mChain->GetDeviceProcessingSettings().nThreads : 1;
  omp_set_num_threads(maxThreads);
#else
  int maxThreads = 1;
#endif
  mThreadBuffers.resize(maxThreads);
  mThreadTracks.resize(maxThreads);
  return (0); // Initialization Went OK
}

void GPUDisplay::ExitGL()
{
  UpdateOffscreenBuffers(true);
  CHKERR(glDeleteBuffers(GPUChainTracking::NSLICES, mVBOId));
  CHKERR(glDeleteBuffers(1, &mIndirectId));
}

inline void GPUDisplay::drawPointLinestrip(int iSlice, int cid, int id, int id_limit)
{
  mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y, mProjectXY ? 0 : mGlobalPos[cid].z);
  if (mGlobalPos[cid].w < id_limit) {
    mGlobalPos[cid].w = id;
  }
}

GPUDisplay::vboList GPUDisplay::DrawSpacePointsTRD(int iSlice, int select, int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();

  if (iCol == 0) {
    for (int i = 0; i < trdTracker().NTracklets(); i++) {
      int iSec = mChain->GetTRDGeometry()->GetSector(trdTracker().Tracklets()[i].GetDetector());
      bool draw = iSlice == iSec && mGlobalPosTRD[i].w == select;
      if (draw) {
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[i].x, mGlobalPosTRD[i].y, mProjectXY ? 0 : mGlobalPosTRD[i].z);
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[i].x, mGlobalPosTRD2[i].y, mProjectXY ? 0 : mGlobalPosTRD2[i].z);
      }
    }
  }

  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawClusters(const GPUTPCTracker& tracker, int select, int iCol)
{
  int iSlice = tracker.ISlice();
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();
  const int firstCluster = (mNCollissions > 1 && iCol > 0) ? mCollisionClusters[iCol - 1][iSlice] : 0;
  const int lastCluster = (mNCollissions > 1 && iCol + 1 < mNCollissions) ? mCollisionClusters[iCol][iSlice] : tracker.Data().NumberOfHits();
  for (int cidInSlice = firstCluster; cidInSlice < lastCluster; cidInSlice++) {
    const int cid = tracker.ClusterData()[cidInSlice].id;
    if (mHideUnmatchedClusters && mQA && mQA->SuppressHit(cid)) {
      continue;
    }
    bool draw = mGlobalPos[cid].w == select;

    if (markAdjacentClusters) {
      const int attach = mMerger.ClusterAttachment()[cid];
      if (attach) {
        if (markAdjacentClusters >= 16) {
          if (mQA && mQA->clusterRemovable(cid, markAdjacentClusters == 17)) {
            draw = select == tMARKED;
          }
        } else if ((markAdjacentClusters & 2) && (attach & GPUTPCGMMerger::attachTube)) {
          draw = select == tMARKED;
        } else if ((markAdjacentClusters & 1) && (attach & (GPUTPCGMMerger::attachGood | GPUTPCGMMerger::attachTube)) == 0) {
          draw = select == tMARKED;
        } else if ((markAdjacentClusters & 4) && (attach & GPUTPCGMMerger::attachGoodLeg) == 0) {
          draw = select == tMARKED;
        } else if (markAdjacentClusters & 8) {
          if (fabsf(mMerger.OutputTracks()[attach & GPUTPCGMMerger::attachTrackMask].GetParam().GetQPt()) > 20.f) {
            draw = select == tMARKED;
          }
        }
      }
    } else if (mMarkClusters) {
      const short flags = tracker.ClusterData()[cidInSlice].flags;
      const bool match = flags & mMarkClusters;
      draw = (select == tMARKED) ? (match) : (draw && !match);
    }
    if (draw) {
      mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y, mProjectXY ? 0 : mGlobalPos[cid].z);
    }
  }
  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawLinks(const GPUTPCTracker& tracker, int id, bool dodown)
{
  int iSlice = tracker.ISlice();
  if (mConfig.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    const GPUTPCRow& row = tracker.Data().Row(i);

    if (i < GPUCA_ROW_COUNT - 2) {
      const GPUTPCRow& rowUp = tracker.Data().Row(i + 2);
      for (int j = 0; j < row.NHits(); j++) {
        if (tracker.Data().HitLinkUpData(row, j) != CALINK_INVAL) {
          const int cid1 = tracker.ClusterData()[tracker.Data().ClusterDataIndex(row, j)].id;
          const int cid2 = tracker.ClusterData()[tracker.Data().ClusterDataIndex(rowUp, tracker.Data().HitLinkUpData(row, j))].id;
          drawPointLinestrip(iSlice, cid1, id);
          drawPointLinestrip(iSlice, cid2, id);
        }
      }
    }

    if (dodown && i >= 2) {
      const GPUTPCRow& rowDown = tracker.Data().Row(i - 2);
      for (int j = 0; j < row.NHits(); j++) {
        if (tracker.Data().HitLinkDownData(row, j) != CALINK_INVAL) {
          const int cid1 = tracker.ClusterData()[tracker.Data().ClusterDataIndex(row, j)].id;
          const int cid2 = tracker.ClusterData()[tracker.Data().ClusterDataIndex(rowDown, tracker.Data().HitLinkDownData(row, j))].id;
          drawPointLinestrip(iSlice, cid1, id);
          drawPointLinestrip(iSlice, cid2, id);
        }
      }
    }
  }
  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawSeeds(const GPUTPCTracker& tracker)
{
  int iSlice = tracker.ISlice();
  if (mConfig.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = 0; i < *tracker.NTracklets(); i++) {
    const GPUTPCHitId& hit = tracker.TrackletStartHit(i);
    size_t startCountInner = mVertexBuffer[iSlice].size();
    int ir = hit.RowIndex();
    calink ih = hit.HitIndex();
    do {
      const GPUTPCRow& row = tracker.Data().Row(ir);
      const int cid = tracker.ClusterData()[tracker.Data().ClusterDataIndex(row, ih)].id;
      drawPointLinestrip(iSlice, cid, tSEED);
      ir += 2;
      ih = tracker.Data().HitLinkUpData(row, ih);
    } while (ih != CALINK_INVAL);
    insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  }
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawTracklets(const GPUTPCTracker& tracker)
{
  int iSlice = tracker.ISlice();
  if (mConfig.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = 0; i < *tracker.NTracklets(); i++) {
    const GPUTPCTracklet& tracklet = tracker.Tracklet(i);
    if (tracklet.NHits() == 0) {
      continue;
    }
    size_t startCountInner = mVertexBuffer[iSlice].size();
    float4 oldpos;
    for (int j = tracklet.FirstRow(); j <= tracklet.LastRow(); j++) {
#ifdef GPUCA_EXTERN_ROW_HITS
      const calink rowHit = tracker.TrackletRowHits()[j * *tracker.NTracklets() + i];
#else
      const calink rowHit = tracklet.RowHit(j);
#endif
      if (rowHit != CALINK_INVAL) {
        const GPUTPCRow& row = tracker.Data().Row(j);
        const int cid = tracker.ClusterData()[tracker.Data().ClusterDataIndex(row, rowHit)].id;
        oldpos = mGlobalPos[cid];
        drawPointLinestrip(iSlice, cid, tTRACKLET);
      }
    }
    insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  }
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawTracks(const GPUTPCTracker& tracker, int global)
{
  int iSlice = tracker.ISlice();
  if (mConfig.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = (global ? tracker.CommonMemory()->nLocalTracks : 0); i < (global ? *tracker.NTracks() : tracker.CommonMemory()->nLocalTracks); i++) {
    GPUTPCTrack& track = tracker.Tracks()[i];
    size_t startCountInner = mVertexBuffer[iSlice].size();
    for (int j = 0; j < track.NHits(); j++) {
      const GPUTPCHitId& hit = tracker.TrackHits()[track.FirstHitID() + j];
      const GPUTPCRow& row = tracker.Data().Row(hit.RowIndex());
      const int cid = tracker.ClusterData()[tracker.Data().ClusterDataIndex(row, hit.HitIndex())].id;
      drawPointLinestrip(iSlice, cid, tSLICETRACK + global);
    }
    insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  }
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

void GPUDisplay::DrawFinal(int iSlice, int /*iCol*/, GPUTPCGMPropagator* prop, std::array<vecpod<int>, 2>& trackList, threadVertexBuffer& threadBuffer)
{
  auto& vBuf = threadBuffer.vBuf;
  auto& buffer = threadBuffer.buffer;
  unsigned int nTracks = std::max(trackList[0].size(), trackList[1].size());
  if (mConfig.clustersOnly) {
    nTracks = 0;
  }
  for (unsigned int ii = 0; ii < nTracks; ii++) {
    int i = 0;
    const GPUTPCGMMergedTrack* track = nullptr;
    int lastCluster = -1;
    while (true) {
      if (ii >= trackList[0].size()) {
        break;
      }
      i = trackList[0][ii];
      track = &mMerger.OutputTracks()[i];

      size_t startCountInner = mVertexBuffer[iSlice].size();
      bool drawing = false;
      if (mTRDTrackIds[i]) {
        auto& trk = trdTracker().Tracks()[mTRDTrackIds[i]];
        for (int k = 5; k >= 0; k--) {
          int cid = trk.GetTrackletIndex(k);
          if (cid < 0) {
            continue;
          }
          drawing = true;
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[cid].x, mGlobalPosTRD2[cid].y, mProjectXY ? 0 : mGlobalPosTRD2[cid].z);
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[cid].x, mGlobalPosTRD[cid].y, mProjectXY ? 0 : mGlobalPosTRD[cid].z);
          mGlobalPosTRD[cid].w = tTRDATTACHED;
        }
      }
      for (unsigned int k = 0; k < track->NClusters(); k++) {
        if (mHideRejectedClusters && (mMerger.Clusters()[track->FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject)) {
          continue;
        }
        int cid = mMerger.Clusters()[track->FirstClusterRef() + k].num;
        int w = mGlobalPos[cid].w;
        if (drawing) {
          drawPointLinestrip(iSlice, cid, tFINALTRACK, SEPERATE_GLOBAL_TRACKS_LIMIT);
        }
        if (w == SEPERATE_GLOBAL_TRACKS_LIMIT) {
          if (drawing) {
            insertVertexList(vBuf[0], startCountInner, mVertexBuffer[iSlice].size());
          }
          drawing = false;
        } else {
          if (!drawing) {
            startCountInner = mVertexBuffer[iSlice].size();
          }
          if (!drawing) {
            drawPointLinestrip(iSlice, cid, tFINALTRACK, SEPERATE_GLOBAL_TRACKS_LIMIT);
          }
          if (!drawing && lastCluster != -1) {
            drawPointLinestrip(iSlice, mMerger.Clusters()[track->FirstClusterRef() + lastCluster].num, 7, SEPERATE_GLOBAL_TRACKS_LIMIT);
          }
          drawing = true;
        }
        lastCluster = k;
      }
      insertVertexList(vBuf[0], startCountInner, mVertexBuffer[iSlice].size());
      break;
    }

    for (int iMC = 0; iMC < 2; iMC++) {
      if (iMC) {
        if (ii >= trackList[1].size()) {
          continue;
        }
        i = trackList[1][ii];
      } else {
        if (track == nullptr) {
          continue;
        }
        if (lastCluster == -1) {
          continue;
        }
      }

      size_t startCountInner = mVertexBuffer[iSlice].size();
      for (int inFlyDirection = 0; inFlyDirection < 2; inFlyDirection++) {
        GPUTPCGMPhysicalTrackModel trkParam;
        float ZOffset = 0;
        float x = 0;
        int slice = iSlice;
        float alpha = param().Alpha(slice);
        if (iMC == 0) {
          trkParam.Set(track->GetParam());
          ZOffset = track->GetParam().GetZOffset();
          auto cl = mMerger.Clusters()[track->FirstClusterRef() + lastCluster];
          x = cl.x;
        } else {
          const GPUTPCMCInfo& mc = ioptrs().mcInfosTPC[i];
          if (mc.charge == 0.f) {
            break;
          }
          if (mc.pid < 0) {
            break;
          }

          float c = cosf(alpha);
          float s = sinf(alpha);
          float mclocal[4];
          x = mc.x;
          float y = mc.y;
          mclocal[0] = x * c + y * s;
          mclocal[1] = -x * s + y * c;
          float px = mc.pX;
          float py = mc.pY;
          mclocal[2] = px * c + py * s;
          mclocal[3] = -px * s + py * c;
          float charge = mc.charge > 0 ? 1.f : -1.f;

          x = mclocal[0];
          if (fabsf(mc.z) > 250) {
            ZOffset = mc.z > 0 ? (mc.z - 250) : (mc.z + 250);
          }
          trkParam.Set(mclocal[0], mclocal[1], mc.z - ZOffset, mclocal[2], mclocal[3], mc.pZ, charge);
        }
        trkParam.X() += mXadd;
        x += mXadd;
        float z0 = trkParam.Z();
        if (iMC && inFlyDirection == 0) {
          buffer.clear();
        }
        if (x < 1) {
          break;
        }
        if (fabsf(trkParam.SinPhi()) > 1) {
          break;
        }
        alpha = param().Alpha(slice);
        vecpod<GLvertex>& useBuffer = iMC && inFlyDirection == 0 ? buffer : mVertexBuffer[iSlice];
        int nPoints = 0;

        while (nPoints++ < 5000) {
          if ((inFlyDirection == 0 && x < 0) || (inFlyDirection && x * x + trkParam.Y() * trkParam.Y() > (iMC ? (450 * 450) : (300 * 300)))) {
            break;
          }
          if (fabsf(trkParam.Z() + ZOffset) > mMaxClusterZ + (iMC ? 0 : 0)) {
            break;
          }
          if (fabsf(trkParam.Z() - z0) > (iMC ? 250 : 250)) {
            break;
          }
          if (inFlyDirection) {
            if (fabsf(trkParam.SinPhi()) > 0.4) {
              float dalpha = asinf(trkParam.SinPhi());
              trkParam.Rotate(dalpha);
              alpha += dalpha;
            }
            x = trkParam.X() + 1.f;
            if (!mPropagateLoopers) {
              float diff = fabsf(alpha - param().Alpha(slice)) / (2. * M_PI);
              diff -= floor(diff);
              if (diff > 0.25 && diff < 0.75) {
                break;
              }
            }
          }
          float B[3];
          prop->GetBxByBz(alpha, trkParam.GetX(), trkParam.GetY(), trkParam.GetZ(), B);
          float dLp = 0;
          if (trkParam.PropagateToXBxByBz(x, B[0], B[1], B[2], dLp)) {
            break;
          }
          if (fabsf(trkParam.SinPhi()) > 0.9) {
            break;
          }
          float sa = sinf(alpha), ca = cosf(alpha);
          useBuffer.emplace_back((ca * trkParam.X() - sa * trkParam.Y()) / GL_SCALE_FACTOR, (ca * trkParam.Y() + sa * trkParam.X()) / GL_SCALE_FACTOR, mProjectXY ? 0 : (trkParam.Z() + ZOffset) / GL_SCALE_FACTOR);
          x += inFlyDirection ? 1 : -1;
        }

        if (inFlyDirection == 0) {
          if (iMC) {
            for (int k = (int)buffer.size() - 1; k >= 0; k--) {
              mVertexBuffer[iSlice].emplace_back(buffer[k]);
            }
          } else {
            insertVertexList(vBuf[1], startCountInner, mVertexBuffer[iSlice].size());
            startCountInner = mVertexBuffer[iSlice].size();
          }
        }
      }
      insertVertexList(vBuf[iMC ? 3 : 2], startCountInner, mVertexBuffer[iSlice].size());
    }
  }
}

GPUDisplay::vboList GPUDisplay::DrawGrid(const GPUTPCTracker& tracker)
{
  int iSlice = tracker.ISlice();
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    const GPUTPCRow& row = tracker.Data().Row(i);
    for (int j = 0; j <= (signed)row.Grid().Ny(); j++) {
      float z1 = row.Grid().ZMin();
      float z2 = row.Grid().ZMax();
      float x = row.X() + mXadd;
      float y = row.Grid().YMin() + (float)j / row.Grid().StepYInv();
      float zz1, zz2, yy1, yy2, xx1, xx2;
      tracker.Param().Slice2Global(tracker.ISlice(), x, y, z1, &xx1, &yy1, &zz1);
      tracker.Param().Slice2Global(tracker.ISlice(), x, y, z2, &xx2, &yy2, &zz2);
      if (iSlice < 18) {
        zz1 += mZadd;
        zz2 += mZadd;
      } else {
        zz1 -= mZadd;
        zz2 -= mZadd;
      }
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
    }
    for (int j = 0; j <= (signed)row.Grid().Nz(); j++) {
      float y1 = row.Grid().YMin();
      float y2 = row.Grid().YMax();
      float x = row.X() + mXadd;
      float z = row.Grid().ZMin() + (float)j / row.Grid().StepZInv();
      float zz1, zz2, yy1, yy2, xx1, xx2;
      tracker.Param().Slice2Global(tracker.ISlice(), x, y1, z, &xx1, &yy1, &zz1);
      tracker.Param().Slice2Global(tracker.ISlice(), x, y2, z, &xx2, &yy2, &zz2);
      if (iSlice < 18) {
        zz1 += mZadd;
        zz2 += mZadd;
      } else {
        zz1 -= mZadd;
        zz2 -= mZadd;
      }
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
    }
  }
  insertVertexList(tracker.ISlice(), startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

int GPUDisplay::DrawGLScene(bool mixAnimation, float mAnimateTime)
{
  try {
    if (DrawGLScene_internal(mixAnimation, mAnimateTime)) {
      return (1);
    }
  } catch (const std::runtime_error& e) {
    return (1);
  }
  return (0);
}

int GPUDisplay::DrawGLScene_internal(bool mixAnimation, float mAnimateTime) // Here's Where We Do All The Drawing
{
  bool showTimer = false;

  // Make sure event gets not overwritten during display
  if (mAnimateTime < 0) {
    mSemLockDisplay.Lock();
  }

  if (!mixAnimation && mOffscreenBuffer.created) {
    setFrameBuffer(1, mOffscreenBuffer.fb_id);
  }

  // Initialize
  if (!mixAnimation) {
    if (mInvertColors) {
      CHKERR(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
    } else {
      CHKERR(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity(); // Reset The Current Modelview Matrix
  }

  int mMouseWheelTmp = mBackend->mMouseWheel;
  mBackend->mMouseWheel = 0;
  bool lookOrigin = mCamLookOrigin ^ mBackend->mKeys[mBackend->KEY_ALT];
  bool yUp = mCamYUp ^ mBackend->mKeys[mBackend->KEY_CTRL] ^ lookOrigin;

  // Calculate rotation / translation scaling factors
  float scalefactor = mBackend->mKeys[mBackend->KEY_SHIFT] ? 0.2 : 1.0;
  float rotatescalefactor = scalefactor * 0.25f;
  if (mCfg.drawSlice != -1) {
    scalefactor *= 0.2f;
  }
  float sqrdist = sqrtf(sqrtf(mCurrentMatrix[12] * mCurrentMatrix[12] + mCurrentMatrix[13] * mCurrentMatrix[13] + mCurrentMatrix[14] * mCurrentMatrix[14]) / GL_SCALE_FACTOR) * 0.8;
  if (sqrdist < 0.2) {
    sqrdist = 0.2;
  }
  if (sqrdist > 5) {
    sqrdist = 5;
  }
  scalefactor *= sqrdist;

  float mixSlaveImage = 0.f;
  float time = mAnimateTime;
  if (mAnimate && time < 0) {
    if (mAnimateScreenshot) {
      time = mAnimationFrame / 30.f;
    } else {
      time = mAnimationTimer.GetCurrentElapsedTime();
    }

    float maxTime = mAnimateVectors[0].back();
    mAnimationFrame++;
    if (time >= maxTime) {
      time = maxTime;
      mAnimate = 0;
      SetInfo("Animation finished. (%1.2f seconds, %d frames)", time, mAnimationFrame);
    } else {
      SetInfo("Running mAnimation: time %1.2f/%1.2f, frames %d", time, maxTime, mAnimationFrame);
    }
  }
  // Perform new rotation / translation
  if (mAnimate) {
    float vals[8];
    for (int i = 0; i < 8; i++) {
      vals[i] = mAnimationSplines[i].evaluate(time);
    }
    if (mAnimationChangeConfig && mixAnimation == false) {
      int base = 0;
      int k = mAnimateVectors[0].size() - 1;
      while (base < k && time > mAnimateVectors[0][base]) {
        base++;
      }
      if (base > mAnimationLastBase + 1) {
        mAnimationLastBase = base - 1;
      }

      if (base != mAnimationLastBase && mAnimateVectors[0][mAnimationLastBase] != mAnimateVectors[0][base] && memcmp(&mAnimateConfig[base], &mAnimateConfig[mAnimationLastBase], sizeof(mAnimateConfig[base]))) {
        mCfg = mAnimateConfig[mAnimationLastBase];
        updateConfig();
        if (mDrawQualityRenderToTexture) {
          setFrameBuffer(1, mMixBuffer.fb_id);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
          DrawGLScene_internal(true, time);
          setFrameBuffer();
        } else {
          DrawGLScene_internal(true, time);
          CHKERR(glBlitNamedFramebuffer(mMainBufferStack.back(), mMixBuffer.fb_id, 0, 0, mRenderwidth, mRenderheight, 0, 0, mRenderwidth, mRenderheight, GL_COLOR_BUFFER_BIT, GL_NEAREST));
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
        }
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity(); // Reset The Current Modelview Matrix
        mixSlaveImage = 1.f - (time - mAnimateVectors[0][mAnimationLastBase]) / (mAnimateVectors[0][base] - mAnimateVectors[0][mAnimationLastBase]);
      }

      if (memcmp(&mAnimateConfig[base], &mCfg, sizeof(mCfg))) {
        mCfg = mAnimateConfig[base];
        updateConfig();
      }
    }

    if (mCfg.animationMode != 6) {
      if (mCfg.animationMode & 1) // Rotation from euler angles
      {
        glRotatef(-vals[4] * 180.f / M_PI, 1, 0, 0);
        glRotatef(vals[5] * 180.f / M_PI, 0, 1, 0);
        glRotatef(-vals[6] * 180.f / M_PI, 0, 0, 1);
      } else { // Rotation from quaternion
        const float mag = sqrtf(vals[4] * vals[4] + vals[5] * vals[5] + vals[6] * vals[6] + vals[7] * vals[7]);
        if (mag < 0.0001) {
          vals[7] = 1;
        } else {
          for (int i = 0; i < 4; i++) {
            vals[4 + i] /= mag;
          }
        }

        float xx = vals[4] * vals[4], xy = vals[4] * vals[5], xz = vals[4] * vals[6], xw = vals[4] * vals[7], yy = vals[5] * vals[5], yz = vals[5] * vals[6], yw = vals[5] * vals[7], zz = vals[6] * vals[6], zw = vals[6] * vals[7];
        float mat[16] = { 1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), 0, 2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), 0, 2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), 0, 0, 0, 0, 1 };
        glMultMatrixf(mat);
      }
    }
    if (mCfg.animationMode & 4) // Compute cartesian translation from sperical coordinates (euler angles)
    {
      const float r = vals[3], phi = vals[1], theta = vals[2];
      vals[2] = r * cosf(phi) * cosf(theta);
      vals[0] = r * sinf(phi) * cosf(theta);
      vals[1] = r * sinf(theta);
    } else if (mCfg.animationMode & 2) { // Scale cartesion translation to interpolated radius
      float r = sqrtf(vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2]);
      if (fabsf(r) < 0.0001) {
        r = 1;
      }
      r = vals[3] / r;
      for (int i = 0; i < 3; i++) {
        vals[i] *= r;
      }
    }
    if (mCfg.animationMode == 6) {
      gluLookAt(vals[0], vals[1], vals[2], 0, 0, 0, 0, 1, 0);
    } else {
      glTranslatef(-vals[0], -vals[1], -vals[2]);
    }
  } else if (mResetScene) {
    glTranslatef(0, 0, -8);

    mCfg.pointSize = 2.0;
    mCfg.drawSlice = -1;
    mXadd = mZadd = 0;
    mCamLookOrigin = mCamYUp = false;
    mAngleRollOrigin = -1e9;

    mResetScene = 0;
    mUpdateDLList = true;
  } else {
    float moveZ = scalefactor * ((float)mMouseWheelTmp / 150 + (float)(mBackend->mKeys['W'] - mBackend->mKeys['S']) * (!mBackend->mKeys[mBackend->KEY_SHIFT]) * 0.2 * mFPSScale);
    float moveY = scalefactor * ((float)(mBackend->mKeys[mBackend->KEY_PAGEDOWN] - mBackend->mKeys[mBackend->KEY_PAGEUP]) * 0.2 * mFPSScale);
    float moveX = scalefactor * ((float)(mBackend->mKeys['A'] - mBackend->mKeys['D']) * (!mBackend->mKeys[mBackend->KEY_SHIFT]) * 0.2 * mFPSScale);
    float rotRoll = rotatescalefactor * mFPSScale * 2 * (mBackend->mKeys['E'] - mBackend->mKeys['F']) * (!mBackend->mKeys[mBackend->KEY_SHIFT]);
    float rotYaw = rotatescalefactor * mFPSScale * 2 * (mBackend->mKeys[mBackend->KEY_RIGHT] - mBackend->mKeys[mBackend->KEY_LEFT]);
    float rotPitch = rotatescalefactor * mFPSScale * 2 * (mBackend->mKeys[mBackend->KEY_DOWN] - mBackend->mKeys[mBackend->KEY_UP]);

    if (mBackend->mMouseDnR && mBackend->mMouseDn) {
      moveZ += -scalefactor * ((float)mBackend->mouseMvY - (float)mBackend->mMouseDnY) / 4;
      rotRoll += rotatescalefactor * ((float)mBackend->mouseMvX - (float)mBackend->mMouseDnX);
    } else if (mBackend->mMouseDnR) {
      moveX += -scalefactor * 0.5 * ((float)mBackend->mMouseDnX - (float)mBackend->mouseMvX) / 4;
      moveY += -scalefactor * 0.5 * ((float)mBackend->mouseMvY - (float)mBackend->mMouseDnY) / 4;
    } else if (mBackend->mMouseDn) {
      rotYaw += rotatescalefactor * ((float)mBackend->mouseMvX - (float)mBackend->mMouseDnX);
      rotPitch += rotatescalefactor * ((float)mBackend->mouseMvY - (float)mBackend->mMouseDnY);
    }

    if (mBackend->mKeys['<'] && !mBackend->mKeysShift['<']) {
      mAnimationDelay += moveX;
      if (mAnimationDelay < 0.05) {
        mAnimationDelay = 0.05;
      }
      moveX = 0.f;
      moveY = 0.f;
      SetInfo("Animation delay set to %1.2f", mAnimationDelay);
    }

    if (yUp) {
      mAngleRollOrigin = 0;
    } else if (!lookOrigin) {
      mAngleRollOrigin = -1e6;
    }
    if (lookOrigin) {
      if (!yUp) {
        if (mAngleRollOrigin < -1e6) {
          mAngleRollOrigin = yUp ? 0. : -mAngle[2];
        }
        mAngleRollOrigin += rotRoll;
        glRotatef(mAngleRollOrigin, 0, 0, 1);
        float tmpX = moveX, tmpY = moveY;
        moveX = tmpX * cosf(mAngle[2]) - tmpY * sinf(mAngle[2]);
        moveY = tmpX * sinf(mAngle[2]) + tmpY * cosf(mAngle[2]);
      }

      const float x = mXYZ[0], y = mXYZ[1], z = mXYZ[2];
      float r = sqrtf(x * x + +y * y + z * z);
      float r2 = sqrtf(x * x + z * z);
      float phi = atan2f(z, x);
      phi += moveX * 0.1f;
      float theta = atan2f(mXYZ[1], r2);
      theta -= moveY * 0.1f;
      const float max_theta = M_PI / 2 - 0.01;
      if (theta >= max_theta) {
        theta = max_theta;
      } else if (theta <= -max_theta) {
        theta = -max_theta;
      }
      if (moveZ >= r - 0.1) {
        moveZ = r - 0.1;
      }
      r -= moveZ;
      r2 = r * cosf(theta);
      mXYZ[0] = r2 * cosf(phi);
      mXYZ[2] = r2 * sinf(phi);
      mXYZ[1] = r * sinf(theta);

      gluLookAt(mXYZ[0], mXYZ[1], mXYZ[2], 0, 0, 0, 0, 1, 0);
    } else {
      glTranslatef(moveX, moveY, moveZ);
      if (rotYaw != 0.f) {
        glRotatef(rotYaw, 0, 1, 0);
      }
      if (rotPitch != 0.f) {
        glRotatef(rotPitch, 1, 0, 0);
      }
      if (!yUp && rotRoll != 0.f) {
        glRotatef(rotRoll, 0, 0, 1);
      }
      glMultMatrixf(mCurrentMatrix); // Apply previous translation / rotation

      if (yUp) {
        glGetFloatv(GL_MODELVIEW_MATRIX, mCurrentMatrix);
        calcXYZ();
        glLoadIdentity();
        glRotatef(mAngle[2] * 180.f / M_PI, 0, 0, 1);
        glMultMatrixf(mCurrentMatrix);
      }
    }

    // Graphichs Options
    float minSize = 0.4 / (mDrawQualityDownsampleFSAA > 1 ? mDrawQualityDownsampleFSAA : 1);
    int deltaLine = mBackend->mKeys['+'] * mBackend->mKeysShift['+'] - mBackend->mKeys['-'] * mBackend->mKeysShift['-'];
    mCfg.lineWidth += (float)deltaLine * mFPSScale * 0.02 * mCfg.lineWidth;
    if (mCfg.lineWidth < minSize) {
      mCfg.lineWidth = minSize;
    }
    if (deltaLine) {
      SetInfo("%s line width: %f", deltaLine > 0 ? "Increasing" : "Decreasing", mCfg.lineWidth);
    }
    minSize *= 2;
    int deltaPoint = mBackend->mKeys['+'] * (!mBackend->mKeysShift['+']) - mBackend->mKeys['-'] * (!mBackend->mKeysShift['-']);
    mCfg.pointSize += (float)deltaPoint * mFPSScale * 0.02 * mCfg.pointSize;
    if (mCfg.pointSize < minSize) {
      mCfg.pointSize = minSize;
    }
    if (deltaPoint) {
      SetInfo("%s point size: %f", deltaPoint > 0 ? "Increasing" : "Decreasing", mCfg.pointSize);
    }
  }

  // Store position
  if (mAnimateTime < 0) {
    glGetFloatv(GL_MODELVIEW_MATRIX, mCurrentMatrix);
    calcXYZ();
  }

  if (mBackend->mMouseDn || mBackend->mMouseDnR) {
    mBackend->mMouseDnX = mBackend->mouseMvX;
    mBackend->mMouseDnY = mBackend->mouseMvY;
  }

  // Open GL Default Values
  if (mCfg.smoothPoints) {
    CHKERR(glEnable(GL_POINT_SMOOTH));
  } else {
    CHKERR(glDisable(GL_POINT_SMOOTH));
  }
  if (mCfg.smoothLines) {
    CHKERR(glEnable(GL_LINE_SMOOTH));
  } else {
    CHKERR(glDisable(GL_LINE_SMOOTH));
  }
  CHKERR(glEnable(GL_BLEND));
  CHKERR(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  CHKERR(glPointSize(mCfg.pointSize * (mDrawQualityDownsampleFSAA > 1 ? mDrawQualityDownsampleFSAA : 1)));
  CHKERR(glLineWidth(mCfg.lineWidth * (mDrawQualityDownsampleFSAA > 1 ? mDrawQualityDownsampleFSAA : 1)));

  // Extract global cluster information
  if (mUpdateDLList) {
    showTimer = true;
    mTimerDraw.ResetStart();
    mCurrentClusters = 0;
    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      mCurrentClusters += sliceTracker(iSlice).NHitsTotal();
    }
    if (mNMaxClusters < mCurrentClusters) {
      mNMaxClusters = mCurrentClusters;
      mGlobalPosPtr.reset(new float4[mNMaxClusters]);
      mGlobalPos = mGlobalPosPtr.get();
    }

    mCurrentSpacePointsTRD = trdTracker().NTracklets();
    if (mCurrentSpacePointsTRD > mNMaxSpacePointsTRD) {
      mNMaxSpacePointsTRD = mCurrentSpacePointsTRD;
      mGlobalPosPtrTRD.reset(new float4[mNMaxSpacePointsTRD]);
      mGlobalPosPtrTRD2.reset(new float4[mNMaxSpacePointsTRD]);
      mGlobalPosTRD = mGlobalPosPtrTRD.get();
      mGlobalPosTRD2 = mGlobalPosPtrTRD2.get();
    }
    if ((size_t)mMerger.NOutputTracks() > mTRDTrackIds.size()) {
      mTRDTrackIds.resize(mMerger.NOutputTracks());
    }
    memset(mTRDTrackIds.data(), 0, sizeof(mTRDTrackIds[0]) * mMerger.NOutputTracks());
    for (int i = 0; i < trdTracker().NTracks(); i++) {
      if (trdTracker().Tracks()[i].GetNtracklets()) {
        mTRDTrackIds[trdTracker().Tracks()[i].GetTPCtrackId()] = i;
      }
    }

    mMaxClusterZ = 0;
    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (unsigned int i = 0; i < ioptrs().nClusterData[iSlice]; i++) {
        const auto& cl = ioptrs().clusterData[iSlice][i];
        const int cid = cl.id;
        if (cid >= mNMaxClusters) {
          printf("Cluster Buffer Size exceeded (id %d max %d)\n", cid, mNMaxClusters);
          return (1);
        }
        float4* ptr = &mGlobalPos[cid];
        mChain->GetParam().Slice2Global(iSlice, cl.x + mXadd, cl.y, cl.z, &ptr->x, &ptr->y, &ptr->z);
        if (fabsf(ptr->z) > mMaxClusterZ) {
          mMaxClusterZ = fabsf(ptr->z);
        }
        if (iSlice < 18) {
          ptr->z += mZadd;
          ptr->z += mZadd;
        } else {
          ptr->z -= mZadd;
          ptr->z -= mZadd;
        }

        ptr->x /= GL_SCALE_FACTOR;
        ptr->y /= GL_SCALE_FACTOR;
        ptr->z /= GL_SCALE_FACTOR;
        ptr->w = tCLUSTER;
      }
    }

    for (int i = 0; i < mCurrentSpacePointsTRD; i++) {
      const auto& sp = trdTracker().SpacePoints()[i];
      int iSec = mChain->GetTRDGeometry()->GetSector(trdTracker().Tracklets()[i].GetDetector());
      float4* ptr = &mGlobalPosTRD[i];
      mChain->GetParam().Slice2Global(iSec, sp.mR + mXadd, sp.mX[0], sp.mX[1], &ptr->x, &ptr->y, &ptr->z);
      ptr->x /= GL_SCALE_FACTOR;
      ptr->y /= GL_SCALE_FACTOR;
      ptr->z /= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
      ptr = &mGlobalPosTRD2[i];
      mChain->GetParam().Slice2Global(iSec, sp.mR + mXadd + 4.5f, sp.mX[0] + 1.5f * sp.mDy, sp.mX[1], &ptr->x, &ptr->y, &ptr->z);
      ptr->x /= GL_SCALE_FACTOR;
      ptr->y /= GL_SCALE_FACTOR;
      ptr->z /= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
    }

    mTimerFPS.ResetStart();
    mFramesDoneFPS = 0;
    mFPSScaleadjust = 0;
    mGlDLrecent = 0;
    mUpdateDLList = 0;
  }

  // Prepare Event
  if (!mGlDLrecent) {
    for (int i = 0; i < NSLICES; i++) {
      mVertexBuffer[i].clear();
      mVertexBufferStart[i].clear();
      mVertexBufferCount[i].clear();
    }

    for (int i = 0; i < mCurrentClusters; i++) {
      mGlobalPos[i].w = tCLUSTER;
    }
    for (int i = 0; i < mCurrentSpacePointsTRD; i++) {
      mGlobalPosTRD[i].w = tTRDCLUSTER;
    }

    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (int i = 0; i < N_POINTS_TYPE; i++) {
        mGlDLPoints[iSlice][i].resize(mNCollissions);
      }
      for (int i = 0; i < N_FINAL_TYPE; i++) {
        mGlDLFinal[iSlice].resize(mNCollissions);
      }
    }
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel num_threads(mChain->GetDeviceProcessingSettings().nThreads)
    {
      int numThread = omp_get_thread_num();
      int numThreads = omp_get_num_threads();
#pragma omp for
#else
    {
      int numThread = 0, numThreads = 1;
#endif
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        GPUTPCTracker& tracker = (GPUTPCTracker&)sliceTracker(iSlice);
        tracker.Data().SetPointersScratch(tracker.LinkTmpMemory());
        mGlDLLines[iSlice][tINITLINK] = DrawLinks(tracker, tINITLINK, true);
        tracker.Data().SetPointersScratch(mChain->rec()->Res(tracker.MemoryResLinksScratch()).Ptr());
      }
      GPUTPCGMPropagator prop;
      const float kRho = 1.025e-3;  // 0.9e-3;
      const float kRadLen = 29.532; // 28.94;
      prop.SetMaxSinPhi(.999);
      prop.SetMaterial(kRadLen, kRho);
      prop.SetPolynomialField(mMerger.pField());
      prop.SetToyMCEventsFlag(mMerger.Param().ToyMCEventsFlag);

#ifdef GPUCA_HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        const GPUTPCTracker& tracker = sliceTracker(iSlice);

        mGlDLLines[iSlice][tLINK] = DrawLinks(tracker, tLINK);
        mGlDLLines[iSlice][tSEED] = DrawSeeds(tracker);
        mGlDLLines[iSlice][tTRACKLET] = DrawTracklets(tracker);
        mGlDLLines[iSlice][tSLICETRACK] = DrawTracks(tracker, 0);
        mGlDLGrid[iSlice] = DrawGrid(tracker);
      }

#ifdef GPUCA_HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        const GPUTPCTracker& tracker = sliceTracker(iSlice);
        mGlDLLines[iSlice][tGLOBALTRACK] = DrawTracks(tracker, 1);
      }

#ifdef GPUCA_HAVE_OPENMP
#pragma omp barrier
#endif
      mThreadTracks[numThread].resize(mNCollissions);
      for (int i = 0; i < mNCollissions; i++) {
        for (int j = 0; j < NSLICES; j++) {
          for (int k = 0; k < 2; k++) {
            mThreadTracks[numThread][i][j][k].clear();
          }
        }
      }
#ifdef GPUCA_HAVE_OPENMP
#pragma omp for
#endif
      for (int i = 0; i < mMerger.NOutputTracks(); i++) {
        const GPUTPCGMMergedTrack* track = &mMerger.OutputTracks()[i];
        if (track->NClusters() == 0) {
          continue;
        }
        if (mHideRejectedTracks && !track->OK()) {
          continue;
        }
        int slice = mMerger.Clusters()[track->FirstClusterRef() + track->NClusters() - 1].slice;
        unsigned int col = 0;
        if (mNCollissions > 1) {
          int label = mQA ? mQA->GetMCLabel(i) : -1;
          if (label != -1e9 && label < -1) {
            label = -label - 2;
          }
          while (col < mCollisionClusters.size() && mCollisionClusters[col][NSLICES] < label) {
            col++;
          }
        }
        mThreadTracks[numThread][col][slice][0].emplace_back(i);
      }
#ifdef GPUCA_HAVE_OPENMP
#pragma omp for
#endif
      for (unsigned int i = 0; i < ioptrs().nMCInfosTPC; i++) {
        const GPUTPCMCInfo& mc = ioptrs().mcInfosTPC[i];
        if (mc.charge == 0.f) {
          continue;
        }
        if (mc.pid < 0) {
          continue;
        }

        float alpha = atan2f(mc.y, mc.x);
        if (alpha < 0) {
          alpha += 2 * M_PI;
        }
        int slice = alpha / (2 * M_PI) * 18;
        if (mc.z < 0) {
          slice += 18;
        }
        unsigned int col = 0;
        if (mNCollissions > 1) {
          while (col < mCollisionClusters.size() && mCollisionClusters[col][NSLICES] < (int)i) {
            col++;
          }
        }
        mThreadTracks[numThread][col][slice][1].emplace_back(i);
      }
#ifdef GPUCA_HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mThreadBuffers[numThread].clear();
          for (int iSet = 0; iSet < numThreads; iSet++) {
            DrawFinal(iSlice, iCol, &prop, mThreadTracks[iSet][iCol][iSlice], mThreadBuffers[numThread]);
          }
          vboList* list = &mGlDLFinal[iSlice][iCol][0];
          for (int i = 0; i < N_FINAL_TYPE; i++) {
            size_t startCount = mVertexBufferStart[iSlice].size();
            for (unsigned int j = 0; j < mThreadBuffers[numThread].start[i].size(); j++) {
              mVertexBufferStart[iSlice].emplace_back(mThreadBuffers[numThread].start[i][j]);
              mVertexBufferCount[iSlice].emplace_back(mThreadBuffers[numThread].count[i][j]);
            }
            list[i] = vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice);
          }
        }
      }

#ifdef GPUCA_HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        const GPUTPCTracker& tracker = sliceTracker(iSlice);
        for (int i = 0; i < N_POINTS_TYPE_TPC; i++) {
          for (int iCol = 0; iCol < mNCollissions; iCol++) {
            mGlDLPoints[iSlice][i][iCol] = DrawClusters(tracker, i, iCol);
          }
        }
      }
    }
    // End omp parallel

    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (int i = N_POINTS_TYPE_TPC; i < N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD; i++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mGlDLPoints[iSlice][i][iCol] = DrawSpacePointsTRD(iSlice, i, iCol);
        }
      }
    }

    mGlDLrecent = 1;
    size_t totalVertizes = 0;
    for (int i = 0; i < NSLICES; i++) {
      totalVertizes += mVertexBuffer[i].size();
    }

    mUseMultiVBO = (totalVertizes * sizeof(mVertexBuffer[0][0]) >= 0x100000000ll);
    if (mUseMultiVBO) {
      for (int i = 0; i < NSLICES; i++) {
        CHKERR(glNamedBufferData(mVBOId[i], mVertexBuffer[i].size() * sizeof(mVertexBuffer[i][0]), mVertexBuffer[i].data(), GL_STATIC_DRAW));
        mVertexBuffer[i].clear();
      }
    } else {
      size_t totalYet = mVertexBuffer[0].size();
      mVertexBuffer[0].resize(totalVertizes);
      for (int i = 1; i < NSLICES; i++) {
        for (unsigned int j = 0; j < mVertexBufferStart[i].size(); j++) {
          mVertexBufferStart[i][j] += totalYet;
        }
        memcpy(&mVertexBuffer[0][totalYet], &mVertexBuffer[i][0], mVertexBuffer[i].size() * sizeof(mVertexBuffer[i][0]));
        totalYet += mVertexBuffer[i].size();
        mVertexBuffer[i].clear();
      }
      CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0])); // Bind ahead of time, since it is not going to change
      CHKERR(glNamedBufferData(mVBOId[0], totalVertizes * sizeof(mVertexBuffer[0][0]), mVertexBuffer[0].data(), GL_STATIC_DRAW));
      mVertexBuffer[0].clear();
    }

    if (mUseGLIndirectDraw) {
      mCmdBuffer.clear();
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        mIndirectSliceOffset[iSlice] = mCmdBuffer.size();
        for (unsigned int k = 0; k < mVertexBufferStart[iSlice].size(); k++) {
          mCmdBuffer.emplace_back(mVertexBufferCount[iSlice][k], 1, mVertexBufferStart[iSlice][k], 0);
        }
      }
      CHKERR(glBufferData(GL_DRAW_INDIRECT_BUFFER, mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), GL_STATIC_DRAW));
    }

    if (showTimer) {
      printf("Draw time: %'d us (vertices %'lld / %'lld bytes)\n", (int)(mTimerDraw.GetCurrentElapsedTime() * 1000000.), (long long int)totalVertizes, (long long int)(totalVertizes * sizeof(mVertexBuffer[0][0])));
    }
  }

  // Draw Event
  mNDrawCalls = 0;
  CHKERR(glEnableClientState(GL_VERTEX_ARRAY));
  CHKERR(glVertexPointer(3, GL_FLOAT, 0, nullptr));

#define LOOP_SLICE for (int iSlice = (mCfg.drawSlice == -1 ? 0 : mCfg.drawRelatedSlices ? (mCfg.drawSlice % 9) : mCfg.drawSlice); iSlice < NSLICES; iSlice += (mCfg.drawSlice == -1 ? 1 : mCfg.drawRelatedSlices ? 9 : NSLICES))
#define LOOP_COLLISION for (int iCol = (mCfg.showCollision == -1 ? 0 : mCfg.showCollision); iCol < mNCollissions; iCol += (mCfg.showCollision == -1 ? 1 : mNCollissions))
#define LOOP_COLLISION_COL(cmd) \
  LOOP_COLLISION                \
  {                             \
    if (mCfg.colorCollisions) { \
      SetCollisionColor(iCol);  \
    }                           \
    cmd;                        \
  }

  if (mCfg.drawGrid) {
    SetColorGrid();
    LOOP_SLICE drawVertices(mGlDLGrid[iSlice], GL_LINES);
  }
  if (mCfg.drawClusters) {
    if (mCfg.drawTRD) {
      SetColorTRD();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDCLUSTER][iCol], GL_LINES));
      if (mCfg.drawFinal) {
        if (mCfg.colorClusters) {
          SetColorFinal();
        }
        LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDATTACHED][iCol], GL_POINTS));
      } else {
        LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDATTACHED][iCol], GL_LINES));
      }
    }
    if (mCfg.drawTPC) {
      SetColorClusters();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tCLUSTER][iCol], GL_POINTS));

      if (mCfg.drawInitLinks) {
        if (mCfg.excludeClusters) {
          goto skip1;
        }
        if (mCfg.colorClusters) {
          SetColorInitLinks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tINITLINK][iCol], GL_POINTS));

      if (mCfg.drawLinks) {
        if (mCfg.excludeClusters) {
          goto skip1;
        }
        if (mCfg.colorClusters) {
          SetColorLinks();
        }
      } else {
        SetColorClusters();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tLINK][iCol], GL_POINTS));

      if (mCfg.drawSeeds) {
        if (mCfg.excludeClusters) {
          goto skip1;
        }
        if (mCfg.colorClusters) {
          SetColorSeeds();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSEED][iCol], GL_POINTS));

    skip1:
      SetColorClusters();
      if (mCfg.drawTracklets) {
        if (mCfg.excludeClusters) {
          goto skip2;
        }
        if (mCfg.colorClusters) {
          SetColorTracklets();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRACKLET][iCol], GL_POINTS));

      if (mCfg.drawTracks) {
        if (mCfg.excludeClusters) {
          goto skip2;
        }
        if (mCfg.colorClusters) {
          SetColorTracks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSLICETRACK][iCol], GL_POINTS));

    skip2:;
      if (mCfg.drawGlobalTracks) {
        if (mCfg.excludeClusters) {
          goto skip3;
        }
        if (mCfg.colorClusters) {
          SetColorGlobalTracks();
        }
      } else {
        SetColorClusters();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tGLOBALTRACK][iCol], GL_POINTS));
      SetColorClusters();

      if (mCfg.drawFinal && mCfg.propagateTracks < 2) {
        if (mCfg.excludeClusters) {
          goto skip3;
        }
        if (mCfg.colorClusters) {
          SetColorFinal();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tFINALTRACK][iCol], GL_POINTS));
    skip3:;
    }
  }

  if (!mConfig.clustersOnly && !mCfg.excludeClusters) {
    if (mCfg.drawTPC) {
      if (mCfg.drawInitLinks) {
        SetColorInitLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tINITLINK], GL_LINES);
      }
      if (mCfg.drawLinks) {
        SetColorLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tLINK], GL_LINES);
      }
      if (mCfg.drawSeeds) {
        SetColorSeeds();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSEED], GL_LINE_STRIP);
      }
      if (mCfg.drawTracklets) {
        SetColorTracklets();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tTRACKLET], GL_LINE_STRIP);
      }
      if (mCfg.drawTracks) {
        SetColorTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSLICETRACK], GL_LINE_STRIP);
      }
      if (mCfg.drawGlobalTracks) {
        SetColorGlobalTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tGLOBALTRACK], GL_LINE_STRIP);
      }
    }
    if (mCfg.drawFinal) {
      SetColorFinal();
      LOOP_SLICE LOOP_COLLISION
      {
        if (mCfg.colorCollisions) {
          SetCollisionColor(iCol);
        }
        if (mCfg.propagateTracks < 2) {
          drawVertices(mGlDLFinal[iSlice][iCol][0], GL_LINE_STRIP);
        }
        if (mCfg.propagateTracks > 0 && mCfg.propagateTracks < 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][1], GL_LINE_STRIP);
        }
        if (mCfg.propagateTracks == 2) {
          drawVertices(mGlDLFinal[iSlice][iCol][2], GL_LINE_STRIP);
        }
        if (mCfg.propagateTracks == 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][3], GL_LINE_STRIP);
        }
      }
    }
    if (mMarkClusters || markAdjacentClusters) {
      SetColorMarked();
      LOOP_SLICE LOOP_COLLISION drawVertices(mGlDLPoints[iSlice][tMARKED][iCol], GL_POINTS);
    }
  }

  CHKERR(glDisableClientState(GL_VERTEX_ARRAY));

  if (mixSlaveImage > 0) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.f, mRenderwidth, 0.f, mRenderheight);
    CHKERR(glEnable(GL_TEXTURE_2D));
    glDisable(GL_DEPTH_TEST);
    CHKERR(glBindTexture(GL_TEXTURE_2D, mMixBuffer.fbCol_id));
    glColor4f(1, 1, 1, mixSlaveImage);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, 1);
    glVertex3f(0, mRenderheight, 0);
    glTexCoord2f(1, 1);
    glVertex3f(mRenderwidth, mRenderheight, 0);
    glTexCoord2f(1, 0);
    glVertex3f(mRenderwidth, 0, 0);
    glEnd();
    glColor4f(1, 1, 1, 0);
    CHKERR(glDisable(GL_TEXTURE_2D));
    setDepthBuffer();
    glPopMatrix();
  }

  if (mixAnimation) {
    glColorMask(false, false, false, true);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(true, true, true, true);
  } else if (mOffscreenBuffer.created) {
    setFrameBuffer();
    GLuint srcid = mOffscreenBuffer.fb_id;
    if (mDrawQualityMSAA > 1 && mDrawQualityDownsampleFSAA > 1) {
      CHKERR(glBlitNamedFramebuffer(srcid, mOffscreenBufferNoMSAA.fb_id, 0, 0, mRenderwidth, mRenderheight, 0, 0, mRenderwidth, mRenderheight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
      srcid = mOffscreenBufferNoMSAA.fb_id;
    }
    CHKERR(glBlitNamedFramebuffer(srcid, mMainBufferStack.back(), 0, 0, mRenderwidth, mRenderheight, 0, 0, mScreenwidth, mScreenheight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
  }

  if (mAnimate && mAnimateScreenshot && mAnimateTime < 0) {
    char mAnimateScreenshotFile[48];
    sprintf(mAnimateScreenshotFile, "mAnimation%d_%05d.bmp", mAnimationExport, mAnimationFrame);
    DoScreenshot(mAnimateScreenshotFile, time);
  }

  if (mAnimateTime < 0) {
    mFramesDone++;
    mFramesDoneFPS++;
    double fpstime = mTimerFPS.GetCurrentElapsedTime();
    char info[1024];
    float fps = (double)mFramesDoneFPS / fpstime;
    sprintf(info,
            "FPS: %6.2f (Slice: %d, 1:Clusters %d, 2:Prelinks %d, 3:Links %d, 4:Seeds %d, 5:Tracklets %d, 6:Tracks %d, 7:GTracks %d, 8:Merger %d) (%d frames, %d draw calls) "
            "(X %1.2f Y %1.2f Z %1.2f / R %1.2f Phi %1.1f Theta %1.1f) / Yaw %1.1f Pitch %1.1f Roll %1.1f)",
            fps, mCfg.drawSlice, mCfg.drawClusters, mCfg.drawInitLinks, mCfg.drawLinks, mCfg.drawSeeds, mCfg.drawTracklets, mCfg.drawTracks, mCfg.drawGlobalTracks, mCfg.drawFinal, mFramesDone, mNDrawCalls, mXYZ[0], mXYZ[1], mXYZ[2], mRPhiTheta[0], mRPhiTheta[1] * 180 / M_PI,
            mRPhiTheta[2] * 180 / M_PI, mAngle[1] * 180 / M_PI, mAngle[0] * 180 / M_PI, mAngle[2] * 180 / M_PI);
    if (fpstime > 1.) {
      if (mPrintInfoText & 2) {
        printf("%s\n", info);
      }
      if (mFPSScaleadjust++) {
        mFPSScale = 60 / fps;
      }
      mTimerFPS.ResetStart();
      mFramesDoneFPS = 0;
    }

    if (mPrintInfoText & 1) {
      setFrameBuffer(0, 0);
      showInfo(info);
      setFrameBuffer(-2);
    }
  }

  if (mAnimateTime < 0) {
    mSemLockDisplay.Unlock();
  }

  return (0);
}

void GPUDisplay::DoScreenshot(char* filename, float mAnimateTime)
{
  int SCALE_Y = screenshot_scale, SCALE_X = screenshot_scale;

  float tmpPointSize = mCfg.pointSize;
  float tmpLineWidth = mCfg.lineWidth;
  mCfg.pointSize *= (float)(SCALE_X + SCALE_Y) / 2.;
  mCfg.lineWidth *= (float)(SCALE_X + SCALE_Y) / 2.;

  int oldWidth = mScreenwidth, oldHeight = mScreenheight;
  GLfb screenshotBuffer;

  bool needBuffer = SCALE_X != 1 || SCALE_Y != 1;

  if (needBuffer) {
    deleteFB(mMixBuffer);
    mScreenwidth *= SCALE_X;
    mScreenheight *= SCALE_Y;
    mRenderwidth = mScreenwidth;
    mRenderheight = mScreenheight;
    createFB(screenshotBuffer, 0, 1, false); // Create screenshotBuffer of size mScreenwidth * SCALE, mRenderwidth * SCALE
    UpdateOffscreenBuffers();                // Create other buffers of size mScreenwidth * SCALE * downscale, ...
    setFrameBuffer(1, screenshotBuffer.fb_id);
    glViewport(0, 0, mRenderwidth, mRenderheight);
    DrawGLScene(false, mAnimateTime);
  }
  size_t size = 4 * mScreenwidth * mScreenheight;
  unsigned char* pixels = new unsigned char[size];
  CHKERR(glPixelStorei(GL_PACK_ALIGNMENT, 1));
  CHKERR(glReadBuffer(needBuffer ? GL_COLOR_ATTACHMENT0 : GL_BACK));
  CHKERR(glReadPixels(0, 0, mScreenwidth, mScreenheight, GL_BGRA, GL_UNSIGNED_BYTE, pixels));

  if (filename) {
    FILE* fp = fopen(filename, "w+b");

    BITMAPFILEHEADER bmpFH;
    BITMAPINFOHEADER bmpIH;
    memset(&bmpFH, 0, sizeof(bmpFH));
    memset(&bmpIH, 0, sizeof(bmpIH));

    bmpFH.bfType = 19778; //"BM"
    bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + size;
    bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

    bmpIH.biSize = sizeof(bmpIH);
    bmpIH.biWidth = mScreenwidth;
    bmpIH.biHeight = mScreenheight;
    bmpIH.biPlanes = 1;
    bmpIH.biBitCount = 32;
    bmpIH.biCompression = BI_RGB;
    bmpIH.biSizeImage = size;
    bmpIH.biXPelsPerMeter = 5670;
    bmpIH.biYPelsPerMeter = 5670;

    fwrite(&bmpFH, 1, sizeof(bmpFH), fp);
    fwrite(&bmpIH, 1, sizeof(bmpIH), fp);
    fwrite(pixels, 1, size, fp);
    fclose(fp);
  }
  delete[] pixels;

  mCfg.pointSize = tmpPointSize;
  mCfg.lineWidth = tmpLineWidth;
  if (needBuffer) {
    setFrameBuffer();
    deleteFB(screenshotBuffer);
    mScreenwidth = oldWidth;
    mScreenheight = oldHeight;
    UpdateOffscreenBuffers();
    glViewport(0, 0, mRenderwidth, mRenderheight);
    DrawGLScene(false, mAnimateTime);
  }
}

void GPUDisplay::showInfo(const char* info)
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0.f, mScreenwidth, 0.f, mScreenheight);
  glViewport(0, 0, mScreenwidth, mScreenheight);
  float colorValue = mInvertColors ? 0.f : 1.f;
  mBackend->OpenGLPrint(info, 40.f, 40.f, colorValue, colorValue, colorValue, 1);
  if (mInfoText2Timer.IsRunning()) {
    if (mInfoText2Timer.GetCurrentElapsedTime() >= 6) {
      mInfoText2Timer.Reset();
    } else {
      mBackend->OpenGLPrint(mInfoText2, 40.f, 20.f, colorValue, colorValue, colorValue, 6 - mInfoText2Timer.GetCurrentElapsedTime());
    }
  }
  if (mInfoHelpTimer.IsRunning()) {
    if (mInfoHelpTimer.GetCurrentElapsedTime() >= 6) {
      mInfoHelpTimer.Reset();
    } else {
      PrintGLHelpText(colorValue);
    }
  }
  glColor4f(colorValue, colorValue, colorValue, 0);
  glViewport(0, 0, mRenderwidth, mRenderheight);
  glPopMatrix();
}

void GPUDisplay::ShowNextEvent()
{
  mSemLockDisplay.Unlock();
  mBackend->mNeedUpdate = 1;
  mUpdateDLList = true;
}

void GPUDisplay::WaitForNextEvent() { mSemLockDisplay.Lock(); }

int GPUDisplay::StartDisplay()
{
  if (mBackend->StartDisplay()) {
    return (1);
  }
  while (mInitResult == 0) {
    Sleep(10);
  }
  return (mInitResult != 1);
}
