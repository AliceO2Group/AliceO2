// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplay.cxx
/// \author David Rohr

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#endif

#include "GPUDisplay.h"

#ifdef GPUCA_BUILD_EVENT_DISPLAY
#include "GPUTPCDef.h"

#include <GL/glu.h>
#include <vector>
#include <array>
#include <tuple>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#ifndef _WIN32
#include "bitmapfile.h"
#include "../utils/linux_helpers.h"
#endif
#ifdef WITH_OPENMP
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
#include "GPUTrackParamConvert.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.inc"
#include "GPUTPCConvertImpl.h"
#include "utils/qconfig.h"

#ifdef GPUCA_HAVE_O2HEADERS
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "TOFBase/Geo.h"
#include "ITSBase/GeometryTGeo.h"
#endif
#ifdef GPUCA_O2_LIB
#include "ITSMFTBase/DPLAlpideParam.h"
#endif

#include "GPUDisplayShaders.h"

constexpr hmm_mat4 MY_HMM_IDENTITY = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
constexpr hmm_mat4 MY_HMM_FROM(float (&v)[16]) { return {{{v[0], v[1], v[2], v[3]}, {v[4], v[5], v[6], v[7]}, {v[8], v[9], v[10], v[11]}, {v[12], v[13], v[14], v[15]}}}; }

using namespace GPUCA_NAMESPACE::gpu;

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

#define OPENGL_EMULATE_MULTI_DRAW 0

#define GL_SCALE_FACTOR 100.f

#define SEPERATE_GLOBAL_TRACKS_LIMIT (mCfgH.separateGlobalTracks ? tGLOBALTRACK : TRACK_TYPE_ID_LIMIT)

#define GET_CID(slice, i) (mParam->par.earlyTpcTransform ? mIOPtrs->clusterData[slice][i].id : (mIOPtrs->clustersNative->clusterOffset[slice][0] + i))

static const GPUSettingsDisplay& GPUDisplay_GetConfig(GPUChainTracking* chain)
{
  static GPUSettingsDisplay defaultConfig;
  if (chain && chain->mConfigDisplay) {
    return *chain->mConfigDisplay;
  } else {
    return defaultConfig;
  }
}

GPUDisplay::GPUDisplay(GPUDisplayBackend* backend, GPUChainTracking* chain, GPUQA* qa, const GPUParam* param, const GPUCalibObjectsConst* calib, const GPUSettingsDisplay* config) : mBackend(backend), mChain(chain), mConfig(config ? *config : GPUDisplay_GetConfig(chain)), mQA(qa)
{
  backend->mDisplay = this;
  mCfgR.openGLCore = GPUCA_DISPLAY_OPENGL_CORE_FLAGS;
  mParam = param ? param : &mChain->GetParam();
  mCalib = calib;
  mCfgL = mConfig.light;
  mCfgH = mConfig.heavy;
  mCfgR = mConfig.renderer;
}

inline const GPUTRDGeometry& GPUDisplay::trdGeometry() { return *(GPUTRDGeometry*)mCalib->trdGeometry; }
const GPUTPCTracker& GPUDisplay::sliceTracker(int iSlice) { return mChain->GetTPCSliceTrackers()[iSlice]; }
const GPUTRDTrackerGPU& GPUDisplay::trdTracker() { return *mChain->GetTRDTrackerGPU(); }
inline int GPUDisplay::getNumThreads()
{
  if (mChain) {
    return mChain->GetProcessingSettings().ompThreads;
  } else {
#ifdef WITH_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }
}

void GPUDisplay::disableUnsupportedOptions()
{
  if (!mIOPtrs->mergedTrackHitAttachment) {
    mCfgH.markAdjacentClusters = 0;
  }
  if (!mQA) {
    mCfgH.markFakeClusters = 0;
  }
  if (!mChain) {
    mCfgL.excludeClusters = mCfgL.drawInitLinks = mCfgL.drawLinks = mCfgL.drawSeeds = mCfgL.drawTracklets = mCfgL.drawTracks = mCfgL.drawGlobalTracks = 0;
  }
  if (mConfig.showTPCTracksFromO2Format && mParam->par.earlyTpcTransform) {
    throw std::runtime_error("Cannot run GPU display with early Transform when input is O2 tracks");
  }
}

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
    if (mCfgR.openGLCore) {
      CHKERR(glBindVertexArray(mVertexArray));
    }
    CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[iSlice]));
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    if (!mCfgR.openGLCore) {
      CHKERR(glVertexPointer(3, GL_FLOAT, 0, nullptr));
    } else
#endif
    {
      CHKERR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr));
      glEnableVertexAttribArray(0);
    }
  }

  if (mCfgR.useGLIndirectDraw) {
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

void GPUDisplay::calcXYZ(const float* matrix)
{
  mXYZ[0] = -(matrix[0] * matrix[12] + matrix[1] * matrix[13] + matrix[2] * matrix[14]);
  mXYZ[1] = -(matrix[4] * matrix[12] + matrix[5] * matrix[13] + matrix[6] * matrix[14]);
  mXYZ[2] = -(matrix[8] * matrix[12] + matrix[9] * matrix[13] + matrix[10] * matrix[14]);

  mAngle[0] = -asinf(matrix[6]); // Invert rotY*rotX*rotZ
  float A = cosf(mAngle[0]);
  if (fabsf(A) > 0.005) {
    mAngle[1] = atan2f(-matrix[2] / A, matrix[10] / A);
    mAngle[2] = atan2f(matrix[4] / A, matrix[5] / A);
  } else {
    mAngle[1] = 0;
    mAngle[2] = atan2f(-matrix[1], -matrix[0]);
  }

  mRPhiTheta[0] = sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[1] * mXYZ[1] + mXYZ[2] * mXYZ[2]);
  mRPhiTheta[1] = atan2f(mXYZ[0], mXYZ[2]);
  mRPhiTheta[2] = atan2f(mXYZ[1], sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[2] * mXYZ[2]));

  createQuaternionFromMatrix(mQuat, matrix);

  /*float mAngle[1] = -asinf(matrix[2]); //Calculate Y-axis angle - for rotX*rotY*rotZ
  float C = cosf( angle_y );
  if (fabsf(C) > 0.005) //Gimball lock?
  {
      mAngle[0]  = atan2f(-matrix[6] / C, matrix[10] / C);
      mAngle[2]  = atan2f(-matrix[1] / C, matrix[0] / C);
  }
  else
  {
      mAngle[0]  = 0; //set x-angle
      mAngle[2]  = atan2f(matrix[4], matrix[5]);
  }*/
}

void GPUDisplay::SetCollisionFirstCluster(unsigned int collision, int slice, int cluster)
{
  mNCollissions = std::max<unsigned int>(mNCollissions, collision + 1);
  mCollisionClusters.resize(mNCollissions);
  mCollisionClusters[collision][slice] = cluster;
}

void GPUDisplay::mAnimationCloseAngle(float& newangle, float lastAngle)
{
  const float delta = lastAngle > newangle ? (2 * CAMath::Pi()) : (-2 * CAMath::Pi());
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
  if (mCfgL.animationMode & 4) // Spherical
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
  if (mCfgL.animationMode & 1) // Euler-angles
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
    createQuaternionFromMatrix(v, mViewMatrixP);
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
  mAnimateConfig.emplace_back(mCfgL);
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

inline void GPUDisplay::ActivateColor()
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mCfgR.openGLCore) {
    glColor3f(mDrawColor[0], mDrawColor[1], mDrawColor[2]);
  } else
#endif
  {
    glUniform3fv(mColorId, 1, &mDrawColor[0]);
  }
}

inline void GPUDisplay::SetColorClusters()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.3, 0.7};
  } else {
    mDrawColor = {0, 0.7, 1.0};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTRD()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.7, 0.3, 0};
  } else {
    mDrawColor = {1.0, 0.7, 0};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorITS()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {1.00, 0.1, 0.1};
  } else {
    mDrawColor = {1.00, 0.3, 0.3};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTOF()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.1, 1.0, 0.1};
  } else {
    mDrawColor = {0.5, 1.0, 0.5};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorInitLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.42, 0.4, 0.1};
  } else {
    mDrawColor = {0.42, 0.4, 0.1};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.1, 0.1};
  } else {
    mDrawColor = {0.8, 0.2, 0.2};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorSeeds()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.0, 0.65};
  } else {
    mDrawColor = {0.8, 0.1, 0.85};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracklets()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0, 0};
  } else {
    mDrawColor = {1, 1, 1};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0, 0.1};
  } else {
    mDrawColor = {0.8, 1., 0.15};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGlobalTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0.2, 0};
  } else {
    mDrawColor = {1.0, 0.4, 0};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorFinal()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.6, 0.1};
  } else {
    mDrawColor = {0, 0.7, 0.2};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGrid()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.0};
  } else {
    mDrawColor = {0.7, 0.7, 0.0};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGridTRD()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.5};
  } else {
    mDrawColor = {0.7, 0.7, 0.5};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorMarked()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0, 0};
  } else {
    mDrawColor = {1.0, 0.0, 0.0};
  }
  ActivateColor();
}
inline void GPUDisplay::SetCollisionColor(int col)
{
  int red = (col * 2) % 5;
  int blue = (2 + col * 3) % 7;
  int green = (4 + col * 5) % 6;
  if (mCfgL.invertColors && red == 4 && blue == 5 && green == 6) {
    red = 0;
  }
  if (!mCfgL.invertColors && red == 0 && blue == 0 && green == 0) {
    red = 4;
  }
  mDrawColor = {red / 4.f, green / 5.f, blue / 6.f};
  ActivateColor();
}

void GPUDisplay::setQuality()
{
  // Doesn't seem to make a difference in this applicattion
  if (mCfgR.drawQualityMSAA > 1) {
    CHKERR(glEnable(GL_MULTISAMPLE));
  } else {
    CHKERR(glDisable(GL_MULTISAMPLE));
  }
}

void GPUDisplay::setDepthBuffer()
{
  if (mCfgL.depthBuffer) {
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
    CHKERR(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, mCfgR.drawQualityMSAA, storage, mRenderwidth, mRenderheight, false));
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
    CHKERR(glRenderbufferStorageMultisample(GL_RENDERBUFFER, mCfgR.drawQualityMSAA, storage, mRenderwidth, mRenderheight));
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
    GPUError("Error creating framebuffer (tex %d) - incomplete (%d)", (int)tex, status);
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

  if (mCfgR.drawQualityDownsampleFSAA > 1) {
    mRenderwidth = mScreenwidth * mCfgR.drawQualityDownsampleFSAA;
    mRenderheight = mScreenheight * mCfgR.drawQualityDownsampleFSAA;
  } else {
    mRenderwidth = mScreenwidth;
    mRenderheight = mScreenheight;
  }
  if (mCfgR.drawQualityMSAA > 1 || mCfgR.drawQualityDownsampleFSAA > 1) {
    createFB(mOffscreenBuffer, false, true, mCfgR.drawQualityMSAA > 1);
    if (mCfgR.drawQualityMSAA > 1 && mCfgR.drawQualityDownsampleFSAA > 1) {
      createFB(mOffscreenBufferNoMSAA, false, true, false);
    }
  }
  createFB(mMixBuffer, true, true, false);
  glViewport(0, 0, mRenderwidth, mRenderheight);
  setQuality();
}

void GPUDisplay::ReSizeGLScene(int width, int height, bool init)
{
  if (height == 0) { // Prevent A Divide By Zero By
    height = 1;      // Making Height Equal One
  }
  mScreenwidth = width;
  mScreenheight = height;
  UpdateOffscreenBuffers();

  if (init) {
    mResetScene = 1;
    mViewMatrix = MY_HMM_IDENTITY;
    mModelMatrix = MY_HMM_IDENTITY;
  }
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
  int glVersion[2] = {0, 0};
  glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]);
  glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);
  if (glVersion[0] < GPUDisplayBackend::GL_MIN_VERSION_MAJOR || (glVersion[0] == GPUDisplayBackend::GL_MIN_VERSION_MAJOR && glVersion[1] < GPUDisplayBackend::GL_MIN_VERSION_MINOR)) {
    GPUError("Unsupported OpenGL runtime %d.%d < %d.%d", glVersion[0], glVersion[1], GPUDisplayBackend::GL_MIN_VERSION_MAJOR, GPUDisplayBackend::GL_MIN_VERSION_MINOR);
    return (1);
  }

  CHKERR(glCreateBuffers(GPUChainTracking::NSLICES, mVBOId));
  CHKERR(glBindBuffer(GL_ARRAY_BUFFER, mVBOId[0]));
  CHKERR(glGenBuffers(1, &mIndirectId));
  CHKERR(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, mIndirectId));
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  CHKERR(glShadeModel(GL_SMOOTH)); // Enable Smooth Shading
#endif
  setDepthBuffer();
  setQuality();
  ReSizeGLScene(GPUDisplayBackend::INIT_WIDTH, GPUDisplayBackend::INIT_HEIGHT, true);
  mThreadBuffers.resize(getNumThreads());
  mThreadTracks.resize(getNumThreads());
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

void GPUDisplay::ExitGL()
{
  UpdateOffscreenBuffers(true);
  CHKERR(glDeleteBuffers(GPUChainTracking::NSLICES, mVBOId));
  CHKERR(glDeleteBuffers(1, &mIndirectId));
  CHKERR(glDeleteProgram(mShaderProgram));
  CHKERR(glDeleteShader(mVertexShader));
  CHKERR(glDeleteShader(mFragmentShader));
}

inline void GPUDisplay::drawPointLinestrip(int iSlice, int cid, int id, int id_limit)
{
  mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y, mCfgH.projectXY ? 0 : mGlobalPos[cid].z);
  if (mGlobalPos[cid].w < id_limit) {
    mGlobalPos[cid].w = id;
  }
}

GPUDisplay::vboList GPUDisplay::DrawSpacePointsTRD(int iSlice, int select, int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();

  if (iCol == 0) {
    for (unsigned int i = 0; i < mIOPtrs->nTRDTracklets; i++) {
      int iSec = trdGeometry().GetSector(mIOPtrs->trdTracklets[i].GetDetector());
      bool draw = iSlice == iSec && mGlobalPosTRD[i].w == select;
      if (draw) {
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[i].x, mGlobalPosTRD[i].y, mCfgH.projectXY ? 0 : mGlobalPosTRD[i].z);
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[i].x, mGlobalPosTRD2[i].y, mCfgH.projectXY ? 0 : mGlobalPosTRD2[i].z);
      }
    }
  }

  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawSpacePointsTOF(int iSlice, int select, int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();

  if (iCol == 0 && iSlice == 0) {
    for (unsigned int i = 0; i < mIOPtrs->nTOFClusters; i++) {
      mVertexBuffer[iSlice].emplace_back(mGlobalPosTOF[i].x, mGlobalPosTOF[i].y, mCfgH.projectXY ? 0 : mGlobalPosTOF[i].z);
    }
  }

  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawSpacePointsITS(int iSlice, int select, int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();

  if (iCol == 0 && iSlice == 0 && mIOPtrs->itsClusters) {
    for (unsigned int i = 0; i < mIOPtrs->nItsClusters; i++) {
      mVertexBuffer[iSlice].emplace_back(mGlobalPosITS[i].x, mGlobalPosITS[i].y, mCfgH.projectXY ? 0 : mGlobalPosITS[i].z);
    }
  }

  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawClusters(int iSlice, int select, unsigned int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();
  const int firstCluster = (mCollisionClusters.size() > 1 && iCol > 0) ? mCollisionClusters[iCol - 1][iSlice] : 0;
  const int lastCluster = (mCollisionClusters.size() > 1 && iCol + 1 < mCollisionClusters.size()) ? mCollisionClusters[iCol][iSlice] : (mParam->par.earlyTpcTransform ? mIOPtrs->nClusterData[iSlice] : mIOPtrs->clustersNative ? mIOPtrs->clustersNative->nClustersSector[iSlice] : 0);
  for (int cidInSlice = firstCluster; cidInSlice < lastCluster; cidInSlice++) {
    const int cid = GET_CID(iSlice, cidInSlice);
    if (mCfgH.hideUnmatchedClusters && mQA && mQA->SuppressHit(cid)) {
      continue;
    }
    bool draw = mGlobalPos[cid].w == select;

    if (mCfgH.markAdjacentClusters) {
      const int attach = mIOPtrs->mergedTrackHitAttachment[cid];
      if (attach) {
        if (mCfgH.markAdjacentClusters >= 32) {
          if (mQA && mQA->clusterRemovable(attach, mCfgH.markAdjacentClusters == 33)) {
            draw = select == tMARKED;
          }
        } else if ((mCfgH.markAdjacentClusters & 2) && (attach & gputpcgmmergertypes::attachTube)) {
          draw = select == tMARKED;
        } else if ((mCfgH.markAdjacentClusters & 1) && (attach & (gputpcgmmergertypes::attachGood | gputpcgmmergertypes::attachTube)) == 0) {
          draw = select == tMARKED;
        } else if ((mCfgH.markAdjacentClusters & 4) && (attach & gputpcgmmergertypes::attachGoodLeg) == 0) {
          draw = select == tMARKED;
        } else if ((mCfgH.markAdjacentClusters & 16) && (attach & gputpcgmmergertypes::attachHighIncl)) {
          draw = select == tMARKED;
        } else if (mCfgH.markAdjacentClusters & 8) {
          if (fabsf(mIOPtrs->mergedTracks[attach & gputpcgmmergertypes::attachTrackMask].GetParam().GetQPt()) > 20.f) {
            draw = select == tMARKED;
          }
        }
      }
    } else if (mCfgH.markClusters) {
      short flags;
      if (mParam->par.earlyTpcTransform) {
        flags = mIOPtrs->clusterData[iSlice][cidInSlice].flags;
      } else {
        flags = mIOPtrs->clustersNative->clustersLinear[cid].getFlags();
      }
      const bool match = flags & mCfgH.markClusters;
      draw = (select == tMARKED) ? (match) : (draw && !match);
    } else if (mCfgH.markFakeClusters) {
      const bool fake = (mQA->HitAttachStatus(cid));
      draw = (select == tMARKED) ? (fake) : (draw && !fake);
    }
    if (draw) {
      mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y, mCfgH.projectXY ? 0 : mGlobalPos[cid].z);
    }
  }
  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawLinks(const GPUTPCTracker& tracker, int id, bool dodown)
{
  int iSlice = tracker.ISlice();
  if (mCfgH.clustersOnly) {
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
          const int cid1 = GET_CID(iSlice, tracker.Data().ClusterDataIndex(row, j));
          const int cid2 = GET_CID(iSlice, tracker.Data().ClusterDataIndex(rowUp, tracker.Data().HitLinkUpData(row, j)));
          drawPointLinestrip(iSlice, cid1, id);
          drawPointLinestrip(iSlice, cid2, id);
        }
      }
    }

    if (dodown && i >= 2) {
      const GPUTPCRow& rowDown = tracker.Data().Row(i - 2);
      for (int j = 0; j < row.NHits(); j++) {
        if (tracker.Data().HitLinkDownData(row, j) != CALINK_INVAL) {
          const int cid1 = GET_CID(iSlice, tracker.Data().ClusterDataIndex(row, j));
          const int cid2 = GET_CID(iSlice, tracker.Data().ClusterDataIndex(rowDown, tracker.Data().HitLinkDownData(row, j)));
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
  if (mCfgH.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = 0; i < *tracker.NStartHits(); i++) {
    const GPUTPCHitId& hit = tracker.TrackletStartHit(i);
    size_t startCountInner = mVertexBuffer[iSlice].size();
    int ir = hit.RowIndex();
    calink ih = hit.HitIndex();
    do {
      const GPUTPCRow& row = tracker.Data().Row(ir);
      const int cid = GET_CID(iSlice, tracker.Data().ClusterDataIndex(row, ih));
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
  if (mCfgH.clustersOnly) {
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
      const calink rowHit = tracker.TrackletRowHits()[tracklet.FirstHit() + (j - tracklet.FirstRow())];
      if (rowHit != CALINK_INVAL) {
        const GPUTPCRow& row = tracker.Data().Row(j);
        const int cid = GET_CID(iSlice, tracker.Data().ClusterDataIndex(row, rowHit));
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
  if (mCfgH.clustersOnly) {
    return (vboList(0, 0, iSlice));
  }
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = (global ? tracker.CommonMemory()->nLocalTracks : 0); i < (global ? *tracker.NTracks() : tracker.CommonMemory()->nLocalTracks); i++) {
    GPUTPCTrack& track = tracker.Tracks()[i];
    size_t startCountInner = mVertexBuffer[iSlice].size();
    for (int j = 0; j < track.NHits(); j++) {
      const GPUTPCHitId& hit = tracker.TrackHits()[track.FirstHitID() + j];
      const GPUTPCRow& row = tracker.Data().Row(hit.RowIndex());
      const int cid = GET_CID(iSlice, tracker.Data().ClusterDataIndex(row, hit.HitIndex()));
      drawPointLinestrip(iSlice, cid, tSLICETRACK + global);
    }
    insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  }
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

void GPUDisplay::DrawTrackITS(int trackId, int iSlice)
{
  const auto& trk = mIOPtrs->itsTracks[trackId];
  unsigned int rof;
  for (rof = 1; rof < mIOPtrs->nItsTrackROF; rof++) {
    if (mIOPtrs->itsTrackROF[rof].getFirstEntry() > trackId) {
      break;
    }
  }
  rof--;
  int clusIndOffs = mIOPtrs->itsClusterROF[rof].getFirstEntry();
  for (int k = 0; k < trk.getNClusters(); k++) {
    int cid = clusIndOffs + mIOPtrs->itsTrackClusIdx[trk.getFirstClusterEntry() + k];
    mVertexBuffer[iSlice].emplace_back(mGlobalPosITS[cid].x, mGlobalPosITS[cid].y, mCfgH.projectXY ? 0 : mGlobalPosITS[cid].z);
    mGlobalPosITS[cid].w = tITSATTACHED;
  }
}

GPUDisplay::vboList GPUDisplay::DrawFinalITS()
{
  const int iSlice = 0;
  size_t startCount = mVertexBufferStart[iSlice].size();
  for (unsigned int i = 0; i < mIOPtrs->nItsTracks; i++) {
    if (mITSStandaloneTracks[i]) {
      size_t startCountInner = mVertexBuffer[iSlice].size();
      DrawTrackITS(i, iSlice);
      insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
    }
  }
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

template <class T>
void GPUDisplay::DrawFinal(int iSlice, int /*iCol*/, GPUTPCGMPropagator* prop, std::array<vecpod<int>, 2>& trackList, threadVertexBuffer& threadBuffer)
{
  auto& vBuf = threadBuffer.vBuf;
  auto& buffer = threadBuffer.buffer;
  unsigned int nTracks = std::max(trackList[0].size(), trackList[1].size());
  if (mCfgH.clustersOnly) {
    nTracks = 0;
  }
  for (unsigned int ii = 0; ii < nTracks; ii++) {
    int i = 0;
    const T* track = nullptr;
    int lastCluster = -1;
    while (true) {
      if (ii >= trackList[0].size()) {
        break;
      }
      i = trackList[0][ii];
      int nClusters;
      if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
        track = &mIOPtrs->mergedTracks[i];
        nClusters = track->NClusters();
      } else if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        track = &mIOPtrs->outputTracksTPCO2[i];
        nClusters = track->getNClusters();
        if (!mIOPtrs->clustersNative) {
          break;
        }
      } else {
        throw std::runtime_error("invalid type");
      }

      size_t startCountInner = mVertexBuffer[iSlice].size();
      bool drawing = false;

      if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (!mCfgH.drawTracksAndFilter && !(mCfgH.drawTPCTracks || (mCfgH.drawITSTracks && mIOPtrs->tpcLinkITS && mIOPtrs->tpcLinkITS[i] != -1) || (mCfgH.drawTRDTracks && mIOPtrs->tpcLinkTRD && mIOPtrs->tpcLinkTRD[i] != -1) || (mCfgH.drawTOFTracks && mIOPtrs->tpcLinkTOF && mIOPtrs->tpcLinkTOF[i] != -1))) {
          break;
        }
        if (mCfgH.drawTracksAndFilter && ((mCfgH.drawITSTracks && !(mIOPtrs->tpcLinkITS && mIOPtrs->tpcLinkITS[i] != -1)) || (mCfgH.drawTRDTracks && !(mIOPtrs->tpcLinkTRD && mIOPtrs->tpcLinkTRD[i] != -1)) || (mCfgH.drawTOFTracks && !(mIOPtrs->tpcLinkTOF && mIOPtrs->tpcLinkTOF[i] != -1)))) {
          break;
        }
      }

      // Print TOF part of track
      if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (mIOPtrs->tpcLinkTOF && mIOPtrs->tpcLinkTOF[i] != -1 && mIOPtrs->nTOFClusters) {
          int cid = mIOPtrs->tpcLinkTOF[i];
          drawing = true;
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTOF[cid].x, mGlobalPosTOF[cid].y, mCfgH.projectXY ? 0 : mGlobalPosTOF[cid].z);
          mGlobalPosTOF[cid].w = tTOFATTACHED;
        }
      }

      // Print TRD part of track
      if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
        if (mCfgH.trackFilter && mChain) {
          if (mCfgH.trackFilter == 2 && (!trdTracker().PreCheckTrackTRDCandidate(*track) || !trdTracker().CheckTrackTRDCandidate((GPUTRDTrackGPU)*track))) {
            break;
          }
          if (mCfgH.trackFilter == 1 && mTRDTrackIds[i] == -1) {
            break;
          }
        }
        if (mTRDTrackIds[i] != -1 && mIOPtrs->nTRDTracklets) {
          auto& trk = mIOPtrs->trdTracks[mTRDTrackIds[i]];
          for (int k = 5; k >= 0; k--) {
            int cid = trk.getTrackletIndex(k);
            if (cid < 0) {
              continue;
            }
            drawing = true;
            mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[cid].x, mGlobalPosTRD2[cid].y, mCfgH.projectXY ? 0 : mGlobalPosTRD2[cid].z);
            mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[cid].x, mGlobalPosTRD[cid].y, mCfgH.projectXY ? 0 : mGlobalPosTRD[cid].z);
            mGlobalPosTRD[cid].w = tTRDATTACHED;
          }
        }
      } else if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (mIOPtrs->tpcLinkTRD && mIOPtrs->tpcLinkTRD[i] != -1 && mIOPtrs->nTRDTracklets) {
          if ((mIOPtrs->tpcLinkTRD[i] & 0x40000000) ? mIOPtrs->nTRDTracksITSTPCTRD : mIOPtrs->nTRDTracksTPCTRD) {
            const auto* container = (mIOPtrs->tpcLinkTRD[i] & 0x40000000) ? mIOPtrs->trdTracksITSTPCTRD : mIOPtrs->trdTracksTPCTRD;
            const auto& trk = container[mIOPtrs->tpcLinkTRD[i] & 0x3FFFFFFF];
            for (int k = 5; k >= 0; k--) {
              int cid = trk.getTrackletIndex(k);
              if (cid < 0) {
                continue;
              }
              drawing = true;
              mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[cid].x, mGlobalPosTRD2[cid].y, mCfgH.projectXY ? 0 : mGlobalPosTRD2[cid].z);
              mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[cid].x, mGlobalPosTRD[cid].y, mCfgH.projectXY ? 0 : mGlobalPosTRD[cid].z);
              mGlobalPosTRD[cid].w = tTRDATTACHED;
            }
          }
        }
      }

      // Print TPC part of track
      for (int k = 0; k < nClusters; k++) {
        if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
          if (mCfgH.hideRejectedClusters && (mIOPtrs->mergedTrackHits[track->FirstClusterRef() + k].state & GPUTPCGMMergedTrackHit::flagReject)) {
            continue;
          }
        }
        int cid;
        if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
          cid = mIOPtrs->mergedTrackHits[track->FirstClusterRef() + k].num;
        } else {
          cid = &track->getCluster(mIOPtrs->outputClusRefsTPCO2, k, *mIOPtrs->clustersNative) - mIOPtrs->clustersNative->clustersLinear;
        }
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
            if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
              cid = mIOPtrs->mergedTrackHits[track->FirstClusterRef() + lastCluster].num;
            } else {
              cid = &track->getCluster(mIOPtrs->outputClusRefsTPCO2, lastCluster, *mIOPtrs->clustersNative) - mIOPtrs->clustersNative->clustersLinear;
            }
            drawPointLinestrip(iSlice, cid, 7, SEPERATE_GLOBAL_TRACKS_LIMIT);
          }
          drawing = true;
        }
        lastCluster = k;
      }

      // Print ITS part of track
      if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (mIOPtrs->tpcLinkITS && mIOPtrs->tpcLinkITS[i] != -1 && mIOPtrs->nItsTracks && mIOPtrs->nItsClusters) {
          DrawTrackITS(mIOPtrs->tpcLinkITS[i], iSlice);
        }
      }
      insertVertexList(vBuf[0], startCountInner, mVertexBuffer[iSlice].size());
      break;
    }

    if (!mIOPtrs->clustersNative) {
      continue;
    }

    // Propagate track paramters / plot MC tracks
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
        float alphaOrg = 0;
        if (iMC == 0) {
          if (!inFlyDirection && mIOPtrs->tpcLinkITS && mIOPtrs->tpcLinkITS[i] != -1) {
            continue;
          }
          if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
            trkParam.Set(track->GetParam());
            alphaOrg = mParam->Alpha(iSlice);
          } else {
            GPUTPCGMTrackParam t;
            convertTrackParam(t, *track);
            alphaOrg = track->getAlpha();
            trkParam.Set(t);
          }

          if (mParam->par.earlyTpcTransform) {
            if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
              x = mIOPtrs->mergedTrackHitsXYZ[track->FirstClusterRef() + lastCluster].x;
              ZOffset = track->GetParam().GetTZOffset();
            }
          } else {
            float y, z;
            if constexpr (std::is_same_v<T, GPUTPCGMMergedTrack>) {
              auto cl = mIOPtrs->mergedTrackHits[track->FirstClusterRef() + lastCluster];
              const auto& cln = mIOPtrs->clustersNative->clustersLinear[cl.num];
              GPUTPCConvertImpl::convert(*mCalib->fastTransform, *mParam, cl.slice, cl.row, cln.getPad(), cln.getTime(), x, y, z);
              ZOffset = mCalib->fastTransform->convVertexTimeToZOffset(iSlice, track->GetParam().GetTZOffset(), mParam->par.continuousMaxTimeBin);
            } else {
              uint8_t sector, row;
              auto cln = track->getCluster(mIOPtrs->outputClusRefsTPCO2, lastCluster, *mIOPtrs->clustersNative, sector, row);
              GPUTPCConvertImpl::convert(*mCalib->fastTransform, *mParam, sector, row, cln.getPad(), cln.getTime(), x, y, z);
              ZOffset = mCalib->fastTransform->convVertexTimeToZOffset(sector, track->getTime0(), mParam->par.continuousMaxTimeBin);
            }
          }
        } else {
          const GPUTPCMCInfo& mc = mIOPtrs->mcInfosTPC[i];
          if (mc.charge == 0.f) {
            break;
          }
          if (mc.pid < 0) {
            break;
          }

          alphaOrg = mParam->Alpha(iSlice);
          float c = cosf(alphaOrg);
          float s = sinf(alphaOrg);
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
#ifdef GPUCA_TPC_GEOMETRY_O2
          trkParam.Set(mclocal[0], mclocal[1], mc.z, mclocal[2], mclocal[3], mc.pZ, charge);
          if (mParam->par.continuousTracking) {
            ZOffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, mc.t0, mParam->par.continuousMaxTimeBin)) * (mc.z < 0 ? -1 : 1);
          }
#else
          if (fabsf(mc.z) > 250) {
            ZOffset = mc.z > 0 ? (mc.z - 250) : (mc.z + 250);
          }
          trkParam.Set(mclocal[0], mclocal[1], mc.z - ZOffset, mclocal[2], mclocal[3], mc.pZ, charge);
#endif
        }
        trkParam.X() += mCfgH.xAdd;
        x += mCfgH.xAdd;
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
        float alpha = alphaOrg;
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
            if (!mCfgH.propagateLoopers) {
              float diff = fabsf(alpha - alphaOrg) / (2. * CAMath::Pi());
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
          useBuffer.emplace_back((ca * trkParam.X() - sa * trkParam.Y()) / GL_SCALE_FACTOR, (ca * trkParam.Y() + sa * trkParam.X()) / GL_SCALE_FACTOR, mCfgH.projectXY ? 0 : (trkParam.Z() + ZOffset) / GL_SCALE_FACTOR);
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
      float x = row.X() + mCfgH.xAdd;
      float y = row.Grid().YMin() + (float)j / row.Grid().StepYInv();
      float zz1, zz2, yy1, yy2, xx1, xx2;
      mParam->Slice2Global(tracker.ISlice(), x, y, z1, &xx1, &yy1, &zz1);
      mParam->Slice2Global(tracker.ISlice(), x, y, z2, &xx2, &yy2, &zz2);
      if (iSlice < 18) {
        zz1 += mCfgH.zAdd;
        zz2 += mCfgH.zAdd;
      } else {
        zz1 -= mCfgH.zAdd;
        zz2 -= mCfgH.zAdd;
      }
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
    }
    for (int j = 0; j <= (signed)row.Grid().Nz(); j++) {
      float y1 = row.Grid().YMin();
      float y2 = row.Grid().YMax();
      float x = row.X() + mCfgH.xAdd;
      float z = row.Grid().ZMin() + (float)j / row.Grid().StepZInv();
      float zz1, zz2, yy1, yy2, xx1, xx2;
      mParam->Slice2Global(tracker.ISlice(), x, y1, z, &xx1, &yy1, &zz1);
      mParam->Slice2Global(tracker.ISlice(), x, y2, z, &xx2, &yy2, &zz2);
      if (iSlice < 18) {
        zz1 += mCfgH.zAdd;
        zz2 += mCfgH.zAdd;
      } else {
        zz1 -= mCfgH.zAdd;
        zz2 -= mCfgH.zAdd;
      }
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
    }
  }
  insertVertexList(tracker.ISlice(), startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawGridTRD(int sector)
{
  // TODO: tilted pads ignored at the moment
  size_t startCount = mVertexBufferStart[sector].size();
  size_t startCountInner = mVertexBuffer[sector].size();
#ifdef GPUCA_HAVE_O2HEADERS
  auto* geo = &trdGeometry();
  if (geo) {
    int trdsector = NSLICES / 2 - 1 - sector;
    float alpha = geo->GetAlpha() / 2.f + geo->GetAlpha() * trdsector;
    if (trdsector >= 9) {
      alpha -= 2 * CAMath::Pi();
    }
    for (int iLy = 0; iLy < GPUTRDTracker::EGPUTRDTracker::kNLayers; ++iLy) {
      for (int iStack = 0; iStack < GPUTRDTracker::EGPUTRDTracker::kNStacks; ++iStack) {
        int iDet = geo->GetDetector(iLy, iStack, trdsector);
        auto matrix = geo->GetClusterMatrix(iDet);
        if (!matrix) {
          continue;
        }
        auto pp = geo->GetPadPlane(iDet);
        for (int i = 0; i < pp->GetNrows(); ++i) {
          float xyzLoc1[3];
          float xyzLoc2[3];
          float xyzGlb1[3];
          float xyzGlb2[3];
          xyzLoc1[0] = xyzLoc2[0] = geo->AnodePos();
          xyzLoc1[1] = pp->GetCol0();
          xyzLoc2[1] = pp->GetColEnd();
          xyzLoc1[2] = xyzLoc2[2] = pp->GetRowPos(i) - pp->GetRowPos(pp->GetNrows() / 2);
          matrix->LocalToMaster(xyzLoc1, xyzGlb1);
          matrix->LocalToMaster(xyzLoc2, xyzGlb2);
          float x1Tmp = xyzGlb1[0];
          xyzGlb1[0] = xyzGlb1[0] * cosf(alpha) + xyzGlb1[1] * sinf(alpha);
          xyzGlb1[1] = -x1Tmp * sinf(alpha) + xyzGlb1[1] * cosf(alpha);
          float x2Tmp = xyzGlb2[0];
          xyzGlb2[0] = xyzGlb2[0] * cosf(alpha) + xyzGlb2[1] * sinf(alpha);
          xyzGlb2[1] = -x2Tmp * sinf(alpha) + xyzGlb2[1] * cosf(alpha);
          mVertexBuffer[sector].emplace_back(xyzGlb1[0] / GL_SCALE_FACTOR, xyzGlb1[1] / GL_SCALE_FACTOR, xyzGlb1[2] / GL_SCALE_FACTOR);
          mVertexBuffer[sector].emplace_back(xyzGlb2[0] / GL_SCALE_FACTOR, xyzGlb2[1] / GL_SCALE_FACTOR, xyzGlb2[2] / GL_SCALE_FACTOR);
        }
        for (int j = 0; j < pp->GetNcols(); ++j) {
          float xyzLoc1[3];
          float xyzLoc2[3];
          float xyzGlb1[3];
          float xyzGlb2[3];
          xyzLoc1[0] = xyzLoc2[0] = geo->AnodePos();
          xyzLoc1[1] = xyzLoc2[1] = pp->GetColPos(j) + pp->GetColSize(j) / 2.f;
          xyzLoc1[2] = pp->GetRow0() - pp->GetRowPos(pp->GetNrows() / 2);
          xyzLoc2[2] = pp->GetRowEnd() - pp->GetRowPos(pp->GetNrows() / 2);
          matrix->LocalToMaster(xyzLoc1, xyzGlb1);
          matrix->LocalToMaster(xyzLoc2, xyzGlb2);
          float x1Tmp = xyzGlb1[0];
          xyzGlb1[0] = xyzGlb1[0] * cosf(alpha) + xyzGlb1[1] * sinf(alpha);
          xyzGlb1[1] = -x1Tmp * sinf(alpha) + xyzGlb1[1] * cosf(alpha);
          float x2Tmp = xyzGlb2[0];
          xyzGlb2[0] = xyzGlb2[0] * cosf(alpha) + xyzGlb2[1] * sinf(alpha);
          xyzGlb2[1] = -x2Tmp * sinf(alpha) + xyzGlb2[1] * cosf(alpha);
          mVertexBuffer[sector].emplace_back(xyzGlb1[0] / GL_SCALE_FACTOR, xyzGlb1[1] / GL_SCALE_FACTOR, xyzGlb1[2] / GL_SCALE_FACTOR);
          mVertexBuffer[sector].emplace_back(xyzGlb2[0] / GL_SCALE_FACTOR, xyzGlb2[1] / GL_SCALE_FACTOR, xyzGlb2[2] / GL_SCALE_FACTOR);
        }
      }
    }
  }
#endif
  insertVertexList(sector, startCountInner, mVertexBuffer[sector].size());
  return (vboList(startCount, mVertexBufferStart[sector].size() - startCount, sector));
}

int GPUDisplay::DrawGLScene(bool mixAnimation, float mAnimateTime)
{
  if (mChain) {
    mIOPtrs = &mChain->mIOPtrs;
    mCalib = &mChain->calib();
  }
  try {
    if (DrawGLScene_internal(mixAnimation, mAnimateTime)) {
      return (1);
    }
  } catch (const std::runtime_error& e) {
    return (1);
  }
  return (0);
}

int GPUDisplay::DrawGLScene_internal(bool mixAnimation, float mAnimateTime)
{
  bool showTimer = false;

  // Make sure event gets not overwritten during display
  if (mAnimateTime < 0) {
    mSemLockDisplay.Lock();
  }

  if (!mIOPtrs) {
    mNCollissions = 0;
  } else if (!mCollisionClusters.size()) {
    mNCollissions = std::max(1u, mIOPtrs->nMCInfosTPCCol);
  }

  if (!mixAnimation && (mUpdateDLList || mResetScene || !mGlDLrecent) && mIOPtrs) {
    disableUnsupportedOptions();
  }

  // Extract global cluster information
  if (!mixAnimation && (mUpdateDLList || mResetScene) && mIOPtrs) {
    showTimer = true;
    mTimerDraw.ResetStart();
    if (mIOPtrs->clustersNative) {
      mCurrentClusters = mIOPtrs->clustersNative->nClustersTotal;
    } else {
      mCurrentClusters = 0;
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        mCurrentClusters += mIOPtrs->nClusterData[iSlice];
      }
    }
    if (mNMaxClusters < mCurrentClusters) {
      mNMaxClusters = mCurrentClusters;
      mGlobalPosPtr.reset(new float4[mNMaxClusters]);
      mGlobalPos = mGlobalPosPtr.get();
    }

    mCurrentSpacePointsTRD = mIOPtrs->nTRDTracklets;
    if (mCurrentSpacePointsTRD > mNMaxSpacePointsTRD) {
      mNMaxSpacePointsTRD = mCurrentSpacePointsTRD;
      mGlobalPosPtrTRD.reset(new float4[mNMaxSpacePointsTRD]);
      mGlobalPosPtrTRD2.reset(new float4[mNMaxSpacePointsTRD]);
      mGlobalPosTRD = mGlobalPosPtrTRD.get();
      mGlobalPosTRD2 = mGlobalPosPtrTRD2.get();
    }

    mCurrentClustersITS = mIOPtrs->itsClusters ? mIOPtrs->nItsClusters : 0;
    if (mNMaxClustersITS < mCurrentClustersITS) {
      mNMaxClustersITS = mCurrentClustersITS;
      mGlobalPosPtrITS.reset(new float4[mNMaxClustersITS]);
      mGlobalPosITS = mGlobalPosPtrITS.get();
    }

    mCurrentClustersTOF = mIOPtrs->nTOFClusters;
    if (mNMaxClustersTOF < mCurrentClustersTOF) {
      mNMaxClustersTOF = mCurrentClustersTOF;
      mGlobalPosPtrTOF.reset(new float4[mNMaxClustersTOF]);
      mGlobalPosTOF = mGlobalPosPtrTOF.get();
    }

    unsigned int nTpcMergedTracks = mConfig.showTPCTracksFromO2Format ? mIOPtrs->nOutputTracksTPCO2 : mIOPtrs->nMergedTracks;
    if ((size_t)nTpcMergedTracks > mTRDTrackIds.size()) {
      mTRDTrackIds.resize(nTpcMergedTracks);
    }
    if (mIOPtrs->nItsTracks > mITSStandaloneTracks.size()) {
      mITSStandaloneTracks.resize(mIOPtrs->nItsTracks);
    }
    for (unsigned int i = 0; i < nTpcMergedTracks; i++) {
      mTRDTrackIds[i] = -1;
    }
    for (unsigned int i = 0; i < mIOPtrs->nTRDTracks; i++) {
      if (mIOPtrs->trdTracks[i].getNtracklets()) {
        mTRDTrackIds[mIOPtrs->trdTracks[i].getRefGlobalTrackIdRaw()] = i;
      }
    }
    if (mIOPtrs->nItsTracks) {
      std::fill(mITSStandaloneTracks.begin(), mITSStandaloneTracks.end(), true);
      if (mIOPtrs->tpcLinkITS) {
        for (unsigned int i = 0; i < nTpcMergedTracks; i++) {
          if (mIOPtrs->tpcLinkITS[i] != -1) {
            mITSStandaloneTracks[mIOPtrs->tpcLinkITS[i]] = false;
          }
        }
      }
    }

    mMaxClusterZ = 0;
    GPUCA_OPENMP(parallel for num_threads(getNumThreads()) reduction(max : mMaxClusterZ))
    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      int row = 0;
      unsigned int nCls = mParam->par.earlyTpcTransform ? mIOPtrs->nClusterData[iSlice] : mIOPtrs->clustersNative ? mIOPtrs->clustersNative->nClustersSector[iSlice] : 0;
      for (unsigned int i = 0; i < nCls; i++) {
        int cid;
        if (mParam->par.earlyTpcTransform) {
          const auto& cl = mIOPtrs->clusterData[iSlice][i];
          cid = cl.id;
          row = cl.row;
        } else {
          cid = mIOPtrs->clustersNative->clusterOffset[iSlice][0] + i;
          while (row < GPUCA_ROW_COUNT && mIOPtrs->clustersNative->clusterOffset[iSlice][row + 1] <= (unsigned int)cid) {
            row++;
          }
        }
        if (cid >= mNMaxClusters) {
          throw std::runtime_error("Cluster Buffer Size exceeded");
        }
        float4* ptr = &mGlobalPos[cid];
        if (mParam->par.earlyTpcTransform) {
          const auto& cl = mIOPtrs->clusterData[iSlice][i];
          mParam->Slice2Global(iSlice, (mCfgH.clustersOnNominalRow ? mParam->tpcGeometry.Row2X(row) : cl.x) + mCfgH.xAdd, cl.y, cl.z, &ptr->x, &ptr->y, &ptr->z);
        } else {
          float x, y, z;
          const auto& cln = mIOPtrs->clustersNative->clusters[iSlice][0][i];
          GPUTPCConvertImpl::convert(*mCalib->fastTransform, *mParam, iSlice, row, cln.getPad(), cln.getTime(), x, y, z);
          if (mCfgH.clustersOnNominalRow) {
            x = mParam->tpcGeometry.Row2X(row);
          }
          mParam->Slice2Global(iSlice, x + mCfgH.xAdd, y, z, &ptr->x, &ptr->y, &ptr->z);
        }

        if (fabsf(ptr->z) > mMaxClusterZ) {
          mMaxClusterZ = fabsf(ptr->z);
        }
        ptr->z += iSlice < 18 ? mCfgH.zAdd : -mCfgH.zAdd;
        ptr->x /= GL_SCALE_FACTOR;
        ptr->y /= GL_SCALE_FACTOR;
        ptr->z /= GL_SCALE_FACTOR;
        ptr->w = tCLUSTER;
      }
    }

    int trdTriggerRecord = -1;
    float trdZoffset = 0;
    GPUCA_OPENMP(parallel for num_threads(getNumThreads()) reduction(max : mMaxClusterZ) firstprivate(trdTriggerRecord, trdZoffset))
    for (int i = 0; i < mCurrentSpacePointsTRD; i++) {
      while (mParam->par.continuousTracking && trdTriggerRecord < (int)mIOPtrs->nTRDTriggerRecords - 1 && mIOPtrs->trdTrackletIdxFirst[trdTriggerRecord + 1] <= i) {
        trdTriggerRecord++;
        float trdTime = mIOPtrs->trdTriggerTimes[trdTriggerRecord] * 1e3 / o2::constants::lhc::LHCBunchSpacingNS / o2::tpc::constants::LHCBCPERTIMEBIN;
        trdZoffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, trdTime, mParam->par.continuousMaxTimeBin));
      }
      const auto& sp = mIOPtrs->trdSpacePoints[i];
      int iSec = trdGeometry().GetSector(mIOPtrs->trdTracklets[i].GetDetector());
      float4* ptr = &mGlobalPosTRD[i];
      mParam->Slice2Global(iSec, sp.getX() + mCfgH.xAdd, sp.getY(), sp.getZ(), &ptr->x, &ptr->y, &ptr->z);
      ptr->z += ptr->z > 0 ? trdZoffset : -trdZoffset;
      if (fabsf(ptr->z) > mMaxClusterZ) {
        mMaxClusterZ = fabsf(ptr->z);
      }
      ptr->x /= GL_SCALE_FACTOR;
      ptr->y /= GL_SCALE_FACTOR;
      ptr->z /= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
      ptr = &mGlobalPosTRD2[i];
      mParam->Slice2Global(iSec, sp.getX() + mCfgH.xAdd + 4.5f, sp.getY() + 1.5f * sp.getDy(), sp.getZ(), &ptr->x, &ptr->y, &ptr->z);
      ptr->z += ptr->z > 0 ? trdZoffset : -trdZoffset;
      if (fabsf(ptr->z) > mMaxClusterZ) {
        mMaxClusterZ = fabsf(ptr->z);
      }
      ptr->x /= GL_SCALE_FACTOR;
      ptr->y /= GL_SCALE_FACTOR;
      ptr->z /= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
    }

    GPUCA_OPENMP(parallel for num_threads(getNumThreads()) reduction(max : mMaxClusterZ))
    for (int i = 0; i < mCurrentClustersTOF; i++) {
#ifdef GPUCA_HAVE_O2HEADERS
      float4* ptr = &mGlobalPosTOF[i];
      mParam->Slice2Global(mIOPtrs->tofClusters[i].getSector(), mIOPtrs->tofClusters[i].getX() + mCfgH.xAdd, mIOPtrs->tofClusters[i].getY(), mIOPtrs->tofClusters[i].getZ(), &ptr->x, &ptr->y, &ptr->z);
      float ZOffset = 0;
      if (mParam->par.continuousTracking) {
        float tofTime = mIOPtrs->tofClusters[i].getTime() * 1e-3 / o2::constants::lhc::LHCBunchSpacingNS / o2::tpc::constants::LHCBCPERTIMEBIN;
        ZOffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, tofTime, mParam->par.continuousMaxTimeBin));
        ptr->z += ptr->z > 0 ? ZOffset : -ZOffset;
      }
      if (fabsf(ptr->z) > mMaxClusterZ) {
        mMaxClusterZ = fabsf(ptr->z);
      }
      ptr->x /= GL_SCALE_FACTOR;
      ptr->y /= GL_SCALE_FACTOR;
      ptr->z /= GL_SCALE_FACTOR;
      ptr->w = tTOFCLUSTER;
#endif
    }

    if (mCurrentClustersITS) {
#ifdef GPUCA_HAVE_O2HEADERS
      float itsROFhalfLen = 0;
#ifdef GPUCA_O2_LIB // Not available in standalone benchmark
      if (mParam->par.continuousTracking) {
        const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
        itsROFhalfLen = alpParams.roFrameLengthInBC / (float)o2::tpc::constants::LHCBCPERTIMEBIN / 2;
      }
#endif
      int i = 0;
      for (unsigned int j = 0; j < mIOPtrs->nItsClusterROF; j++) {
        float ZOffset = 0;
        if (mParam->par.continuousTracking) {
          o2::InteractionRecord startIR = o2::InteractionRecord(0, mIOPtrs->settingsTF && mIOPtrs->settingsTF->hasTfStartOrbit ? mIOPtrs->settingsTF->tfStartOrbit : 0);
          float itsROFtime = mIOPtrs->itsClusterROF[j].getBCData().differenceInBC(startIR) / (float)o2::tpc::constants::LHCBCPERTIMEBIN;
          ZOffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, itsROFtime + itsROFhalfLen, mParam->par.continuousMaxTimeBin));
        }
        if (i != mIOPtrs->itsClusterROF[j].getFirstEntry()) {
          throw std::runtime_error("Inconsistent ITS data, number of clusters does not match ROF content");
        }
        for (int k = 0; k < mIOPtrs->itsClusterROF[j].getNEntries(); k++) {
          float4* ptr = &mGlobalPosITS[i];
          const auto& cl = mIOPtrs->itsClusters[i];
          auto* itsGeo = o2::its::GeometryTGeo::Instance();
          auto p = cl.getXYZGlo(*itsGeo);
          ptr->x = p.X();
          ptr->y = p.Y();
          ptr->z = p.Z();
          ptr->z += ptr->z > 0 ? ZOffset : -ZOffset;
          if (fabsf(ptr->z) > mMaxClusterZ) {
            mMaxClusterZ = fabsf(ptr->z);
          }
          ptr->x /= GL_SCALE_FACTOR;
          ptr->y /= GL_SCALE_FACTOR;
          ptr->z /= GL_SCALE_FACTOR;
          ptr->w = tITSCLUSTER;
          i++;
        }
      }
#endif
    }

    mTimerFPS.ResetStart();
    mFramesDoneFPS = 0;
    mFPSScaleadjust = 0;
    mGlDLrecent = 0;
    mUpdateDLList = 0;
  }

  if (!mixAnimation && mOffscreenBuffer.created) {
    setFrameBuffer(1, mOffscreenBuffer.fb_id);
  }
  // Initialize
  if (!mixAnimation) {
    if (mCfgL.invertColors) {
      CHKERR(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
    } else {
      CHKERR(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
  }

  hmm_mat4 nextViewMatrix = MY_HMM_IDENTITY;
  int mMouseWheelTmp = mBackend->mMouseWheel;
  mBackend->mMouseWheel = 0;
  bool lookOrigin = mCfgR.camLookOrigin ^ mBackend->mKeys[mBackend->KEY_ALT];
  bool yUp = mCfgR.camYUp ^ mBackend->mKeys[mBackend->KEY_CTRL] ^ lookOrigin;
  bool rotateModel = mBackend->mKeys[mBackend->KEY_RCTRL] || mBackend->mKeys[mBackend->KEY_RALT];
  bool rotateModelTPC = mBackend->mKeys[mBackend->KEY_RALT];

  // Calculate rotation / translation scaling factors
  float scalefactor = mBackend->mKeys[mBackend->KEY_SHIFT] ? 0.2 : 1.0;
  float rotatescalefactor = scalefactor * 0.25f;
  if (mCfgL.drawSlice != -1) {
    scalefactor *= 0.2f;
  }
  float sqrdist = sqrtf(sqrtf(mViewMatrixP[12] * mViewMatrixP[12] + mViewMatrixP[13] * mViewMatrixP[13] + mViewMatrixP[14] * mViewMatrixP[14]) / GL_SCALE_FACTOR) * 0.8;
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
        mCfgL = mAnimateConfig[mAnimationLastBase];
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
        mixSlaveImage = 1.f - (time - mAnimateVectors[0][mAnimationLastBase]) / (mAnimateVectors[0][base] - mAnimateVectors[0][mAnimationLastBase]);
      }

      if (memcmp(&mAnimateConfig[base], &mCfgL, sizeof(mCfgL))) {
        mCfgL = mAnimateConfig[base];
        updateConfig();
      }
    }

    if (mCfgL.animationMode != 6) {
      if (mCfgL.animationMode & 1) // Rotation from euler angles
      {
        nextViewMatrix = nextViewMatrix * HMM_Rotate(-vals[4] * 180.f / CAMath::Pi(), {1, 0, 0}) * HMM_Rotate(vals[5] * 180.f / CAMath::Pi(), {0, 1, 0}) * HMM_Rotate(-vals[6] * 180.f / CAMath::Pi(), {0, 0, 1});
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
        float mat[16] = {1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), 0, 2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), 0, 2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), 0, 0, 0, 0, 1};
        nextViewMatrix = nextViewMatrix * MY_HMM_FROM(mat);
      }
    }
    if (mCfgL.animationMode & 4) // Compute cartesian translation from sperical coordinates (euler angles)
    {
      const float r = vals[3], phi = vals[1], theta = vals[2];
      vals[2] = r * cosf(phi) * cosf(theta);
      vals[0] = r * sinf(phi) * cosf(theta);
      vals[1] = r * sinf(theta);
    } else if (mCfgL.animationMode & 2) { // Scale cartesion translation to interpolated radius
      float r = sqrtf(vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2]);
      if (fabsf(r) < 0.0001) {
        r = 1;
      }
      r = vals[3] / r;
      for (int i = 0; i < 3; i++) {
        vals[i] *= r;
      }
    }
    if (mCfgL.animationMode == 6) {
      nextViewMatrix = HMM_LookAt({vals[0], vals[1], vals[2]}, {0, 0, 0}, {0, 1, 0});
    } else {
      nextViewMatrix = nextViewMatrix * HMM_Translate({-vals[0], -vals[1], -vals[2]});
    }
  } else if (mResetScene) {
    nextViewMatrix = nextViewMatrix * HMM_Translate({0, 0, mParam->par.continuousTracking ? (-mMaxClusterZ / GL_SCALE_FACTOR - 8) : -8});
    mViewMatrix = MY_HMM_IDENTITY;
    mModelMatrix = MY_HMM_IDENTITY;

    mCfgL.pointSize = 2.0;
    mCfgL.drawSlice = -1;
    mCfgH.xAdd = mCfgH.zAdd = 0;
    mCfgR.camLookOrigin = mCfgR.camYUp = false;
    mAngleRollOrigin = -1e9;
    mFOV = 45.f;

    mResetScene = 0;
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
        nextViewMatrix = nextViewMatrix * HMM_Rotate(mAngleRollOrigin, {0, 0, 1});
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
      const float max_theta = CAMath::Pi() / 2 - 0.01;
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

      if (yUp) {
        nextViewMatrix = MY_HMM_IDENTITY;
      }
      nextViewMatrix = nextViewMatrix * HMM_LookAt({mXYZ[0], mXYZ[1], mXYZ[2]}, {0, 0, 0}, {0, 1, 0});
    } else {
      nextViewMatrix = nextViewMatrix * HMM_Translate({moveX, moveY, moveZ});
      if (!rotateModel) {
        if (rotYaw != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotYaw, {0, 1, 0});
        }
        if (rotPitch != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotPitch, {1, 0, 0});
        }
        if (!yUp && rotRoll != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotRoll, {0, 0, 1});
        }
      }
      nextViewMatrix = nextViewMatrix * mViewMatrix; // Apply previous translation / rotation
      if (yUp) {
        calcXYZ(&nextViewMatrix.Elements[0][0]);
        nextViewMatrix = HMM_Rotate(mAngle[2] * 180.f / CAMath::Pi(), {0, 0, 1}) * nextViewMatrix;
      }
      if (rotateModel) {
        if (rotYaw != 0.f) {
          mModelMatrix = HMM_Rotate(rotYaw, {nextViewMatrix.Elements[0][1], nextViewMatrix.Elements[1][1], nextViewMatrix.Elements[2][1]}) * mModelMatrix;
        }
        if (rotPitch != 0.f) {
          mModelMatrix = HMM_Rotate(rotPitch, {nextViewMatrix.Elements[0][0], nextViewMatrix.Elements[1][0], nextViewMatrix.Elements[2][0]}) * mModelMatrix;
        }
        if (rotRoll != 0.f) {
          if (rotateModelTPC) {
            mModelMatrix = HMM_Rotate(-rotRoll, {0, 0, 1}) * mModelMatrix;
          } else {
            mModelMatrix = HMM_Rotate(-rotRoll, {nextViewMatrix.Elements[0][2], nextViewMatrix.Elements[1][2], nextViewMatrix.Elements[2][2]}) * mModelMatrix;
          }
        }
      }
    }

    // Graphichs Options
    float minSize = 0.4 / (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1);
    int deltaLine = mBackend->mKeys['+'] * mBackend->mKeysShift['+'] - mBackend->mKeys['-'] * mBackend->mKeysShift['-'];
    mCfgL.lineWidth += (float)deltaLine * mFPSScale * 0.02 * mCfgL.lineWidth;
    if (mCfgL.lineWidth < minSize) {
      mCfgL.lineWidth = minSize;
    }
    if (deltaLine) {
      SetInfo("%s line width: %f", deltaLine > 0 ? "Increasing" : "Decreasing", mCfgL.lineWidth);
    }
    minSize *= 2;
    int deltaPoint = mBackend->mKeys['+'] * (!mBackend->mKeysShift['+']) - mBackend->mKeys['-'] * (!mBackend->mKeysShift['-']);
    mCfgL.pointSize += (float)deltaPoint * mFPSScale * 0.02 * mCfgL.pointSize;
    if (mCfgL.pointSize < minSize) {
      mCfgL.pointSize = minSize;
    }
    if (deltaPoint) {
      SetInfo("%s point size: %f", deltaPoint > 0 ? "Increasing" : "Decreasing", mCfgL.pointSize);
    }
  }

  // Store position
  if (mAnimateTime < 0) {
    mViewMatrix = nextViewMatrix;
    calcXYZ(mViewMatrixP);
  }

  if (mBackend->mMouseDn || mBackend->mMouseDnR) {
    mBackend->mMouseDnX = mBackend->mouseMvX;
    mBackend->mMouseDnY = mBackend->mouseMvY;
  }
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (mCfgL.smoothPoints && !mCfgR.openGLCore) {
    CHKERR(glEnable(GL_POINT_SMOOTH));
  } else {
    CHKERR(glDisable(GL_POINT_SMOOTH));
  }
  if (mCfgL.smoothLines && !mCfgR.openGLCore) {
    CHKERR(glEnable(GL_LINE_SMOOTH));
  } else {
    CHKERR(glDisable(GL_LINE_SMOOTH));
  }
#endif
  CHKERR(glEnable(GL_BLEND));
  CHKERR(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  CHKERR(glPointSize(mCfgL.pointSize * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1)));
  CHKERR(glLineWidth(mCfgL.lineWidth * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1)));

  // Prepare Event
  if (!mGlDLrecent && mIOPtrs) {
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
    GPUCA_OPENMP(parallel num_threads(getNumThreads()))
    {
#ifdef WITH_OPENMP
      int numThread = omp_get_thread_num();
      int numThreads = omp_get_num_threads();
#else
      int numThread = 0, numThreads = 1;
#endif
      if (mChain && (mChain->GetRecoSteps() & GPUDataTypes::RecoStep::TPCSliceTracking)) {
        GPUCA_OPENMP(for)
        for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
          GPUTPCTracker& tracker = (GPUTPCTracker&)sliceTracker(iSlice);
          tracker.SetPointersDataLinks(tracker.LinkTmpMemory());
          mGlDLLines[iSlice][tINITLINK] = DrawLinks(tracker, tINITLINK, true);
          tracker.SetPointersDataLinks(mChain->rec()->Res(tracker.MemoryResLinks()).Ptr());
        }
        GPUCA_OPENMP(barrier)

        GPUCA_OPENMP(for)
        for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
          const GPUTPCTracker& tracker = sliceTracker(iSlice);

          mGlDLLines[iSlice][tLINK] = DrawLinks(tracker, tLINK);
          mGlDLLines[iSlice][tSEED] = DrawSeeds(tracker);
          mGlDLLines[iSlice][tTRACKLET] = DrawTracklets(tracker);
          mGlDLLines[iSlice][tSLICETRACK] = DrawTracks(tracker, 0);
          mGlDLGrid[iSlice] = DrawGrid(tracker);
          if (iSlice < NSLICES / 2) {
            mGlDLGridTRD[iSlice] = DrawGridTRD(iSlice);
          }
        }
        GPUCA_OPENMP(barrier)

        GPUCA_OPENMP(for)
        for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
          const GPUTPCTracker& tracker = sliceTracker(iSlice);
          mGlDLLines[iSlice][tGLOBALTRACK] = DrawTracks(tracker, 1);
        }
        GPUCA_OPENMP(barrier)
      }
      mThreadTracks[numThread].resize(mNCollissions);
      for (int i = 0; i < mNCollissions; i++) {
        for (int j = 0; j < NSLICES; j++) {
          for (int k = 0; k < 2; k++) {
            mThreadTracks[numThread][i][j][k].clear();
          }
        }
      }
      if (mConfig.showTPCTracksFromO2Format) {
        unsigned int col = 0;
        GPUCA_OPENMP(for)
        for (unsigned int i = 0; i < mIOPtrs->nOutputTracksTPCO2; i++) {
          uint8_t sector, row;
          if (mIOPtrs->clustersNative) {
            mIOPtrs->outputTracksTPCO2[i].getCluster(mIOPtrs->outputClusRefsTPCO2, 0, *mIOPtrs->clustersNative, sector, row);
          } else {
            sector = 0;
          }
          mThreadTracks[numThread][col][sector][0].emplace_back(i);
        }
      } else {
        GPUCA_OPENMP(for)
        for (unsigned int i = 0; i < mIOPtrs->nMergedTracks; i++) {
          const GPUTPCGMMergedTrack* track = &mIOPtrs->mergedTracks[i];
          if (track->NClusters() == 0) {
            continue;
          }
          if (mCfgH.hideRejectedTracks && !track->OK()) {
            continue;
          }
          int slice = mIOPtrs->mergedTrackHits[track->FirstClusterRef() + track->NClusters() - 1].slice;
          unsigned int col = 0;
          if (mCollisionClusters.size() > 1) {
            int label = mQA ? mQA->GetMCTrackLabel(i) : -1;
            while (col < mCollisionClusters.size() && mCollisionClusters[col][NSLICES] < label) {
              col++;
            }
          }
          mThreadTracks[numThread][col][slice][0].emplace_back(i);
        }
      }
      for (unsigned int col = 0; col < mIOPtrs->nMCInfosTPCCol; col++) {
        GPUCA_OPENMP(for)
        for (unsigned int i = mIOPtrs->mcInfosTPCCol[col].first; i < mIOPtrs->mcInfosTPCCol[col].first + mIOPtrs->mcInfosTPCCol[col].num; i++) {
          const GPUTPCMCInfo& mc = mIOPtrs->mcInfosTPC[i];
          if (mc.charge == 0.f) {
            continue;
          }
          if (mc.pid < 0) {
            continue;
          }

          float alpha = atan2f(mc.y, mc.x);
          if (alpha < 0) {
            alpha += 2 * CAMath::Pi();
          }
          int slice = alpha / (2 * CAMath::Pi()) * 18;
          if (mc.z < 0) {
            slice += 18;
          }
          mThreadTracks[numThread][col][slice][1].emplace_back(i);
        }
      }
      GPUCA_OPENMP(barrier)

      GPUTPCGMPropagator prop;
      prop.SetMaxSinPhi(.999);
      prop.SetMaterialTPC();
      prop.SetPolynomialField(&mParam->polynomialField);

      GPUCA_OPENMP(for)
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mThreadBuffers[numThread].clear();
          for (int iSet = 0; iSet < numThreads; iSet++) {
#ifdef GPUCA_HAVE_O2HEADERS
            if (mConfig.showTPCTracksFromO2Format) {
              DrawFinal<o2::tpc::TrackTPC>(iSlice, iCol, &prop, mThreadTracks[iSet][iCol][iSlice], mThreadBuffers[numThread]);
            } else
#endif
            {
              DrawFinal<GPUTPCGMMergedTrack>(iSlice, iCol, &prop, mThreadTracks[iSet][iCol][iSlice], mThreadBuffers[numThread]);
            }
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

      GPUCA_OPENMP(barrier)
      GPUCA_OPENMP(for)
      for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
        for (int i = 0; i < N_POINTS_TYPE_TPC; i++) {
          for (int iCol = 0; iCol < mNCollissions; iCol++) {
            mGlDLPoints[iSlice][i][iCol] = DrawClusters(iSlice, i, iCol);
          }
        }
      }
    }
    // End omp parallel

    mGlDLFinalITS = DrawFinalITS();

    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (int i = N_POINTS_TYPE_TPC; i < N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD; i++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mGlDLPoints[iSlice][i][iCol] = DrawSpacePointsTRD(iSlice, i, iCol);
        }
      }
    }

    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (int i = N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD; i < N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD + N_POINTS_TYPE_TOF; i++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mGlDLPoints[iSlice][i][iCol] = DrawSpacePointsTOF(iSlice, i, iCol);
        }
      }
      break; // TODO: Only slice 0 filled for now
    }

    for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (int i = N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD + N_POINTS_TYPE_TOF; i < N_POINTS_TYPE_TPC + N_POINTS_TYPE_TRD + N_POINTS_TYPE_TOF + N_POINTS_TYPE_ITS; i++) {
        for (int iCol = 0; iCol < mNCollissions; iCol++) {
          mGlDLPoints[iSlice][i][iCol] = DrawSpacePointsITS(iSlice, i, iCol);
        }
      }
      break; // TODO: Only slice 0 filled for now
    }

    mGlDLrecent = 1;
    size_t totalVertizes = 0;
    for (int i = 0; i < NSLICES; i++) {
      totalVertizes += mVertexBuffer[i].size();
    }

    // TODO: Check if this can be parallelized
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

    if (mCfgR.useGLIndirectDraw) {
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
      printf("Event visualization time: %'d us (vertices %'lld / %'lld bytes)\n", (int)(mTimerDraw.GetCurrentElapsedTime() * 1000000.), (long long int)totalVertizes, (long long int)(totalVertizes * sizeof(mVertexBuffer[0][0])));
    }
  }

  // Draw Event
  mNDrawCalls = 0;
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mCfgR.openGLCore) {
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

  {
    const float zFar = ((mParam->par.continuousTracking ? (mMaxClusterZ / GL_SCALE_FACTOR) : 8.f) + 50.f) * 2.f;
    const hmm_mat4 proj = HMM_Perspective(mFOV, (GLfloat)mScreenwidth / (GLfloat)mScreenheight, 0.1f, zFar);
    nextViewMatrix = nextViewMatrix * mModelMatrix;
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    if (!mCfgR.openGLCore) {
      CHKERR(glMatrixMode(GL_PROJECTION));
      CHKERR(glLoadMatrixf(&proj.Elements[0][0]));
      CHKERR(glMatrixMode(GL_MODELVIEW));
      CHKERR(glLoadMatrixf(&nextViewMatrix.Elements[0][0]));
    } else
#endif
    {
      const hmm_mat4 modelViewProj = proj * nextViewMatrix;
      CHKERR(glUniformMatrix4fv(mModelViewProjId, 1, GL_FALSE, &modelViewProj.Elements[0][0]));
    }
  }

#define LOOP_SLICE for (int iSlice = (mCfgL.drawSlice == -1 ? 0 : mCfgL.drawRelatedSlices ? (mCfgL.drawSlice % (NSLICES / 4)) : mCfgL.drawSlice); iSlice < NSLICES; iSlice += (mCfgL.drawSlice == -1 ? 1 : mCfgL.drawRelatedSlices ? (NSLICES / 4) : NSLICES))
#define LOOP_SLICE2 for (int iSlice = (mCfgL.drawSlice == -1 ? 0 : mCfgL.drawRelatedSlices ? (mCfgL.drawSlice % (NSLICES / 4)) : mCfgL.drawSlice) % (NSLICES / 2); iSlice < NSLICES / 2; iSlice += (mCfgL.drawSlice == -1 ? 1 : mCfgL.drawRelatedSlices ? (NSLICES / 4) : NSLICES))
#define LOOP_COLLISION for (int iCol = (mCfgL.showCollision == -1 ? 0 : mCfgL.showCollision); iCol < mNCollissions; iCol += (mCfgL.showCollision == -1 ? 1 : mNCollissions))
#define LOOP_COLLISION_COL(cmd) \
  LOOP_COLLISION                \
  {                             \
    if (mCfgL.colorCollisions) { \
      SetCollisionColor(iCol);  \
    }                           \
    cmd;                        \
  }

  if (mCfgL.drawGrid) {
    if (mCfgL.drawTPC) {
      SetColorGrid();
      LOOP_SLICE drawVertices(mGlDLGrid[iSlice], GL_LINES);
    }
    if (mCfgL.drawTRD) {
      SetColorGridTRD();
      LOOP_SLICE2 drawVertices(mGlDLGridTRD[iSlice], GL_LINES);
    }
  }
  if (mCfgL.drawClusters) {
    if (mCfgL.drawTRD) {
      SetColorTRD();
      CHKERR(glLineWidth(mCfgL.lineWidth * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1) * 2));
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDCLUSTER][iCol], GL_LINES));
      if (mCfgL.drawFinal && mCfgL.colorClusters) {
        SetColorFinal();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDATTACHED][iCol], GL_LINES));
      CHKERR(glLineWidth(mCfgL.lineWidth * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1)));
    }
    if (mCfgL.drawTOF) {
      SetColorTOF();
      CHKERR(glPointSize(mCfgL.pointSize * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1) * 2));
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[0][tTOFCLUSTER][0], GL_POINTS));
      CHKERR(glPointSize(mCfgL.pointSize * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1)));
    }
    if (mCfgL.drawITS) {
      SetColorITS();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[0][tITSCLUSTER][0], GL_POINTS));
    }
    if (mCfgL.drawTPC) {
      SetColorClusters();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tCLUSTER][iCol], GL_POINTS));

      if (mCfgL.drawInitLinks) {
        if (mCfgL.excludeClusters) {
          goto skip1;
        }
        if (mCfgL.colorClusters) {
          SetColorInitLinks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tINITLINK][iCol], GL_POINTS));

      if (mCfgL.drawLinks) {
        if (mCfgL.excludeClusters) {
          goto skip1;
        }
        if (mCfgL.colorClusters) {
          SetColorLinks();
        }
      } else {
        SetColorClusters();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tLINK][iCol], GL_POINTS));

      if (mCfgL.drawSeeds) {
        if (mCfgL.excludeClusters) {
          goto skip1;
        }
        if (mCfgL.colorClusters) {
          SetColorSeeds();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSEED][iCol], GL_POINTS));

    skip1:
      SetColorClusters();
      if (mCfgL.drawTracklets) {
        if (mCfgL.excludeClusters) {
          goto skip2;
        }
        if (mCfgL.colorClusters) {
          SetColorTracklets();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRACKLET][iCol], GL_POINTS));

      if (mCfgL.drawTracks) {
        if (mCfgL.excludeClusters) {
          goto skip2;
        }
        if (mCfgL.colorClusters) {
          SetColorTracks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSLICETRACK][iCol], GL_POINTS));

    skip2:;
      if (mCfgL.drawGlobalTracks) {
        if (mCfgL.excludeClusters) {
          goto skip3;
        }
        if (mCfgL.colorClusters) {
          SetColorGlobalTracks();
        }
      } else {
        SetColorClusters();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tGLOBALTRACK][iCol], GL_POINTS));
      SetColorClusters();

      if (mCfgL.drawFinal && mCfgL.propagateTracks < 2) {
        if (mCfgL.excludeClusters) {
          goto skip3;
        }
        if (mCfgL.colorClusters) {
          SetColorFinal();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tFINALTRACK][iCol], GL_POINTS));
    skip3:;
    }
  }

  if (!mCfgH.clustersOnly && !mCfgL.excludeClusters) {
    if (mCfgL.drawTPC) {
      if (mCfgL.drawInitLinks) {
        SetColorInitLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tINITLINK], GL_LINES);
      }
      if (mCfgL.drawLinks) {
        SetColorLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tLINK], GL_LINES);
      }
      if (mCfgL.drawSeeds) {
        SetColorSeeds();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSEED], GL_LINE_STRIP);
      }
      if (mCfgL.drawTracklets) {
        SetColorTracklets();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tTRACKLET], GL_LINE_STRIP);
      }
      if (mCfgL.drawTracks) {
        SetColorTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSLICETRACK], GL_LINE_STRIP);
      }
      if (mCfgL.drawGlobalTracks) {
        SetColorGlobalTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tGLOBALTRACK], GL_LINE_STRIP);
      }
    }
    if (mCfgL.drawFinal) {
      SetColorFinal();
      LOOP_SLICE LOOP_COLLISION
      {
        if (mCfgL.colorCollisions) {
          SetCollisionColor(iCol);
        }
        if (mCfgL.propagateTracks < 2) {
          drawVertices(mGlDLFinal[iSlice][iCol][0], GL_LINE_STRIP);
        }
        if (mCfgL.propagateTracks > 0 && mCfgL.propagateTracks < 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][1], GL_LINE_STRIP);
        }
        if (mCfgL.propagateTracks == 2) {
          drawVertices(mGlDLFinal[iSlice][iCol][2], GL_LINE_STRIP);
        }
        if (mCfgL.propagateTracks == 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][3], GL_LINE_STRIP);
        }
      }
      if (mCfgH.drawTracksAndFilter ? (mCfgH.drawTPCTracks || mCfgH.drawTRDTracks || mCfgH.drawTOFTracks) : mCfgH.drawITSTracks) {
        drawVertices(mGlDLFinalITS, GL_LINE_STRIP);
      }
    }
    if (mCfgH.markClusters || mCfgH.markAdjacentClusters || mCfgH.markFakeClusters) {
      if (mCfgH.markFakeClusters) {
        CHKERR(glPointSize(mCfgL.pointSize * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1) * 3));
      }
      SetColorMarked();
      LOOP_SLICE LOOP_COLLISION drawVertices(mGlDLPoints[iSlice][tMARKED][iCol], GL_POINTS);
      if (mCfgH.markFakeClusters) {
        CHKERR(glPointSize(mCfgL.pointSize * (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1)));
      }
    }
  }
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mCfgR.openGLCore) {
    CHKERR(glDisableClientState(GL_VERTEX_ARRAY));
  } else
#endif
  {
    CHKERR(glDisableVertexAttribArray(0));
    CHKERR(glUseProgram(0));
  }

  if (mixSlaveImage > 0) {
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    if (!mCfgR.openGLCore) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glMatrixMode(GL_PROJECTION);
      hmm_mat4 proj = HMM_Orthographic(0.f, mRenderwidth, 0.f, mRenderheight, -1.f, 1.f);
      glLoadMatrixf(&proj.Elements[0][0]);
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
    } else
#endif
    {
      GPUWarning("Image mixing unsupported in OpenGL CORE profile");
    }
  }

  if (mixAnimation) {
    glColorMask(false, false, false, true);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(true, true, true, true);
  } else if (mOffscreenBuffer.created) {
    setFrameBuffer();
    GLuint srcid = mOffscreenBuffer.fb_id;
    if (mCfgR.drawQualityMSAA > 1 && mCfgR.drawQualityDownsampleFSAA > 1) {
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
            fps, mCfgL.drawSlice, mCfgL.drawClusters, mCfgL.drawInitLinks, mCfgL.drawLinks, mCfgL.drawSeeds, mCfgL.drawTracklets, mCfgL.drawTracks, mCfgL.drawGlobalTracks, mCfgL.drawFinal, mFramesDone, mNDrawCalls, mXYZ[0], mXYZ[1], mXYZ[2], mRPhiTheta[0], mRPhiTheta[1] * 180 / CAMath::Pi(),
            mRPhiTheta[2] * 180 / CAMath::Pi(), mAngle[1] * 180 / CAMath::Pi(), mAngle[0] * 180 / CAMath::Pi(), mAngle[2] * 180 / CAMath::Pi());
    if (fpstime > 1.) {
      if (mPrintInfoText & 2) {
        GPUInfo("%s", info);
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
  int SCALE_Y = mCfgR.screenshotScaleFactor, SCALE_X = mCfgR.screenshotScaleFactor;

  float tmpPointSize = mCfgL.pointSize;
  float tmpLineWidth = mCfgL.lineWidth;
  mCfgL.pointSize *= (float)(SCALE_X + SCALE_Y) / 2.;
  mCfgL.lineWidth *= (float)(SCALE_X + SCALE_Y) / 2.;

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

  mCfgL.pointSize = tmpPointSize;
  mCfgL.lineWidth = tmpLineWidth;
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
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mCfgR.openGLCore) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    hmm_mat4 proj = HMM_Orthographic(0.f, mScreenwidth, 0.f, mScreenheight, -1, 1);
    glLoadMatrixf(&proj.Elements[0][0]);
    glViewport(0, 0, mScreenwidth, mScreenheight);
  }
#endif
  float colorValue = mCfgL.invertColors ? 0.f : 1.f;
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
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!mCfgR.openGLCore) {
    glViewport(0, 0, mRenderwidth, mRenderheight);
  }
#endif
}

void GPUDisplay::ShowNextEvent(const GPUTrackingInOutPointers* ptrs)
{
  if (ptrs) {
    mIOPtrs = ptrs;
  }
  if (mMaxClusterZ <= 0) {
    mResetScene = true;
  }
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

#endif
