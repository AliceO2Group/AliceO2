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

/// \file GPUDisplay.cxx
/// \author David Rohr

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#endif

#include "GPUDisplay.h"

#include "GPUTPCDef.h"

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
#include "GPUO2DataTypes.h"
#include "GPUParam.inc"
#include "GPUTPCConvertImpl.h"
#include "utils/qconfig.h"

#ifdef GPUCA_HAVE_O2HEADERS
#include "GPUTrackParamConvert.h"
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

#include "GPUDisplayFrontend.h"
#include "GPUDisplayBackend.h"

constexpr hmm_mat4 MY_HMM_IDENTITY = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
constexpr hmm_mat4 MY_HMM_FROM(float (&v)[16]) { return {{{v[0], v[1], v[2], v[3]}, {v[4], v[5], v[6], v[7]}, {v[8], v[9], v[10], v[11]}, {v[12], v[13], v[14], v[15]}}}; }

using namespace GPUCA_NAMESPACE::gpu;

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

GPUDisplay::GPUDisplay(GPUDisplayFrontend* frontend, GPUChainTracking* chain, GPUQA* qa, const GPUParam* param, const GPUCalibObjectsConst* calib, const GPUSettingsDisplay* config) : GPUDisplayInterface(), mFrontend(frontend), mChain(chain), mConfig(config ? *config : GPUDisplay_GetConfig(chain)), mQA(qa)
{
  mParam = param ? param : &mChain->GetParam();
  mCalib = calib;
  mCfgL = mConfig.light;
  mCfgH = mConfig.heavy;
  mCfgR = mConfig.renderer;
  mBackend.reset(GPUDisplayBackend::getBackend(mConfig.displayRenderer.c_str()));
  if (!mBackend) {
    throw std::runtime_error("Error obtaining display backend");
  }
  mBackend->mDisplay = this;
  frontend->mDisplay = this;
  frontend->mBackend = mBackend.get();
  mCfgR.openGLCore = mBackend->CoreProfile();
}

inline const GPUTRDGeometry* GPUDisplay::trdGeometry() { return (GPUTRDGeometry*)mCalib->trdGeometry; }
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

inline void GPUDisplay::insertVertexList(std::pair<vecpod<int>*, vecpod<unsigned int>*>& vBuf, size_t first, size_t last)
{
  if (first == last) {
    return;
  }
  vBuf.first->emplace_back(first);
  vBuf.second->emplace_back(last - first);
}
inline void GPUDisplay::insertVertexList(int iSlice, size_t first, size_t last)
{
  std::pair<vecpod<int>*, vecpod<unsigned int>*> vBuf(mVertexBufferStart + iSlice, mVertexBufferCount + iSlice);
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
  mBackend->ActivateColor(mDrawColor);
}

inline void GPUDisplay::SetColorClusters()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.3, 0.7, 1.f};
  } else {
    mDrawColor = {0, 0.7, 1.0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTRD()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.7, 0.3, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.7, 0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorITS()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {1.00, 0.1, 0.1, 1.f};
  } else {
    mDrawColor = {1.00, 0.3, 0.3, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTOF()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0.1, 1.0, 0.1, 1.f};
  } else {
    mDrawColor = {0.5, 1.0, 0.5, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorInitLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.42, 0.4, 0.1, 1.f};
  } else {
    mDrawColor = {0.42, 0.4, 0.1, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorLinks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.1, 0.1, 1.f};
  } else {
    mDrawColor = {0.8, 0.2, 0.2, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorSeeds()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0.0, 0.65, 1.f};
  } else {
    mDrawColor = {0.8, 0.1, 0.85, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracklets()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0, 0, 1.f};
  } else {
    mDrawColor = {1, 1, 1, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.6, 0, 0.1, 1.f};
  } else {
    mDrawColor = {0.8, 1., 0.15, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGlobalTracks()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0.2, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.4, 0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorFinal()
{
  if (mCfgL.colorCollisions) {
    return;
  }
  if (mCfgL.invertColors) {
    mDrawColor = {0, 0.6, 0.1, 1.f};
  } else {
    mDrawColor = {0, 0.7, 0.2, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGrid()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.0, 1.f};
  } else {
    mDrawColor = {0.7, 0.7, 0.0, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorGridTRD()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.5, 0.5, 0.5, 1.f};
  } else {
    mDrawColor = {0.7, 0.7, 0.5, 1.f};
  }
  ActivateColor();
}
inline void GPUDisplay::SetColorMarked()
{
  if (mCfgL.invertColors) {
    mDrawColor = {0.8, 0, 0, 1.f};
  } else {
    mDrawColor = {1.0, 0.0, 0.0, 1.f};
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
  mDrawColor = {red / 4.f, green / 5.f, blue / 6.f, 1.f};
  ActivateColor();
}

void GPUDisplay::ResizeScene(int width, int height, bool init)
{
  if (height == 0) { // Prevent A Divide By Zero By
    height = 1;      // Making Height Equal One
  }

  mBackend->resizeScene(width, height);

  if (init) {
    mResetScene = 1;
    mViewMatrix = MY_HMM_IDENTITY;
    mModelMatrix = MY_HMM_IDENTITY;
  }
}

void GPUDisplay::updateConfig()
{
  mBackend->setQuality();
  mBackend->setDepthBuffer();
}

inline void GPUDisplay::drawVertices(const vboList& v, const GPUDisplayBackend::drawType t)
{
  mNDrawCalls += mBackend->drawVertices(v, t);
}

int GPUDisplay::InitDisplay(bool initFailure)
{
  int retVal = initFailure;
  try {
    if (!initFailure) {
      retVal = InitDisplay_internal();
    }
  } catch (const std::runtime_error& e) {
    GPUError("%s", e.what());
    retVal = 1;
  }
  mInitResult = retVal == 0 ? 1 : -1;
  return (retVal);
}

int GPUDisplay::InitDisplay_internal()
{
  mThreadBuffers.resize(getNumThreads());
  mThreadTracks.resize(getNumThreads());
  if (mBackend->InitBackend()) {
    return 1;
  }
  mYFactor = mBackend->getYFactor();
  mDrawTextInCompatMode = !mBackend->mFreetypeInitialized && mFrontend->mCanDrawText == 1;
  int height = 0, width = 0;
  mFrontend->getSize(width, height);
  if (height == 0 || width == 0) {
    width = GPUDisplayFrontend::INIT_WIDTH;
    height = GPUDisplayFrontend::INIT_HEIGHT;
  }
  ResizeScene(width, height, true);
  return 0;
}

void GPUDisplay::ExitDisplay()
{
  mBackend->ExitBackend();
}

inline void GPUDisplay::drawPointLinestrip(int iSlice, int cid, int id, int id_limit)
{
  mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPos[cid].z);
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
      int iSec = trdGeometry()->GetSector(mIOPtrs->trdTracklets[i].GetDetector());
      bool draw = iSlice == iSec && mGlobalPosTRD[i].w == select;
      if (draw) {
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[i].x, mGlobalPosTRD[i].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTRD[i].z);
        mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[i].x, mGlobalPosTRD2[i].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTRD2[i].z);
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
      mVertexBuffer[iSlice].emplace_back(mGlobalPosTOF[i].x, mGlobalPosTOF[i].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTOF[i].z);
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
      mVertexBuffer[iSlice].emplace_back(mGlobalPosITS[i].x, mGlobalPosITS[i].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosITS[i].z);
    }
  }

  insertVertexList(iSlice, startCountInner, mVertexBuffer[iSlice].size());
  return (vboList(startCount, mVertexBufferStart[iSlice].size() - startCount, iSlice));
}

GPUDisplay::vboList GPUDisplay::DrawClusters(int iSlice, int select, unsigned int iCol)
{
  size_t startCount = mVertexBufferStart[iSlice].size();
  size_t startCountInner = mVertexBuffer[iSlice].size();
  if (mCollisionClusters.size() > 0 || iCol == 0) {
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
        mVertexBuffer[iSlice].emplace_back(mGlobalPos[cid].x, mGlobalPos[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPos[cid].z);
      }
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
      if (rowHit != CALINK_INVAL && rowHit != CALINK_DEAD_CHANNEL) {
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
#ifdef GPUCA_HAVE_O2HEADERS
  const auto& trk = mIOPtrs->itsTracks[trackId];
  for (int k = 0; k < trk.getNClusters(); k++) {
    int cid = mIOPtrs->itsTrackClusIdx[trk.getFirstClusterEntry() + k];
    mVertexBuffer[iSlice].emplace_back(mGlobalPosITS[cid].x, mGlobalPosITS[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosITS[cid].z);
    mGlobalPosITS[cid].w = tITSATTACHED;
  }
#endif
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

      if (mCfgH.trackFilter && !mTrackFilter[i]) {
        break;
      }

      // Print TOF part of track
      if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (mIOPtrs->tpcLinkTOF && mIOPtrs->tpcLinkTOF[i] != -1 && mIOPtrs->nTOFClusters) {
          int cid = mIOPtrs->tpcLinkTOF[i];
          drawing = true;
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTOF[cid].x, mGlobalPosTOF[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTOF[cid].z);
          mGlobalPosTOF[cid].w = tTOFATTACHED;
        }
      }

      // Print TRD part of track
      auto tmpDoTRDTracklets = [&](const auto& trk) {
        for (int k = 5; k >= 0; k--) {
          int cid = trk.getTrackletIndex(k);
          if (cid < 0) {
            continue;
          }
          drawing = true;
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD2[cid].x, mGlobalPosTRD2[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTRD2[cid].z);
          mVertexBuffer[iSlice].emplace_back(mGlobalPosTRD[cid].x, mGlobalPosTRD[cid].y * mYFactor, mCfgH.projectXY ? 0 : mGlobalPosTRD[cid].z);
          mGlobalPosTRD[cid].w = tTRDATTACHED;
        }
      };
      if (std::is_same_v<T, GPUTPCGMMergedTrack> || (!mIOPtrs->tpcLinkTRD && mIOPtrs->trdTracksO2)) {
        if (mChain && ((int)mConfig.showTPCTracksFromO2Format == (int)mChain->GetProcessingSettings().trdTrackModelO2) && mTRDTrackIds[i] != -1 && mIOPtrs->nTRDTracklets) {
          if (mIOPtrs->trdTracksO2) {
#ifdef GPUCA_HAVE_O2HEADERS
            tmpDoTRDTracklets(mIOPtrs->trdTracksO2[mTRDTrackIds[i]]);
#endif
          } else {
            tmpDoTRDTracklets(mIOPtrs->trdTracks[mTRDTrackIds[i]]);
          }
        }
      } else if constexpr (std::is_same_v<T, o2::tpc::TrackTPC>) {
        if (mIOPtrs->tpcLinkTRD && mIOPtrs->tpcLinkTRD[i] != -1 && mIOPtrs->nTRDTracklets) {
          if ((mIOPtrs->tpcLinkTRD[i] & 0x40000000) ? mIOPtrs->nTRDTracksITSTPCTRD : mIOPtrs->nTRDTracksTPCTRD) {
            const auto* container = (mIOPtrs->tpcLinkTRD[i] & 0x40000000) ? mIOPtrs->trdTracksITSTPCTRD : mIOPtrs->trdTracksTPCTRD;
            const auto& trk = container[mIOPtrs->tpcLinkTRD[i] & 0x3FFFFFFF];
            tmpDoTRDTracklets(trk);
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
              ZOffset = mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(iSlice, track->GetParam().GetTZOffset(), mParam->par.continuousMaxTimeBin);
            } else {
              uint8_t sector, row;
              auto cln = track->getCluster(mIOPtrs->outputClusRefsTPCO2, lastCluster, *mIOPtrs->clustersNative, sector, row);
              GPUTPCConvertImpl::convert(*mCalib->fastTransform, *mParam, sector, row, cln.getPad(), cln.getTime(), x, y, z);
              ZOffset = mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(sector, track->getTime0(), mParam->par.continuousMaxTimeBin);
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
            ZOffset = fabsf(mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(0, mc.t0, mParam->par.continuousMaxTimeBin)) * (mc.z < 0 ? -1 : 1);
          }
#else
          if (fabsf(mc.z) > GPUTPCGeometry::TPCLength()) {
            ZOffset = mc.z > 0 ? (mc.z - GPUTPCGeometry::TPCLength()) : (mc.z + GPUTPCGeometry::TPCLength());
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
        vecpod<vtx>& useBuffer = iMC && inFlyDirection == 0 ? buffer : mVertexBuffer[iSlice];
        int nPoints = 0;

        while (nPoints++ < 5000) {
          if ((inFlyDirection == 0 && x < 0) || (inFlyDirection && x * x + trkParam.Y() * trkParam.Y() > (iMC ? (450 * 450) : (300 * 300)))) {
            break;
          }
          if (fabsf(trkParam.Z() + ZOffset) > mMaxClusterZ + (iMC ? 0 : 0)) {
            break;
          }
          if (fabsf(trkParam.Z() - z0) > (iMC ? GPUTPCGeometry::TPCLength() : GPUTPCGeometry::TPCLength())) {
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
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR * mYFactor, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR * mYFactor, zz2 / GL_SCALE_FACTOR);
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
      mVertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR * mYFactor, zz1 / GL_SCALE_FACTOR);
      mVertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR * mYFactor, zz2 / GL_SCALE_FACTOR);
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
  auto* geo = trdGeometry();
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
          mVertexBuffer[sector].emplace_back(xyzGlb1[0] / GL_SCALE_FACTOR, xyzGlb1[1] / GL_SCALE_FACTOR * mYFactor, xyzGlb1[2] / GL_SCALE_FACTOR);
          mVertexBuffer[sector].emplace_back(xyzGlb2[0] / GL_SCALE_FACTOR, xyzGlb2[1] / GL_SCALE_FACTOR * mYFactor, xyzGlb2[2] / GL_SCALE_FACTOR);
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

int GPUDisplay::DrawGLScene()
{
  // Make sure event gets not overwritten during display
  mSemLockDisplay.Lock();

  int retVal = 0;
  if (mChain) {
    mIOPtrs = &mChain->mIOPtrs;
    mCalib = &mChain->calib();
  }
  if (!mIOPtrs) {
    mNCollissions = 0;
  } else if (!mCollisionClusters.size()) {
    mNCollissions = std::max(1u, mIOPtrs->nMCInfosTPCCol);
  }
  try {
    DrawGLScene_internal();
  } catch (const std::runtime_error& e) {
    GPUError("Runtime error %s during display", e.what());
    retVal = 1;
  }
  mSemLockDisplay.Unlock();

  return retVal;
}

void GPUDisplay::DrawGLScene_updateEventData()
{
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
  auto tmpDoTRDTracklets = [&](auto* trdTracks) {
    for (unsigned int i = 0; i < mIOPtrs->nTRDTracks; i++) {
      if (trdTracks[i].getNtracklets()) {
        mTRDTrackIds[trdTracks[i].getRefGlobalTrackIdRaw()] = i;
      }
    }
  };
  if (mIOPtrs->trdTracksO2) {
#ifdef GPUCA_HAVE_O2HEADERS
    tmpDoTRDTracklets(mIOPtrs->trdTracksO2);
#endif
  } else {
    tmpDoTRDTracklets(mIOPtrs->trdTracks);
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

  if (mCfgH.trackFilter) {
    unsigned int nTracks = mConfig.showTPCTracksFromO2Format ? mIOPtrs->nOutputTracksTPCO2 : mIOPtrs->nMergedTracks;
    mTrackFilter.resize(nTracks);
    std::fill(mTrackFilter.begin(), mTrackFilter.end(), true);
    if (buildTrackFilter()) {
      SetInfo("Error running track filter from %s", mConfig.filterMacros[mCfgH.trackFilter - 1].c_str());
    } else {
      unsigned int nFiltered = 0;
      for (unsigned int i = 0; i < mTrackFilter.size(); i++) {
        nFiltered += !mTrackFilter[i];
      }
      if (mUpdateTrackFilter) {
        SetInfo("Applied track filter %s - filtered %u / %u", mConfig.filterMacros[mCfgH.trackFilter - 1].c_str(), nFiltered, (unsigned int)mTrackFilter.size());
      }
    }
  }
  mUpdateTrackFilter = false;

  mMaxClusterZ = 0;
  GPUCA_OPENMP(parallel for num_threads(getNumThreads()) reduction(max : mMaxClusterZ))
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    int row = 0;
    unsigned int nCls = mParam->par.earlyTpcTransform ? mIOPtrs->nClusterData[iSlice] : mIOPtrs->clustersNative ? mIOPtrs->clustersNative->nClustersSector[iSlice]
                                                                                                                : 0;
    for (unsigned int i = 0; i < nCls; i++) {
      int cid;
      if (mParam->par.earlyTpcTransform) {
        const auto& cl = mIOPtrs->clusterData[iSlice][i];
        cid = cl.id;
        row = cl.row;
      } else {
        cid = mIOPtrs->clustersNative->clusterOffset[iSlice][0] + i;
        while (row < GPUCA_ROW_COUNT - 1 && mIOPtrs->clustersNative->clusterOffset[iSlice][row + 1] <= (unsigned int)cid) {
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
#ifdef GPUCA_HAVE_O2HEADERS
      float trdTime = mIOPtrs->trdTriggerTimes[trdTriggerRecord] * 1e3 / o2::constants::lhc::LHCBunchSpacingNS / o2::tpc::constants::LHCBCPERTIMEBIN;
      trdZoffset = fabsf(mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(0, trdTime, mParam->par.continuousMaxTimeBin));
#endif
    }
    const auto& sp = mIOPtrs->trdSpacePoints[i];
    int iSec = trdGeometry()->GetSector(mIOPtrs->trdTracklets[i].GetDetector());
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
      ZOffset = fabsf(mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(0, tofTime, mParam->par.continuousMaxTimeBin));
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
        ZOffset = fabsf(mCalib->fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(0, itsROFtime + itsROFhalfLen, mParam->par.continuousMaxTimeBin));
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
}

void GPUDisplay::DrawGLScene_cameraAndAnimation(float animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix)
{
  int mMouseWheelTmp = mFrontend->mMouseWheel;
  mFrontend->mMouseWheel = 0;
  bool lookOrigin = mCfgR.camLookOrigin ^ mFrontend->mKeys[mFrontend->KEY_ALT];
  bool yUp = mCfgR.camYUp ^ mFrontend->mKeys[mFrontend->KEY_CTRL] ^ lookOrigin;
  bool rotateModel = mFrontend->mKeys[mFrontend->KEY_RCTRL] || mFrontend->mKeys[mFrontend->KEY_RALT];
  bool rotateModelTPC = mFrontend->mKeys[mFrontend->KEY_RALT];

  // Calculate rotation / translation scaling factors
  float scalefactor = mFrontend->mKeys[mFrontend->KEY_SHIFT] ? 0.2 : 1.0;
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

  float time = animateTime;
  if (mAnimate && time < 0) {
    if (mAnimateScreenshot) {
      time = mAnimationFrame / 30.f;
    } else {
      time = mAnimationTimer.GetCurrentElapsedTime();
    }

    float maxTime = mAnimateVectors[0].back();
    if (time >= maxTime) {
      time = maxTime;
      mAnimate = mAnimateScreenshot = 0;
      SetInfo("Animation finished. (%1.2f seconds, %d frames)", time, mAnimationFrame);
    } else {
      SetInfo("Running mAnimation: time %1.2f/%1.2f, frames %d", time, maxTime, mAnimationFrame);
    }
    mAnimationFrame++;
  }
  // Perform new rotation / translation
  if (mAnimate) {
    float vals[8];
    for (int i = 0; i < 8; i++) {
      vals[i] = mAnimationSplines[i].evaluate(time);
    }
    if (mAnimationChangeConfig && animateTime < 0) {
      int base = 0;
      int k = mAnimateVectors[0].size() - 1;
      while (base < k && time > mAnimateVectors[0][base]) {
        base++;
      }
      if (base > mAnimationLastBase + 1) {
        mAnimationLastBase = base - 1;
      }

      if (base != mAnimationLastBase && mAnimateVectors[0][mAnimationLastBase] != mAnimateVectors[0][base] && memcmp(&mAnimateConfig[base], &mAnimateConfig[mAnimationLastBase], sizeof(mAnimateConfig[base]))) {
        mixSlaveImage = 1.f - (time - mAnimateVectors[0][mAnimationLastBase]) / (mAnimateVectors[0][base] - mAnimateVectors[0][mAnimationLastBase]);
        if (mixSlaveImage > 0) {
          mCfgL = mAnimateConfig[mAnimationLastBase];
          updateConfig();
          DrawGLScene_internal(time, true);
        }
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

    mCfgL.pointSize = 2.0f;
    mCfgL.lineWidth = 1.4f;
    mCfgL.drawSlice = -1;
    mCfgH.xAdd = mCfgH.zAdd = 0;
    mCfgR.camLookOrigin = mCfgR.camYUp = false;
    mAngleRollOrigin = -1e9f;
    mCfgR.fov = 45.f;
    mUpdateDrawCommands = 1;

    mResetScene = 0;
  } else {
    float moveZ = scalefactor * ((float)mMouseWheelTmp / 150 + (float)(mFrontend->mKeys[(unsigned char)'W'] - mFrontend->mKeys[(unsigned char)'S']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]) * 0.2 * mFPSScale);
    float moveY = scalefactor * ((float)(mFrontend->mKeys[mFrontend->KEY_PAGEDOWN] - mFrontend->mKeys[mFrontend->KEY_PAGEUP]) * 0.2 * mFPSScale);
    float moveX = scalefactor * ((float)(mFrontend->mKeys[(unsigned char)'A'] - mFrontend->mKeys[(unsigned char)'D']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]) * 0.2 * mFPSScale);
    float rotRoll = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[(unsigned char)'E'] - mFrontend->mKeys[(unsigned char)'F']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]);
    float rotYaw = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[mFrontend->KEY_RIGHT] - mFrontend->mKeys[mFrontend->KEY_LEFT]);
    float rotPitch = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[mFrontend->KEY_DOWN] - mFrontend->mKeys[mFrontend->KEY_UP]);

    float mouseScale = 1920.f / std::max<float>(1920.f, mBackend->mScreenWidth);
    if (mFrontend->mMouseDnR && mFrontend->mMouseDn) {
      moveZ += -scalefactor * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY) / 4;
      rotRoll += -rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvX - (float)mFrontend->mMouseDnX);
    } else if (mFrontend->mMouseDnR) {
      moveX += scalefactor * 0.5 * mouseScale * ((float)mFrontend->mMouseDnX - (float)mFrontend->mMouseMvX) / 4;
      moveY += scalefactor * 0.5 * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY) / 4;
    } else if (mFrontend->mMouseDn) {
      rotYaw += rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvX - (float)mFrontend->mMouseDnX);
      rotPitch += rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY);
    }

    if (mFrontend->mKeys[(unsigned char)'<'] && !mFrontend->mKeysShift[(unsigned char)'<']) {
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
      nextViewMatrix = nextViewMatrix * HMM_Translate({moveX, moveY * mYFactor, moveZ});
      if (!rotateModel) {
        if (rotYaw != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotYaw, {0, 1, 0});
        }
        if (rotPitch != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotPitch * mYFactor, {1, 0, 0});
        }
        if (!yUp && rotRoll != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotRoll * mYFactor, {0, 0, 1});
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
    int deltaLine = mFrontend->mKeys[(unsigned char)'+'] * mFrontend->mKeysShift[(unsigned char)'+'] - mFrontend->mKeys[(unsigned char)'-'] * mFrontend->mKeysShift[(unsigned char)'-'];
    mCfgL.lineWidth += (float)deltaLine * mFPSScale * 0.02 * mCfgL.lineWidth;
    if (mCfgL.lineWidth < minSize) {
      mCfgL.lineWidth = minSize;
    }
    if (deltaLine) {
      SetInfo("%s line width: %f", deltaLine > 0 ? "Increasing" : "Decreasing", mCfgL.lineWidth);
      mUpdateDrawCommands = 1;
    }
    minSize *= 2;
    int deltaPoint = mFrontend->mKeys[(unsigned char)'+'] * (!mFrontend->mKeysShift[(unsigned char)'+']) - mFrontend->mKeys[(unsigned char)'-'] * (!mFrontend->mKeysShift[(unsigned char)'-']);
    mCfgL.pointSize += (float)deltaPoint * mFPSScale * 0.02 * mCfgL.pointSize;
    if (mCfgL.pointSize < minSize) {
      mCfgL.pointSize = minSize;
    }
    if (deltaPoint) {
      SetInfo("%s point size: %f", deltaPoint > 0 ? "Increasing" : "Decreasing", mCfgL.pointSize);
      mUpdateDrawCommands = 1;
    }
  }

  // Store position
  if (animateTime < 0) {
    mViewMatrix = nextViewMatrix;
    calcXYZ(mViewMatrixP);
  }

  if (mFrontend->mMouseDn || mFrontend->mMouseDnR) {
    mFrontend->mMouseDnX = mFrontend->mMouseMvX;
    mFrontend->mMouseDnY = mFrontend->mMouseMvY;
  }
}

size_t GPUDisplay::DrawGLScene_updateVertexList()
{
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
#ifdef GPUCA_HAVE_O2HEADERS
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
#endif
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

  mUpdateVertexLists = 0;
  size_t totalVertizes = 0;
  for (int i = 0; i < NSLICES; i++) {
    totalVertizes += mVertexBuffer[i].size();
  }
  if (totalVertizes > 0xFFFFFFFF) {
    throw std::runtime_error("Display vertex count exceeds 32bit uint counter");
  }
  size_t needMultiVBOSize = mBackend->needMultiVBO();
  mUseMultiVBO = needMultiVBOSize && (totalVertizes * sizeof(mVertexBuffer[0][0]) >= needMultiVBOSize);
  if (!mUseMultiVBO) {
    size_t totalYet = mVertexBuffer[0].size();
    mVertexBuffer[0].resize(totalVertizes);
    for (int i = 1; i < GPUCA_NSLICES; i++) {
      for (unsigned int j = 0; j < mVertexBufferStart[i].size(); j++) {
        mVertexBufferStart[i][j] += totalYet;
      }
      memcpy(&mVertexBuffer[0][totalYet], &mVertexBuffer[i][0], mVertexBuffer[i].size() * sizeof(mVertexBuffer[i][0]));
      totalYet += mVertexBuffer[i].size();
      mVertexBuffer[i].clear();
    }
  }
  mBackend->loadDataToGPU(totalVertizes);
  for (int i = 0; i < (mUseMultiVBO ? GPUCA_NSLICES : 1); i++) {
    mVertexBuffer[i].clear();
  }
  return totalVertizes;
}

void GPUDisplay::DrawGLScene_drawCommands()
{
#define LOOP_SLICE for (int iSlice = (mCfgL.drawSlice == -1 ? 0 : mCfgL.drawRelatedSlices ? (mCfgL.drawSlice % (NSLICES / 4)) : mCfgL.drawSlice); iSlice < NSLICES; iSlice += (mCfgL.drawSlice == -1 ? 1 : mCfgL.drawRelatedSlices ? (NSLICES / 4) : NSLICES))
#define LOOP_SLICE2 for (int iSlice = (mCfgL.drawSlice == -1 ? 0 : mCfgL.drawRelatedSlices ? (mCfgL.drawSlice % (NSLICES / 4)) : mCfgL.drawSlice) % (NSLICES / 2); iSlice < NSLICES / 2; iSlice += (mCfgL.drawSlice == -1 ? 1 : mCfgL.drawRelatedSlices ? (NSLICES / 4) : NSLICES))
#define LOOP_COLLISION for (int iCol = (mCfgL.showCollision == -1 ? 0 : mCfgL.showCollision); iCol < mNCollissions; iCol += (mCfgL.showCollision == -1 ? 1 : mNCollissions))
#define LOOP_COLLISION_COL(cmd)  \
  LOOP_COLLISION                 \
  {                              \
    if (mCfgL.colorCollisions) { \
      SetCollisionColor(iCol);   \
    }                            \
    cmd;                         \
  }

  if (mCfgL.drawGrid) {
    if (mCfgL.drawTPC) {
      SetColorGrid();
      LOOP_SLICE drawVertices(mGlDLGrid[iSlice], GPUDisplayBackend::LINES);
    }
    if (mCfgL.drawTRD) {
      SetColorGridTRD();
      LOOP_SLICE2 drawVertices(mGlDLGridTRD[iSlice], GPUDisplayBackend::LINES);
    }
  }
  if (mCfgL.drawClusters) {
    if (mCfgL.drawTRD) {
      SetColorTRD();
      mBackend->lineWidthFactor(2);
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDCLUSTER][iCol], GPUDisplayBackend::LINES));
      if (mCfgL.drawFinal && mCfgL.colorClusters) {
        SetColorFinal();
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRDATTACHED][iCol], GPUDisplayBackend::LINES));
      mBackend->lineWidthFactor(1);
    }
    if (mCfgL.drawTOF) {
      SetColorTOF();
      mBackend->pointSizeFactor(2);
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[0][tTOFCLUSTER][0], GPUDisplayBackend::POINTS));
      mBackend->pointSizeFactor(1);
    }
    if (mCfgL.drawITS) {
      SetColorITS();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[0][tITSCLUSTER][0], GPUDisplayBackend::POINTS));
    }
    if (mCfgL.drawTPC) {
      SetColorClusters();
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tCLUSTER][iCol], GPUDisplayBackend::POINTS));

      if (mCfgL.drawInitLinks) {
        if (mCfgL.excludeClusters) {
          goto skip1;
        }
        if (mCfgL.colorClusters) {
          SetColorInitLinks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tINITLINK][iCol], GPUDisplayBackend::POINTS));

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
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tLINK][iCol], GPUDisplayBackend::POINTS));

      if (mCfgL.drawSeeds) {
        if (mCfgL.excludeClusters) {
          goto skip1;
        }
        if (mCfgL.colorClusters) {
          SetColorSeeds();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSEED][iCol], GPUDisplayBackend::POINTS));

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
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tTRACKLET][iCol], GPUDisplayBackend::POINTS));

      if (mCfgL.drawTracks) {
        if (mCfgL.excludeClusters) {
          goto skip2;
        }
        if (mCfgL.colorClusters) {
          SetColorTracks();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tSLICETRACK][iCol], GPUDisplayBackend::POINTS));

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
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tGLOBALTRACK][iCol], GPUDisplayBackend::POINTS));
      SetColorClusters();

      if (mCfgL.drawFinal && mCfgL.propagateTracks < 2) {
        if (mCfgL.excludeClusters) {
          goto skip3;
        }
        if (mCfgL.colorClusters) {
          SetColorFinal();
        }
      }
      LOOP_SLICE LOOP_COLLISION_COL(drawVertices(mGlDLPoints[iSlice][tFINALTRACK][iCol], GPUDisplayBackend::POINTS));
    skip3:;
    }
  }

  if (!mCfgH.clustersOnly && mCfgL.excludeClusters != 1) {
    if (mCfgL.drawTPC) {
      if (mCfgL.drawInitLinks) {
        SetColorInitLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tINITLINK], GPUDisplayBackend::LINES);
      }
      if (mCfgL.drawLinks) {
        SetColorLinks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tLINK], GPUDisplayBackend::LINES);
      }
      if (mCfgL.drawSeeds) {
        SetColorSeeds();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSEED], GPUDisplayBackend::LINE_STRIP);
      }
      if (mCfgL.drawTracklets) {
        SetColorTracklets();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tTRACKLET], GPUDisplayBackend::LINE_STRIP);
      }
      if (mCfgL.drawTracks) {
        SetColorTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tSLICETRACK], GPUDisplayBackend::LINE_STRIP);
      }
      if (mCfgL.drawGlobalTracks) {
        SetColorGlobalTracks();
        LOOP_SLICE drawVertices(mGlDLLines[iSlice][tGLOBALTRACK], GPUDisplayBackend::LINE_STRIP);
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
          drawVertices(mGlDLFinal[iSlice][iCol][0], GPUDisplayBackend::LINE_STRIP);
        }
        if (mCfgL.propagateTracks > 0 && mCfgL.propagateTracks < 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][1], GPUDisplayBackend::LINE_STRIP);
        }
        if (mCfgL.propagateTracks == 2) {
          drawVertices(mGlDLFinal[iSlice][iCol][2], GPUDisplayBackend::LINE_STRIP);
        }
        if (mCfgL.propagateTracks == 3) {
          drawVertices(mGlDLFinal[iSlice][iCol][3], GPUDisplayBackend::LINE_STRIP);
        }
      }
      if (mCfgH.drawTracksAndFilter ? (mCfgH.drawTPCTracks || mCfgH.drawTRDTracks || mCfgH.drawTOFTracks) : mCfgH.drawITSTracks) {
        drawVertices(mGlDLFinalITS, GPUDisplayBackend::LINE_STRIP);
      }
    }
    if (mCfgH.markClusters || mCfgH.markAdjacentClusters || mCfgH.markFakeClusters) {
      if (mCfgH.markFakeClusters) {
        mBackend->pointSizeFactor(3);
      }
      SetColorMarked();
      LOOP_SLICE LOOP_COLLISION drawVertices(mGlDLPoints[iSlice][tMARKED][iCol], GPUDisplayBackend::POINTS);
      if (mCfgH.markFakeClusters) {
        mBackend->pointSizeFactor(1);
      }
    }
  }
}

void GPUDisplay::DrawGLScene_internal(float animateTime, bool renderToMixBuffer) // negative time = no mixing
{
  bool showTimer = false;
  bool doScreenshot = (mRequestScreenshot || mAnimateScreenshot) && animateTime < 0;

  if (animateTime < 0 && (mUpdateEventData || mResetScene || mUpdateVertexLists) && mIOPtrs) {
    disableUnsupportedOptions();
  }
  if (mUpdateEventData || mUpdateVertexLists) {
    mUpdateDrawCommands = 1;
  }

  if (animateTime < 0 && (mUpdateEventData || mResetScene) && mIOPtrs) {
    showTimer = true;
    DrawGLScene_updateEventData();
    mTimerFPS.ResetStart();
    mFramesDoneFPS = 0;
    mFPSScaleadjust = 0;
    mUpdateVertexLists = 1;
    mUpdateEventData = 0;
  }

  hmm_mat4 nextViewMatrix = MY_HMM_IDENTITY;
  float mixSlaveImage = 0.f;
  DrawGLScene_cameraAndAnimation(animateTime, mixSlaveImage, nextViewMatrix);

  // Prepare Event
  if (mUpdateVertexLists && mIOPtrs) {
    size_t totalVertizes = DrawGLScene_updateVertexList();
    if (showTimer) {
      printf("Event visualization time: %'d us (vertices %'lld / %'lld bytes)\n", (int)(mTimerDraw.GetCurrentElapsedTime() * 1000000.), (long long int)totalVertizes, (long long int)(totalVertizes * sizeof(mVertexBuffer[0][0])));
    }
  }

  // Draw Event
  nextViewMatrix = nextViewMatrix * mModelMatrix;
  const float zFar = ((mParam->par.continuousTracking ? (mMaxClusterZ / GL_SCALE_FACTOR) : 8.f) + 50.f) * 2.f;
  const hmm_mat4 proj = HMM_Perspective(mCfgR.fov, (float)mBackend->mRenderWidth / (float)mBackend->mRenderHeight, 0.1f, zFar);
  mBackend->prepareDraw(proj, nextViewMatrix, doScreenshot || mRequestScreenshot, renderToMixBuffer, mixSlaveImage);
  mBackend->pointSizeFactor(1);
  mBackend->lineWidthFactor(1);

  if (mUpdateDrawCommands || mBackend->backendNeedRedraw()) {
    mNDrawCalls = 0;
    DrawGLScene_drawCommands();
  }

  if (mCfgL.drawField) {
    mBackend->drawField();
  }

  mUpdateDrawCommands = mUpdateRenderPipeline = 0;
  mBackend->finishDraw(doScreenshot, renderToMixBuffer, mixSlaveImage);

  if (animateTime < 0) {
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
      showInfo(info);
    }
  }

  mBackend->finishFrame(mRequestScreenshot, renderToMixBuffer, mixSlaveImage);
  if (doScreenshot) {
    mRequestScreenshot = false;
    std::vector<char> pixels = mBackend->getPixels();
    char tmpFileName[48];
    if (mAnimateScreenshot) {
      sprintf(tmpFileName, "mAnimation%d_%05d.bmp", mAnimationExport, mAnimationFrame);
    }
    DoScreenshot(mAnimateScreenshot ? tmpFileName : mScreenshotFile.c_str(), pixels);
  }
}

void GPUDisplay::DoScreenshot(const char* filename, std::vector<char>& pixels, float animateTime)
{
  size_t screenshot_x = mBackend->mScreenWidth * mCfgR.screenshotScaleFactor;
  size_t screenshot_y = mBackend->mScreenHeight * mCfgR.screenshotScaleFactor;
  size_t size = 4 * screenshot_x * screenshot_y;
  if (size != pixels.size()) {
    GPUError("Pixel array of incorrect size obtained");
    filename = nullptr;
  }

  if (filename) {
    FILE* fp = fopen(filename, "w+b");
    if (fp == nullptr) {
      GPUError("Error opening screenshot file %s", filename);
      return;
    }

    BITMAPFILEHEADER bmpFH;
    BITMAPINFOHEADER bmpIH;
    memset(&bmpFH, 0, sizeof(bmpFH));
    memset(&bmpIH, 0, sizeof(bmpIH));

    bmpFH.bfType = 19778; //"BM"
    bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + size;
    bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

    bmpIH.biSize = sizeof(bmpIH);
    bmpIH.biWidth = screenshot_x;
    bmpIH.biHeight = screenshot_y;
    bmpIH.biPlanes = 1;
    bmpIH.biBitCount = 32;
    bmpIH.biCompression = BI_RGB;
    bmpIH.biSizeImage = size;
    bmpIH.biXPelsPerMeter = 5670;
    bmpIH.biYPelsPerMeter = 5670;

    fwrite(&bmpFH, 1, sizeof(bmpFH), fp);
    fwrite(&bmpIH, 1, sizeof(bmpIH), fp);
    fwrite(pixels.data(), 1, size, fp);
    fclose(fp);
  }
}

void GPUDisplay::showInfo(const char* info)
{
  mBackend->prepareText();
  float colorValue = mCfgL.invertColors ? 0.f : 1.f;
  OpenGLPrint(info, 40.f, 20.f + std::max(20, mDrawTextFontSize + 4), colorValue, colorValue, colorValue, 1);
  if (mInfoText2Timer.IsRunning()) {
    if (mInfoText2Timer.GetCurrentElapsedTime() >= 6) {
      mInfoText2Timer.Reset();
    } else {
      OpenGLPrint(mInfoText2, 40.f, 20.f, colorValue, colorValue, colorValue, 6 - mInfoText2Timer.GetCurrentElapsedTime());
    }
  }
  if (mInfoHelpTimer.IsRunning()) {
    if (mInfoHelpTimer.GetCurrentElapsedTime() >= 6) {
      mInfoHelpTimer.Reset();
    } else {
      PrintGLHelpText(colorValue);
    }
  }
  mBackend->finishText();
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
  mFrontend->mNeedUpdate = 1;
  mUpdateEventData = true;
}

void GPUDisplay::WaitForNextEvent() { mSemLockDisplay.Lock(); }

int GPUDisplay::StartDisplay()
{
  if (mFrontend->StartDisplay()) {
    return (1);
  }
  while (mInitResult == 0) {
    Sleep(10);
  }
  return (mInitResult != 1);
}

void GPUDisplay::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
  if (mBackend->mFreetypeInitialized) {
    if (!fromBotton) {
      y = mBackend->mScreenHeight - y;
    }
    float color[4] = {r, g, b, a};
    mBackend->OpenGLPrint(s, x, y, color, 1.0f);
  } else if (mFrontend->mCanDrawText) {
    mFrontend->OpenGLPrint(s, x, y, r, g, b, a, fromBotton);
  }
}
