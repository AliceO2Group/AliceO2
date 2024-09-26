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

#include "GPUDisplay.h"

#include "GPUTPCDef.h"

#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include "../utils/linux_helpers.h"
#endif
#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "GPUChainTracking.h"
#include "GPUQA.h"
#include "GPUTPCSliceData.h"
#include "GPUChainTracking.h"
#include "GPUTPCTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUO2DataTypes.h"
#include "utils/qconfig.h"

#include "frontend/GPUDisplayFrontend.h"
#include "backend/GPUDisplayBackend.h"
#include "helpers/GPUDisplayColors.inc"

constexpr hmm_mat4 MY_HMM_IDENTITY = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};

using namespace GPUCA_NAMESPACE::gpu;

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
  mOverlayTFClusters.resize(mNCollissions);
  mOverlayTFClusters[collision][slice] = cluster;
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
  } else if (!mOverlayTFClusters.size()) {
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

void GPUDisplay::DrawGLScene_cameraAndAnimation(float animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix)
{
  int mMouseWheelTmp = mFrontend->mMouseWheel;
  mFrontend->mMouseWheel = 0;
  bool lookOrigin = mCfgR.camLookOrigin ^ mFrontend->mKeys[mFrontend->KEY_ALT];
  bool yUp = mCfgR.camYUp ^ mFrontend->mKeys[mFrontend->KEY_CTRL] ^ lookOrigin;
  bool rotateModel = mFrontend->mKeys[mFrontend->KEY_RCTRL] || mFrontend->mKeys[mFrontend->KEY_RALT];
  bool rotateModelTPC = mFrontend->mKeys[mFrontend->KEY_RALT];

  // Calculate rotation / translation scaling factors
  float scalefactor = mFrontend->mKeys[mFrontend->KEY_SHIFT] ? 0.2f : 1.0f;
  float rotatescalefactor = scalefactor * 0.25f;
  if (mCfgL.drawSlice != -1) {
    scalefactor *= 0.2f;
  }
  float sqrdist = sqrtf(sqrtf(mViewMatrixP[12] * mViewMatrixP[12] + mViewMatrixP[13] * mViewMatrixP[13] + mViewMatrixP[14] * mViewMatrixP[14]) * GL_SCALE_FACTOR) * 0.8f;
  if (sqrdist < 0.2f) {
    sqrdist = 0.2f;
  }
  if (sqrdist > 5.f) {
    sqrdist = 5.f;
  }
  scalefactor *= sqrdist;

  // Perform new rotation / translation / animation

  if (animateCamera(animateTime, mixSlaveImage, nextViewMatrix)) {
    // Do nothing else
  } else if (mResetScene) {
    const float initialZpos = mCfgH.projectXY ? 16 : (mParam->par.continuousTracking ? (mMaxClusterZ * GL_SCALE_FACTOR + 8) : 8);
    nextViewMatrix = nextViewMatrix * HMM_Translate({{0, 0, -initialZpos}});
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
    float moveZ = scalefactor * ((float)mMouseWheelTmp / 150 + (float)(mFrontend->mKeys[(unsigned char)'W'] - mFrontend->mKeys[(unsigned char)'S']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]) * 0.2f * mFPSScale);
    float moveY = scalefactor * ((float)(mFrontend->mKeys[mFrontend->KEY_PAGEDOWN] - mFrontend->mKeys[mFrontend->KEY_PAGEUP]) * 0.2f * mFPSScale);
    float moveX = scalefactor * ((float)(mFrontend->mKeys[(unsigned char)'A'] - mFrontend->mKeys[(unsigned char)'D']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]) * 0.2f * mFPSScale);
    float rotRoll = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[(unsigned char)'E'] - mFrontend->mKeys[(unsigned char)'F']) * (!mFrontend->mKeys[mFrontend->KEY_SHIFT]);
    float rotYaw = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[mFrontend->KEY_RIGHT] - mFrontend->mKeys[mFrontend->KEY_LEFT]);
    float rotPitch = rotatescalefactor * mFPSScale * 2 * (mFrontend->mKeys[mFrontend->KEY_DOWN] - mFrontend->mKeys[mFrontend->KEY_UP]);

    float mouseScale = 1920.f / std::max<float>(1920.f, mBackend->mScreenWidth);
    if (mFrontend->mMouseDnR && mFrontend->mMouseDn) {
      moveZ += -scalefactor * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY) / 4;
      rotRoll += -rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvX - (float)mFrontend->mMouseDnX);
    } else if (mFrontend->mMouseDnR) {
      moveX += scalefactor * 0.5f * mouseScale * ((float)mFrontend->mMouseDnX - (float)mFrontend->mMouseMvX) / 4;
      moveY += scalefactor * 0.5f * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY) / 4;
    } else if (mFrontend->mMouseDn) {
      rotYaw += rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvX - (float)mFrontend->mMouseDnX);
      rotPitch += rotatescalefactor * mouseScale * ((float)mFrontend->mMouseMvY - (float)mFrontend->mMouseDnY);
    }

    if (mFrontend->mKeys[(unsigned char)'<'] && !mFrontend->mKeysShift[(unsigned char)'<']) {
      mAnimationDelay += moveX;
      if (mAnimationDelay < 0.05f) {
        mAnimationDelay = 0.05f;
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
          mAngleRollOrigin = yUp ? 0.f : -mAngle[2];
        }
        mAngleRollOrigin += rotRoll;
        nextViewMatrix = nextViewMatrix * HMM_Rotate(mAngleRollOrigin, {{0, 0, 1}});
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
      const float max_theta = CAMath::Pi() / 2 - 0.01f;
      if (theta >= max_theta) {
        theta = max_theta;
      } else if (theta <= -max_theta) {
        theta = -max_theta;
      }
      if (moveZ >= r - 0.1f) {
        moveZ = r - 0.1f;
      }
      r -= moveZ;
      r2 = r * cosf(theta);
      mXYZ[0] = r2 * cosf(phi);
      mXYZ[2] = r2 * sinf(phi);
      mXYZ[1] = r * sinf(theta);

      if (yUp) {
        nextViewMatrix = MY_HMM_IDENTITY;
      }
      nextViewMatrix = nextViewMatrix * HMM_LookAt({{mXYZ[0], mXYZ[1], mXYZ[2]}}, {{0, 0, 0}}, {{0, 1, 0}});
    } else {
      nextViewMatrix = nextViewMatrix * HMM_Translate({{moveX, moveY * mYFactor, moveZ}});
      if (!rotateModel) {
        if (rotYaw != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotYaw, {{0, 1, 0}});
        }
        if (rotPitch != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotPitch * mYFactor, {{1, 0, 0}});
        }
        if (!yUp && rotRoll != 0.f) {
          nextViewMatrix = nextViewMatrix * HMM_Rotate(rotRoll * mYFactor, {{0, 0, 1}});
        }
      }
      nextViewMatrix = nextViewMatrix * mViewMatrix; // Apply previous translation / rotation
      if (yUp) {
        calcXYZ(&nextViewMatrix.Elements[0][0]);
        nextViewMatrix = HMM_Rotate(mAngle[2] * 180.f / CAMath::Pi(), {{0, 0, 1}}) * nextViewMatrix;
      }
      if (rotateModel) {
        if (rotYaw != 0.f) {
          mModelMatrix = HMM_Rotate(rotYaw, {{nextViewMatrix.Elements[0][1], nextViewMatrix.Elements[1][1], nextViewMatrix.Elements[2][1]}}) * mModelMatrix;
        }
        if (rotPitch != 0.f) {
          mModelMatrix = HMM_Rotate(rotPitch, {{nextViewMatrix.Elements[0][0], nextViewMatrix.Elements[1][0], nextViewMatrix.Elements[2][0]}}) * mModelMatrix;
        }
        if (rotRoll != 0.f) {
          if (rotateModelTPC) {
            mModelMatrix = HMM_Rotate(-rotRoll, {{0, 0, 1}}) * mModelMatrix;
          } else {
            mModelMatrix = HMM_Rotate(-rotRoll, {{nextViewMatrix.Elements[0][2], nextViewMatrix.Elements[1][2], nextViewMatrix.Elements[2][2]}}) * mModelMatrix;
          }
        }
      }
    }

    // Graphichs Options
    float minSize = 0.4f / (mCfgR.drawQualityDownsampleFSAA > 1 ? mCfgR.drawQualityDownsampleFSAA : 1);
    int deltaLine = mFrontend->mKeys[(unsigned char)'+'] * mFrontend->mKeysShift[(unsigned char)'+'] - mFrontend->mKeys[(unsigned char)'-'] * mFrontend->mKeysShift[(unsigned char)'-'];
    mCfgL.lineWidth += (float)deltaLine * mFPSScale * 0.02f * mCfgL.lineWidth;
    if (mCfgL.lineWidth < minSize) {
      mCfgL.lineWidth = minSize;
    }
    if (deltaLine) {
      SetInfo("%s line width: %f", deltaLine > 0 ? "Increasing" : "Decreasing", mCfgL.lineWidth);
      mUpdateDrawCommands = 1;
    }
    minSize *= 2;
    int deltaPoint = mFrontend->mKeys[(unsigned char)'+'] * (!mFrontend->mKeysShift[(unsigned char)'+']) - mFrontend->mKeys[(unsigned char)'-'] * (!mFrontend->mKeysShift[(unsigned char)'-']);
    mCfgL.pointSize += (float)deltaPoint * mFPSScale * 0.02f * mCfgL.pointSize;
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
  const float zFar = ((mParam->par.continuousTracking ? (mMaxClusterZ * GL_SCALE_FACTOR) : 8.f) + 50.f) * 2.f;
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
    snprintf(info, 1024,
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
      snprintf(tmpFileName, 48, "mAnimation%d_%05d.bmp", mAnimationExport, mAnimationFrame);
    }
    DoScreenshot(mAnimateScreenshot ? tmpFileName : mScreenshotFile.c_str(), pixels);
  }
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
