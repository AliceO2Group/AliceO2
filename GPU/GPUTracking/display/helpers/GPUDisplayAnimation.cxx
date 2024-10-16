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

/// \file GPUDisplayAnimation.cxx
/// \author David Rohr

#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

constexpr hmm_mat4 MY_HMM_FROM(float (&v)[16]) { return {{{v[0], v[1], v[2], v[3]}, {v[4], v[5], v[6], v[7]}, {v[8], v[9], v[10], v[11]}, {v[12], v[13], v[14], v[15]}}}; }

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
    for (int32_t i = 0; i < 3; i++) {
      mAnimateVectors[i + 1].emplace_back(mXYZ[i]);
    }
    // Cartesian
  }
  float r = sqrtf(mXYZ[0] * mXYZ[0] + mXYZ[1] * mXYZ[1] + mXYZ[2] * mXYZ[2]);
  mAnimateVectors[4].emplace_back(r);
  if (mCfgL.animationMode & 1) // Euler-angles
  {
    for (int32_t i = 0; i < 3; i++) {
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
    for (int32_t i = 0; i < 4; i++) {
      mAnimateVectors[i + 5].emplace_back(v[i]);
    }
  }
  float delay = 0.f;
  if (mAnimateVectors[0].size()) {
    delay = mAnimateVectors[0].back() + ((int32_t)(mAnimationDelay * 20)) / 20.f;
  }
  mAnimateVectors[0].emplace_back(delay);
  mAnimateConfig.emplace_back(mCfgL);
}

void GPUDisplay::resetAnimation()
{
  for (int32_t i = 0; i < 9; i++) {
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
  for (int32_t i = 0; i < 9; i++) {
    mAnimateVectors[i].pop_back();
  }
  mAnimateConfig.pop_back();
}

void GPUDisplay::startAnimation()
{
  for (int32_t i = 0; i < 8; i++) {
    mAnimationSplines[i].create(mAnimateVectors[0], mAnimateVectors[i + 1]);
  }
  mAnimationTimer.ResetStart();
  mAnimationFrame = 0;
  mAnimate = 1;
  mAnimationLastBase = 0;
}

int32_t GPUDisplay::animateCamera(float& animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix)
{
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
  if (!mAnimate) {
    return 0;
  }
  float vals[8];
  for (int32_t i = 0; i < 8; i++) {
    vals[i] = mAnimationSplines[i].evaluate(time);
  }
  if (mAnimationChangeConfig && animateTime < 0) {
    int32_t base = 0;
    int32_t k = mAnimateVectors[0].size() - 1;
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
      nextViewMatrix = nextViewMatrix * HMM_Rotate(-vals[4] * 180.f / CAMath::Pi(), {{1, 0, 0}}) * HMM_Rotate(vals[5] * 180.f / CAMath::Pi(), {{0, 1, 0}}) * HMM_Rotate(-vals[6] * 180.f / CAMath::Pi(), {{0, 0, 1}});
    } else { // Rotation from quaternion
      const float mag = sqrtf(vals[4] * vals[4] + vals[5] * vals[5] + vals[6] * vals[6] + vals[7] * vals[7]);
      if (mag < 0.0001f) {
        vals[7] = 1;
      } else {
        for (int32_t i = 0; i < 4; i++) {
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
    if (fabsf(r) < 0.0001f) {
      r = 1;
    }
    r = vals[3] / r;
    for (int32_t i = 0; i < 3; i++) {
      vals[i] *= r;
    }
  }
  if (mCfgL.animationMode == 6) {
    nextViewMatrix = HMM_LookAt({{vals[0], vals[1], vals[2]}}, {{0, 0, 0}}, {{0, 1, 0}});
  } else {
    nextViewMatrix = nextViewMatrix * HMM_Translate({{-vals[0], -vals[1], -vals[2]}});
  }
  return 1;
}
