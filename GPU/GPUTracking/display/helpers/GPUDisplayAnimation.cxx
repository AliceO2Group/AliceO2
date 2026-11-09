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
#include "frontend/GPUDisplayInfo.inc"
#include "GPUCommonMath.h"
#include "HandMadeMath.h"

#include <algorithm>
#include <numeric>

using namespace o2::gpu;

const std::array<std::string, 7> GPUDisplay::ANIMATION_MODE_NAMES_SHORT = {"Cartesian, quaternion", "Cartesian, Euler angles", "Spherical Rotation, quaternion", "Spherical Rotation, Euler angles", "Spherical, quaternion", "Spherical, Euler angles", "Spherical, centered on origin"};
const std::array<std::string, 7> GPUDisplay::ANIMATION_MODE_NAMES_LONG = {"Position: Cartesian, Direction: Quaternion", "Position: Cartesian, Direction: Euler angles", "Position: Spherical (spherical rotation), Direction: Quaternion", "Position: Spherical (spherical rotation), Direction: Euler angles", "Position: Spherical (Euler angles), Direction: Quaternion", "Position: Spherical (Euler angles), Direction: Euler angles", "Position: Spherical (Euler angles), Direction: Centered on origin"};

constexpr hmm_mat4 MY_HMM_FROM(float (&v)[16]) { return {{{v[0], v[1], v[2], v[3]}, {v[4], v[5], v[6], v[7]}, {v[8], v[9], v[10], v[11]}, {v[12], v[13], v[14], v[15]}}}; }

static hmm_mat4 cameraViewMatrix(const float (&xyz)[3], const float (&angles)[3]) // Inverse of calcXYZ
{
  const float toDeg = 180.f / CAMath::Pi();
  return HMM_Rotate(-angles[2] * toDeg, {{0, 0, 1}}) * HMM_Rotate(-angles[0] * toDeg, {{1, 0, 0}}) * HMM_Rotate(angles[1] * toDeg, {{0, 1, 0}}) * HMM_Translate({{-xyz[0], -xyz[1], -xyz[2]}});
}

void GPUDisplay::setAnimationPoint()
{
  StopAnimation();
  auto& times = mCfgC.animationPoints.values;
  times.emplace_back(times.size() ? times.back() + ((int32_t)(mCfgO.animationDelay * 20)) / 20.f : 0.f);
  mAnimationVectors.push_back({{mXYZ[0], mXYZ[1], mXYZ[2], mAngle[0], mAngle[1], mAngle[2]}, mCfgL});
  mCfgC.animationPoints.selected = -1;
}

void GPUDisplay::animationPointValues(const animationVector& p, int32_t mode, const float* last, float (&vals)[8])
{
  const float xyz[3] = {p.v[0], p.v[1], p.v[2]}, angles[3] = {p.v[3], p.v[4], p.v[5]};
  if (mode & 4) // Spherical
  {
    float rxy = sqrtf(xyz[0] * xyz[0] + xyz[2] * xyz[2]);
    float anglePhi = atan2f(xyz[0], xyz[2]);
    float angleTheta = atan2f(xyz[1], rxy);
    if (last) {
      mAnimationCloseAngle(anglePhi, last[1]);
      mAnimationCloseAngle(angleTheta, last[2]);
    }
    vals[0] = 0;
    vals[1] = anglePhi;
    vals[2] = angleTheta;
  } else {
    for (int32_t i = 0; i < 3; i++) {
      vals[i] = xyz[i];
    }
    // Cartesian
  }
  vals[3] = sqrtf(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
  if (mode & 1) // Euler-angles
  {
    for (int32_t i = 0; i < 3; i++) {
      float newangle = angles[i];
      if (last) {
        mAnimationCloseAngle(newangle, last[i + 4]);
      }
      vals[i + 4] = newangle;
    }
    vals[7] = 0;
  } else { // Quaternions
    float v[4];
    const hmm_mat4 viewMatrix = cameraViewMatrix(xyz, angles);
    createQuaternionFromMatrix(v, &viewMatrix.Elements[0][0]);
    if (last) {
      mAnimateCloseQuaternion(v, last[4], last[5], last[6], last[7]);
    }
    for (int32_t i = 0; i < 4; i++) {
      vals[i + 4] = v[i];
    }
  }
}

static hmm_mat4 lookAtMatrix(const float (&eye)[3], const float (&target)[3]) // HMM_LookAt with y up, z up when looking along y (the cross product with y would be 0)
{
  const float d[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
  const bool alongY = d[0] * d[0] + d[2] * d[2] <= 1e-10f * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  return HMM_LookAt({{eye[0], eye[1], eye[2]}}, {{target[0], target[1], target[2]}}, alongY ? hmm_vec3{{0, 0, 1}} : hmm_vec3{{0, 1, 0}});
}

void GPUDisplay::setCamera(const float (&xyz)[3], const float (&angles)[3])
{
  *mViewMatrix = cameraViewMatrix(xyz, angles);
  calcXYZ(&mViewMatrix->Elements[0][0]);
}

bool GPUDisplay::lookAt(const float (&target)[3])
{
  if (target[0] == mXYZ[0] && target[1] == mXYZ[1] && target[2] == mXYZ[2]) {
    return false;
  }
  *mViewMatrix = lookAtMatrix(mXYZ, target);
  calcXYZ(&mViewMatrix->Elements[0][0]);
  return true;
}

void GPUDisplay::rotateCamera(const float (&angles)[3])
{
  *mViewMatrix = cameraRotation(angles) * *mViewMatrix;
  calcXYZ(&mViewMatrix->Elements[0][0]);
}

hmm_mat4 GPUDisplay::cameraRotation(const float (&angles)[3]) const
{
  return HMM_Rotate(angles[0], {{0, 1, 0}}) * HMM_Rotate(angles[1] * mYFactor, {{1, 0, 0}}) * HMM_Rotate(angles[2] * mYFactor, {{0, 0, 1}});
}

void GPUDisplay::addAnimationPoint(const float (&xyz)[3], const float (&angles)[3])
{
  const hmm_mat4 oldViewMatrix = *mViewMatrix;
  setCamera(xyz, angles);
  setAnimationPoint(); // From mXYZ / mAngle set by setCamera, as for the current camera
  *mViewMatrix = oldViewMatrix;
  calcXYZ(&mViewMatrix->Elements[0][0]);
}

void GPUDisplay::resetAnimation()
{
  mAnimationVectors.clear();
  mCfgC.animationPoints.values.clear();
  mCfgC.animationPoints.selected = -1;
  StopAnimation();
}

void GPUDisplay::removeAnimationPoint(int32_t i)
{
  StopAnimation();
  if (i < 0) {
    i = mCfgC.animationPoints.selected >= 0 ? mCfgC.animationPoints.selected : (int32_t)mAnimationVectors.size() - 1;
  }
  if (i < 0 || i >= (int32_t)mAnimationVectors.size()) {
    return;
  }
  auto& times = mCfgC.animationPoints.values;
  mAnimationVectors.erase(mAnimationVectors.begin() + i);
  times.erase(times.begin() + i);
  if (i == 0 && times.size()) { // The first point stays at time 0
    const float t0 = times[0];
    for (float& t : times) {
      t -= t0;
    }
  }
  mCfgC.animationPoints.selected = -1;
}

void GPUDisplay::selectAnimationPoint(int32_t i)
{
  StopAnimation();
  if (i < 0 || i >= (int32_t)mAnimationVectors.size()) {
    mCfgC.animationPoints.selected = -1;
    return;
  }
  mCfgC.animationPoints.selected = i;
  mCfgL = mAnimationVectors[i].cfgL;
  const auto& v = mAnimationVectors[i].v;
  setCamera({v[0], v[1], v[2]}, {v[3], v[4], v[5]});
}

void GPUDisplay::StopAnimation()
{
  if (!mAnimate) {
    return;
  }
  mAnimate = mCfgO.animateScreenshot = 0;
  mAnimationMixBase = -1; // Camera and settings stay as they are
  mUpdateDrawCommands = true;
  SetInfo("Animation stopped", 1);
}

void GPUDisplay::setAnimationPointsFromGUI(const GPUDisplayMultiSliderValue& points)
{
  if (mAnimate || points.values.size() != mAnimationVectors.size()) { // Not while the animation runs; the GUI may not yet know a changed number of points
    return;
  }
  std::vector<uint32_t> order(points.values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return points.values[a] < points.values[b]; });
  for (uint32_t i = 1; i < order.size(); i++) {
    if (!(points.values[order[i]] > points.values[order[i - 1]])) { // Points must have different times, the GUI prevents this
      return;
    }
  }
  vecpod<animationVector> sorted(order.size());
  vecpod<float> sortedTimes(order.size());
  int32_t selected = -1;
  for (uint32_t i = 0; i < order.size(); i++) {
    sorted[i] = mAnimationVectors[order[i]];
    sortedTimes[i] = points.values[order[i]] - points.values[order[0]]; // The first point at time 0
    selected = (int32_t)order[i] == points.selected ? i : selected;
  }
  const bool selectionChanged = points.selected != mCfgC.animationPoints.selected;
  mAnimationVectors = std::move(sorted);
  mCfgC.animationPoints.values = std::move(sortedTimes);
  if (selectionChanged) {
    selectAnimationPoint(selected);
  } else {
    mCfgC.animationPoints.selected = selected;
  }
}

void GPUDisplay::updateAnimationPointsGUI()
{
  auto& gui = mCfgC.animationPoints;
  gui.progress = mAnimate ? gui.progress : 0;                                                             // Set by animateCamera while the animation runs
  gui.max = (gui.values.size() ? gui.values.back() : 0.) + ((int32_t)(mCfgO.animationDelay * 20)) / 20.f; // Room for the next point
}

void GPUDisplay::runAnimation()
{
  if (mAnimationVectors.size() > 1) {
    if (mCfgO.animateScreenshot) {
      mAnimationExport++;
    }
    startAnimation();
    SetInfo("Starting Animation", 1);
  } else {
    SetInfo("Insufficient Animation points to start Animation", 1);
  }
}

void GPUDisplay::startAnimation()
{
  mAnimationSplines.resize(ANIMATION_MODE_NAMES_SHORT.size());
  vecpod<float> y[8];
  for (auto& yi : y) {
    yi.resize(mAnimationVectors.size());
  }
  for (uint32_t mode = 0; mode < mAnimationSplines.size(); mode++) { // Splines for all modes, points may have different modes
    float vals[8], previous[8];
    for (uint32_t k = 0; k < mAnimationVectors.size(); k++) {
      animationPointValues(mAnimationVectors[k], mode, k ? previous : nullptr, vals); // Continuous to the previous point in time order
      for (int32_t i = 0; i < 8; i++) {
        y[i][k] = previous[i] = vals[i];
      }
    }
    for (int32_t i = 0; i < 8; i++) {
      mAnimationSplines[mode][i].create(mCfgC.animationPoints.values, y[i]);
    }
  }
  mAnimationTimer.ResetStart();
  mAnimationFrame = 0;
  mAnimate = 1;
  mCfgC.animationPoints.selected = -1; // The animation changes mCfgL, which would otherwise be stored in the selected point when it ends
  mAnimationLastBase = 0;
  mAnimationMixBase = -1;
}

int32_t GPUDisplay::animateCamera(bool suspended, float& animateTime, float& mixSlaveImage, hmm_mat4& nextViewMatrix)
{
  float time = animateTime;
  if (mAnimate && time < 0) {
    if (mCfgO.animateScreenshot) {
      time = mAnimationFrame / 30.f;
    } else {
      time = mAnimationTimer.GetCurrentElapsedTime();
    }

    float maxTime = mCfgC.animationPoints.values.back();
    mCfgC.animationPoints.progress = std::lround(std::clamp(time / maxTime, 0.f, 1.f) * 65535);
    if (time >= maxTime) {
      time = maxTime;
      mAnimate = mCfgO.animateScreenshot = 0;
      SetInfo("Animation finished. (%1.2f seconds, %d frames)", time, mAnimationFrame);
    } else {
      SetInfo("Running mAnimation: time %1.2f/%1.2f, frames %d", time, maxTime, mAnimationFrame);
    }
    mAnimationFrame++;
  }
  if (!mAnimate) {
    return 0;
  }
  if (mCfgO.animationChangeConfig && animateTime < 0) {
    int32_t base = 0;
    const auto& times = mCfgC.animationPoints.values;
    int32_t k = mAnimationVectors.size() - 1;
    while (base < k && time > times[base]) {
      base++;
    }
    if (base > mAnimationLastBase + 1) {
      mAnimationLastBase = base - 1;
    }

    const bool mixing = base != mAnimationLastBase && times[mAnimationLastBase] != times[base] && memcmp(&mAnimationVectors[base].cfgL, &mAnimationVectors[mAnimationLastBase].cfgL, sizeof(mCfgL));
    if ((mixing ? base : -1) != mAnimationMixBase) { // Blended configs changed: cached (Vulkan) command buffers must be re-recorded, otherwise they are kept while mixing
      mAnimationMixBase = mixing ? base : -1;
      mUpdateDrawCommands = true;
    }
    if (mixing) {
      mixSlaveImage = 1.f - (time - times[mAnimationLastBase]) / (times[base] - times[mAnimationLastBase]);
      if (mixSlaveImage > 0) {
        mCfgL = mAnimationVectors[mAnimationLastBase].cfgL;
        updateConfig();
        DrawGLScene_internal(suspended, time, true);
      }
    }

    if (memcmp(&mAnimationVectors[base].cfgL, &mCfgL, sizeof(mCfgL))) {
      mCfgL = mAnimationVectors[base].cfgL;
      updateConfig();
      mUpdateDrawCommands |= !mixing; // mOldCfg is taken after drawing, so checkAndApplyConfigChanges does not see this change: cached (Vulkan) command buffers would keep the previous settings; while mixing, a change of mAnimationMixBase re-records them
    }
  }

  const auto& times = mCfgC.animationPoints.values;
  uint32_t k = 0; // Interval [times[k], times[k + 1]] of the time
  while (k + 2 < times.size() && time > times[k + 1]) {
    k++;
  }
  const int32_t maxMode = mAnimationSplines.size() - 1; // Modes from a loaded file are not checked
  const int32_t modeA = std::clamp(mAnimationVectors[k].cfgL.animationMode, 0, maxMode), modeB = std::clamp(mAnimationVectors[k + 1].cfgL.animationMode, 0, maxMode);
  auto cameraOfMode = [&](int32_t mode, hmm_mat4& viewMatrix) {
    float vals[8];
    for (int32_t i = 0; i < 8; i++) {
      vals[i] = mAnimationSplines[mode][i].evaluate(time);
    }
    animationViewMatrix(vals, mode, viewMatrix);
  };
  if (modeA == modeB) {
    cameraOfMode(modeA, nextViewMatrix);
  } else { // Modes of the points differ: blend the cameras of both modes, position linearly, rotation as quaternion, with sin^2 for a smooth start and end
    hmm_mat4 viewA = nextViewMatrix, viewB = nextViewMatrix;
    cameraOfMode(modeA, viewA);
    cameraOfMode(modeB, viewB);
    const float s = sinf(std::clamp((time - times[k]) / (times[k + 1] - times[k]), 0.f, 1.f) * CAMath::Pi() / 2), w = s * s;
    float qA[4], qB[4], vals[8];
    createQuaternionFromMatrix(qA, &viewA.Elements[0][0]);
    createQuaternionFromMatrix(qB, &viewB.Elements[0][0]);
    const float sign = qA[0] * qB[0] + qA[1] * qB[1] + qA[2] * qB[2] + qA[3] * qB[3] < 0 ? -1.f : 1.f;                                                                               // Shorter way
    auto position = [](const hmm_mat4& m, int32_t i) { return -(m.Elements[i][0] * m.Elements[3][0] + m.Elements[i][1] * m.Elements[3][1] + m.Elements[i][2] * m.Elements[3][2]); }; // Camera position, as calcXYZ
    for (int32_t i = 0; i < 3; i++) {
      vals[i] = (1 - w) * position(viewA, i) + w * position(viewB, i);
    }
    vals[3] = 0;
    for (int32_t i = 0; i < 4; i++) {
      vals[4 + i] = (1 - w) * qA[i] + w * sign * qB[i]; // Normalized by animationViewMatrix
    }
    animationViewMatrix(vals, 0, nextViewMatrix); // Mode 0: Cartesian position, quaternion
  }
  return 1;
}

void GPUDisplay::animationViewMatrix(float (&vals)[8], int32_t mode, hmm_mat4& nextViewMatrix) const
{
  if (mode != 6) {
    if (mode & 1) // Rotation from euler angles, same order as setCamera (inverse of calcXYZ)
    {
      nextViewMatrix = nextViewMatrix * HMM_Rotate(-vals[6] * 180.f / CAMath::Pi(), {{0, 0, 1}}) * HMM_Rotate(-vals[4] * 180.f / CAMath::Pi(), {{1, 0, 0}}) * HMM_Rotate(vals[5] * 180.f / CAMath::Pi(), {{0, 1, 0}});
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
  if (mode & 4) // Compute cartesian translation from sperical coordinates (euler angles)
  {
    const float r = vals[3], phi = vals[1], theta = vals[2];
    vals[2] = r * cosf(phi) * cosf(theta);
    vals[0] = r * sinf(phi) * cosf(theta);
    vals[1] = r * sinf(theta);
  } else if (mode & 2) { // Scale cartesion translation to interpolated radius
    float r = sqrtf(vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2]);
    if (fabsf(r) < 0.0001f) {
      r = 1;
    }
    r = vals[3] / r;
    for (int32_t i = 0; i < 3; i++) {
      vals[i] *= r;
    }
  }
  if (mode == 6) {
    nextViewMatrix = lookAtMatrix({vals[0], vals[1], vals[2]}, {0.f, 0.f, 0.f});
  } else {
    nextViewMatrix = nextViewMatrix * HMM_Translate({{-vals[0], -vals[1], -vals[2]}});
  }
}
