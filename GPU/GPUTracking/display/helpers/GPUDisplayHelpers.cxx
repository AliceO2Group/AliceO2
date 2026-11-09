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

/// \file GPUDisplayHelpers.cxx
/// \author David Rohr

#include "GPUDisplay.h"
#include "frontend/GPUDisplayInfo.inc"
#include "HandMadeMath.h"
#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUParam.h"

#ifndef _WIN32
#include "bitmapfile.h"
#endif

#include "oneapi/tbb.h"

using namespace o2::gpu;

int32_t GPUDisplay::getNumThreads()
{
  if (mChain) {
    return mChain->GetProcessingSettings().nHostThreads;
  } else {
    return tbb::info::default_concurrency();
  }
}

void GPUDisplay::updateOptions()
{
  if (mCfgH.splitCETracks == -1 && mParam) {
    mCfgH.splitCETracks = mParam->continuousMaxTimeBin != 0;
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
    mCfgL.excludeClusters = mCfgL.drawInitLinks = mCfgL.drawLinks = mCfgL.drawSeeds = mCfgL.drawTracklets = mCfgL.drawTracks = mCfgL.drawExtrapolatedTracks = 0;
  }
}

void GPUDisplay::takeScreenshot()
{
  static int32_t nScreenshot = 1;
  mCfgC.requestScreenshot = true;
  mScreenshotFile = mCfgO.mscreenShotName + std::to_string(nScreenshot++) + ".bmp";
}

void GPUDisplay::SaveGLPos()
{
  const char* file = mCfgO.glPosFile.c_str();
  FILE* ftmp = fopen(file, "w+b");
  if (!ftmp) {
    SetInfo("Error opening file %s", file);
    return;
  }
  bool ok = fwrite(mViewMatrix.get(), sizeof(*mViewMatrix), 1, ftmp) == 1;
  ok = fclose(ftmp) == 0 && ok;
  SetInfo(ok ? "Camera position stored to file %s" : "Error writing camera position to file %s", file);
}

void GPUDisplay::LoadGLPos()
{
  const char* file = mCfgO.glPosFile.c_str();
  FILE* ftmp = fopen(file, "rb");
  if (!ftmp) {
    SetInfo("Error opening file %s", file);
    return;
  }
  hmm_mat4 matrix;
  const bool ok = fread(&matrix, sizeof(matrix), 1, ftmp) == 1;
  fclose(ftmp);
  if (ok) {
    *mViewMatrix = matrix;
  }
  SetInfo(ok ? "Camera position loaded from file %s" : "Error reading camera position from file %s", file);
}

void GPUDisplay::SaveGLAnimation()
{
  const char* file = mCfgO.glAnimationFile.c_str();
  FILE* ftmp = fopen(file, "w+b");
  if (!ftmp) {
    SetInfo("Error opening file %s", file);
    return;
  }
  const int32_t size = mAnimationVectors.size();
  bool ok = fwrite(&mCfgL, sizeof(mCfgL), 1, ftmp) == 1 && fwrite(&size, sizeof(size), 1, ftmp) == 1;
  ok = ok && fwrite(mCfgC.animationPoints.values.data(), sizeof(float), size, ftmp) == (size_t)size;
  ok = ok && fwrite(mAnimationVectors.data(), sizeof(mAnimationVectors[0]), size, ftmp) == (size_t)size;
  ok = fclose(ftmp) == 0 && ok;
  SetInfo(ok ? "Animation path stored to file %s" : "Error writing animation path to file %s", file);
}

void GPUDisplay::LoadGLAnimation()
{
  StopAnimation();
  const char* file = mCfgO.glAnimationFile.c_str();
  FILE* ftmp = fopen(file, "rb");
  if (!ftmp) {
    SetInfo("Error opening file %s", file);
    return;
  }
  GPUSettingsDisplayLight cfg;
  int32_t size = 0;
  bool ok = fread(&cfg, sizeof(cfg), 1, ftmp) == 1 && fread(&size, sizeof(size), 1, ftmp) == 1 && size >= 0;
  if (ok) { // Check the file size before allocating memory or changing anything
    const long start = ftell(ftmp);
    fseek(ftmp, 0, SEEK_END);
    ok = (size_t)ftell(ftmp) == start + (size_t)size * (sizeof(float) + sizeof(animationVector));
    fseek(ftmp, start, SEEK_SET);
  }
  if (ok) {
    mCfgC.animationPoints.values.resize(size);
    mAnimationVectors.resize(size);
    ok = ok && fread(mCfgC.animationPoints.values.data(), sizeof(float), size, ftmp) == (size_t)size;
    ok = ok && fread(mAnimationVectors.data(), sizeof(mAnimationVectors[0]), size, ftmp) == (size_t)size;
    mCfgC.animationPoints.selected = -1;
    if (ok) {
      mCfgL = cfg;
      updateConfig();
    } else {
      resetAnimation(); // Read error after the size check: no incomplete path
    }
  }
  fclose(ftmp);
  SetInfo(ok ? "Animation path loaded from file %s" : "Error reading animation path from file %s", file);
}

std::string GPUDisplay::trackFilterName() const
{
  if (mCfgH.trackFilter <= 0) {
    return "";
  }
  return (uint32_t)mCfgH.trackFilter <= mConfig.filterMacros.size() ? "displayTrackFilter/" + mConfig.filterMacros[mCfgH.trackFilter - 1] : "Custom Macro";
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
  if (mCfgO.printInfoTextGUI) {
    mGUIInfoStatus = info;
  }
  if (!(mCfgO.printInfoText & 1)) {
    return;
  }
  float colorValue = mCfgL.invertColors ? 0.f : 1.f;
  GLPrintText(info, 40.f, 20.f + std::max(20, mDrawTextFontSize + 4), colorValue, colorValue, colorValue, 1);
  if (mInfoText2Timer.IsRunning()) {
    const float time = mInfoText2Timer.GetCurrentElapsedTime(), shown = mConfig.internal.setInfoTime, transition = mConfig.internal.setInfoTransition;
    if (time >= shown + transition) {
      mInfoText2Timer.Reset();
    } else {
      GLPrintText(mInfoText2, 40.f, 20.f, colorValue, colorValue, colorValue, time <= shown ? 1.f : 1.f - (time - shown) / transition); // Opaque, then fading out
    }
  }
  if (mInfoHelpTimer.IsRunning() || mCfgO.printInfoTextAlways) {
    if (mInfoHelpTimer.GetCurrentElapsedTime() >= 6) {
      mInfoHelpTimer.Reset();
    } else {
      PrintGLHelpText(colorValue);
    }
  }
}

float GPUDisplay::consoleVerticalFraction()
{
  const bool running = mConsoleTimer.IsRunning();
  if (!mCfgC.showConsole && !running) {
    return 0.f;
  }
  const double time = mConsoleTimer.GetCurrentElapsedTime();
  if (running && !mCfgC.showConsole && time >= mConfig.internal.consoleDelay) { // Keeps running while open, for the blinking cursor
    mConsoleTimer.Stop();
  }
  const float progress = running && mConfig.internal.consoleDelay > 0.f ? std::min(1.f, (float)(time / mConfig.internal.consoleDelay)) : 1.f;
  return mConfig.internal.consoleVertical * (mCfgC.showConsole ? progress : 1.f - progress);
}

void GPUDisplay::GLPrintText(const char* s, float x, float y, float r, float g, float b, float a, uint32_t nCharacters, bool fromBotton)
{
  if (mBackend->mFreetypeInitialized) {
    const float scale = mFrontend->getDisplayScale();
    x *= scale;
    y *= scale;
    if (!fromBotton) {
      y = mBackend->mScreenHeight - y;
    }
    float color[4] = {r, g, b, a};
    mBackend->GLPrintText(s, x, y, color, scale / mBackend->mFontRasterScale, nCharacters);
  }
}
