// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendConfigKeys.inc
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDCONFIGKEYS_INC
#define GPUDISPLAYFRONTENDCONFIGKEYS_INC

#define KEY_DOWN_MAP(character, id) std::tuple((uint8_t)(character), (uint8_t)GPUDisplayFrontend::KEY_MAP_##id) // Key queried as held key: GPUDisplay uses mKeys[mKeyDownMap[KEY_MAP_...]]
#define KEY_LAMBDA(character, obj, value, ...) std::tuple((uint16_t)(character), mFrontend->mConfigRTTI->getRTTIIndex(*this, &std::remove_cvref_t<decltype(obj)>::value), configKeyCallback([this]() { [[maybe_unused]] auto& val = obj.value; __VA_ARGS__ }))

void GPUDisplay::buildConfigKeys()
{
  mFrontend->add_key_binding(
    KEY_LAMBDA('1', mCfgL, drawClusters, val ^= 1;),
    KEY_LAMBDA('2', mCfgL, drawInitLinks, val ^= 1;),
    KEY_LAMBDA('3', mCfgL, drawLinks, val ^= 1;),
    KEY_LAMBDA('4', mCfgL, drawSeeds, val ^= 1;),
    KEY_LAMBDA('5', mCfgL, drawTracklets, val ^= 1;),
    KEY_LAMBDA('6', mCfgL, drawTracks, val ^= 1;),
    KEY_LAMBDA('7', mCfgL, drawExtrapolatedTracks, val ^= 1;),
    KEY_LAMBDA('8', mCfgL, drawFinal, val ^= 1;),
    KEY_LAMBDA('N', mCfgL, colorClusters, val ^= 1; SetInfo("Color coding for seed / trrack attachmend %s", mCfgL.colorClusters ? "enabled" : "disabled");), //
    "Enable display of clusters, preseeds, seeds, starthits, tracklets, tracks, extrapolated tracks, merged tracks / Show assigned clusters in colors",
    "[1] ... [8] / [N]");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_ESCAPE, mCfgC, mDisplayControl, val.store(2); SetInfo("Exiting", 1);),
    KEY_LAMBDA('n', mCfgC, mDisplayControl, val.store(1);),
    KEY_LAMBDA('r', mCfgC, mResetScene, val.store(true); SetInfo("View reset", 1);), //
    KEY_LAMBDA('q', mCfgC, dummy, {
      if (mFrontend->isGUIRunning()) {
        SetInfo("Stopping GUI", 1);
        mFrontend->stopGUI();
      } else {
        SetInfo("Starting GUI", 1);
        mFrontend->startGUI();
      }
    }), //
    "Quit / Next event / Reset Display Settings / Start / Stop Qt GUI");

  mFrontend->add_key_binding(
    KEY_LAMBDA('l', mCfgL, drawSector, {
      if (val >= (mCfgL.drawRelatedSectors ? (NSECTORS / 4 - 1) : (NSECTORS - 1))) {
        val = -1;
        SetInfo("Showing all sectors", 1);
      } else {
        val++;
        SetInfo("Showing sector %d", val);
      }
    }),
    KEY_LAMBDA('k', mCfgL, drawSector, {
      if (val <= -1) {
        val = mCfgL.drawRelatedSectors ? (NSECTORS / 4 - 1) : (NSECTORS - 1);
      } else {
        val--;
      }
      if (val == -1) {
        SetInfo("Showing all sectors", 1);
      } else {
        SetInfo("Showing sector %d", val);
      }
    }),
    KEY_LAMBDA('J', mCfgL, drawRelatedSectors, val ^= 1; SetInfo("Drawing of related sectors %s", mCfgL.drawRelatedSectors ? "enabled" : "disabled");), //
    KEY_LAMBDA(';', mCfgH, xAdd, {
      mCfgH.xAdd += 60;
      mCfgH.zAdd += 60;
      SetInfo("TPC sector separation: %f %f", mCfgH.xAdd, mCfgH.zAdd);
    }),
    KEY_LAMBDA(':', mCfgH, xAdd, {
      mCfgH.xAdd -= 60;
      mCfgH.zAdd -= 60;
      if (mCfgH.zAdd < 0 || mCfgH.xAdd < 0) {
        mCfgH.zAdd = mCfgH.xAdd = 0;
      }
      SetInfo("TPC sector separation: %f %f", mCfgH.xAdd, mCfgH.zAdd);
    }), //
    "Draw single sector (next  / previous sector), draw related sectors (same plane in phi) / Show splitting of TPC in sectors by extruding volume, [:] resets");

  mFrontend->add_key_binding(
    KEY_LAMBDA('y', mCfgC, dummy, {
      if (mAnimateVectors[0].size() > 1) {
        if (mCfgC.mAnimateScreenshot) {
          mAnimationExport++;
        }
        startAnimation();
        SetInfo("Starting Animation", 1);
      } else {
        SetInfo("Insufficient Animation points to start Animation", 1);
      }
    }),
    KEY_LAMBDA('Y', mCfgC, dummy, setAnimationPoint(); SetInfo("Added Animation point (%d points, %6.2f seconds)", (int32_t)mAnimateVectors[0].size(), mAnimateVectors[0].back());), //
    KEY_LAMBDA('\'', mCfgC, dummy, removeAnimationPoint(); SetInfo("Removed Animation point", 1);),                                                                                  //
    KEY_LAMBDA('X', mCfgC, dummy, resetAnimation(); SetInfo("Reset Animation points", 1);),                                                                                          //
    KEY_LAMBDA('M', mCfgL, animationMode, {
      val++;
      if (val == 7) {
        val = 0;
      }
      if (val == 6) {
        SetInfo("Animation mode %d - Centered on origin", val);
      } else {
        SetInfo("Animation mode %d - Position: %s, Direction: %s", val, (val & 2) ? "Spherical (spherical rotation)" : (val & 4) ? "Spherical (Euler angles)" : "Cartesian", (val & 1) ? "Euler angles" : "Quaternion");
      }
    }), //
    "Start Animation, Add / remove Animation point, Reset Points, Cycle animation camera mode (resets)");

  mFrontend->add_key_binding(
    KEY_LAMBDA('#', mCfgL, invertColors, val ^= 1;),
    KEY_LAMBDA('g', mCfgL, drawGrid, val ^= 1; SetInfo("Fast Cluster Search Grid %s", mCfgL.drawGrid ? "shown" : "hidden");),
    KEY_LAMBDA('i', mCfgH, projectXY, val ^= 1; SetInfo("Projection onto xy plane %s", mCfgH.projectXY ? "enabled" : "disabled");), //
    "Invert colors / Draw Grid / Project onto XY-plane");

  mFrontend->add_key_binding(
    KEY_LAMBDA('x', mCfgL, excludeClusters, val = (val + 1) % 3; SetInfo(mCfgL.excludeClusters ? "Clusters of selected category are excluded from display (%d)" : "Clusters are shown", mCfgL.excludeClusters);),
    KEY_LAMBDA('.', mCfgH, hideRejectedTracks, val ^= 1; SetInfo("Rejected tracks are %s", mCfgH.hideRejectedTracks ? "hidden" : "shown");), //
    "Exclude Clusters used in the tracking steps enabled for visualization ([1]-[8]) / Exclude rejected tracks");

  mFrontend->add_key_binding(
    KEY_LAMBDA('c', mCfgH, markClusters, {
      if (val == 0) {
        val = 1;
      } else if (val >= 0x20) {
        val = 0;
      } else {
        val <<= 1;
      }
      SetInfo("Cluster flag highlight mask set to %d (%s)", val,
              val == 0 ? "off" : val == 1 ? "split pad" : val == 2 ? "split time" : val == 4 ? "edge" : val == 8 ? "singlePad" : val == 0x10 ? "reject distance" : "reject error");
    }),
    KEY_LAMBDA('z', mCfgH, markFakeClusters, val ^= 1; SetInfo("Marking fake clusters: %s", mCfgH.markFakeClusters ? "on" : "off");), //
    KEY_LAMBDA('B', mCfgH, markAdjacentClusters, {
      val++;
      if (val == 5) {
        val = 7;
      }
      if (val == 9) {
        val = 15;
      }
      if (val == 17) {
        val = 31;
      }
      if (val == 34) {
        val = 0;
      }
      if (val == 33) {
        SetInfo("Marking protected clusters (%d)", val);
      } else if (val == 32) {
        SetInfo("Marking removable clusters (%d)", val);
      } else {
        SetInfo("Marking adjacent clusters (%d): rejected %s, tube %s, looper leg %s, low Pt %s, high incl %s", val, (val & 1) ? "yes" : " no", (val & 2) ? "yes" : " no", (val & 4) ? "yes" : " no", (val & 8) ? "yes" : " no", (val & 16) ? "yes" : " no");
      }
    }), //
    "Mark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20 / Mark fake attached clusters / Mark clusters attached as adjacent");

  mFrontend->add_key_binding(
    KEY_LAMBDA('L', mCfgL, showCollision, {
      if (val >= mNCollissions - 1) {
        val = -1;
        SetInfo("Showing all collisions", 1);
      } else {
        val++;
        SetInfo("Showing collision %d / %d", val, mNCollissions);
      }
    }),
    KEY_LAMBDA('K', mCfgL, showCollision, {
      if (val <= -1) {
        val = mNCollissions - 1;
      } else {
        val--;
      }
      if (val == -1) {
        SetInfo("Showing all collisions", 1);
      } else {
        SetInfo("Showing collision %d / %d", val, mNCollissions);
      }
    }),
    KEY_LAMBDA('C', mCfgL, colorCollisions, val ^= 1; SetInfo("Color coding of collisions %s", mCfgL.colorCollisions ? "enabled" : "disabled");), //
    "Draw single collisions (next / previous) / Colorcode clusters of different collisions");

  mFrontend->add_key_binding(
    KEY_LAMBDA('v', mCfgH, hideRejectedClusters, val ^= 1; SetInfo("Rejected clusters are %s", mCfgH.hideRejectedClusters ? "hidden" : "shown");), //
    KEY_LAMBDA('j', mCfgH, separateExtrapolatedTracks, {
      if (val) {
        mCfgH.splitCETracks ^= 1;
      }
      val ^= 1;
      SetInfo("Seperated display of tracks propagated to adjacent sectors %s / of CE tracks %s", val ? "enabled" : "disabled", mCfgH.splitCETracks ? "enabled" : "disabled");
    }),
    KEY_LAMBDA('u', mCfgH, trackFilter, val = (val + 1) % (mConfig.filterMacros.size() + 1); SetInfo("Track filter: %s", mCfgH.trackFilter == 0 ? "None" : mConfig.filterMacros[mCfgH.trackFilter - 1].c_str());), //
    "Hide rejected clusters from tracks / Show tracks segments propagated to adjacent sector in different color / splt CE tracks / Cycle through track filter");

  mFrontend->add_key_binding(
    KEY_LAMBDA('E', mCfgL, propagateTracks, {
      val += 1;
      if (val == 4) {
        val = 0;
      }
      const char* infoText[] = {"Hits connected", "Hits connected and propagated to vertex", "Reconstructed track propagated inwards and outwards", "Monte Carlo track"};
      SetInfo("Display of propagated tracks: %s", infoText[val]);
    }),
    KEY_LAMBDA('G', mCfgH, propagateLoopers, val ^= 1; SetInfo("Propagation of loopers %s", mCfgH.propagateLoopers ? "enabled" : "disabled");), //
    "Extrapolate tracks / loopers");

  mFrontend->add_key_binding(
    KEY_LAMBDA('t', mCfgC, mRequestScreenshot, {
      static int32_t nScreenshot = 1;
      char fname[32];
      snprintf(fname, 32, "screenshot%d.bmp", nScreenshot++);
      val = true;
      mScreenshotFile = fname;
      SetInfo("Taking screenshot (%s)", fname);
    }),
    KEY_LAMBDA('T', mCfgC, mAnimateScreenshot, val ^= 1; SetInfo("Writing screenshots from animation set to %d", mCfgC.mAnimateScreenshot);), //
    KEY_LAMBDA('Z', mCfgR, screenshotScaleFactor, {
      val += 1;
      if (val == 5) {
        val = 1;
      }
      SetInfo("Screenshot scaling factor set to %d", val);
    }), //
    "Take Screenshot / Record Animation to pictures / Change screenshot resolution (scaling factor)");

  mFrontend->add_key_binding(
    KEY_LAMBDA('S', mCfgL, smoothPoints, val ^= true; SetInfo("Smoothing of points %s", mCfgL.smoothPoints ? "enabled" : "disabled");), //
    KEY_LAMBDA('A', mCfgL, smoothLines, val ^= true; SetInfo("Smoothing of lines %s", mCfgL.smoothLines ? "enabled" : "disabled");),    //
    KEY_LAMBDA('D', mCfgL, depthBuffer, {
      val ^= true;
      SetInfo("Depth buffer (z-buffer, %u bits) %s", mBackend->DepthBits(), val ? "enabled" : "disabled");
      mBackend->setDepthBuffer();
    }), //
    "Enable or disable smoothing of points / smoothing of lines / depth buffer");

  mFrontend->add_key_binding(
    KEY_LAMBDA('W', mCfgR, drawQualityMSAA, {
      val *= 2;
      if (val < 2) {
        val = 2;
      }
      if (val > 16 || val > mBackend->getMaxMSAA()) {
        val = 0;
      }
      SetInfo("Multisampling anti-aliasing factor set to %d", val);
    }),
    KEY_LAMBDA('U', mCfgR, drawQualityDownsampleFSAA, {
      val++;
      if (val == 1) {
        val = 2;
      }
      if (val == 5) {
        val = 0;
      }
      SetInfo("Downsampling anti-aliasing factor set to %d", val);
    }),
    KEY_LAMBDA('V', mCfgR, drawQualityVSync, val ^= true; SetInfo("VSync: %s", mCfgR.drawQualityVSync ? "enabled" : "disabled");), //
    KEY_LAMBDA('R', mCfgR, fpsLimit, {
      val = (val + 1) % 3;
      if (val == GPUDisplayBackend::FPS_LIMIT_RATE) {
        SetInfo("FPS limit: %d FPS", mCfgR.fpsRate);
      } else if (val == GPUDisplayBackend::FPS_LIMIT_VSYNC) {
        char rate[32] = "unknown rate";
        if (mFrontend->getRefreshRate() > 0.f) {
          snprintf(rate, sizeof(rate), "%.1f Hz", mFrontend->getRefreshRate());
        }
        const int32_t mode = mBackend->fpsLimitMode();
        if (mode == GPUDisplayBackend::FPS_LIMIT_NONE) {
          SetInfo("FPS limit: refresh rate (%s) - VSync disabled, no limit", rate);
        } else if (mode == GPUDisplayBackend::FPS_LIMIT_RATE) {
          SetInfo("FPS limit: refresh rate (%s) - not supported by the backend, fallback to %d FPS", rate, mCfgR.fpsRate);
        } else {
          SetInfo("FPS limit: refresh rate (%s)", rate);
        }
      } else {
        SetInfo("FPS limit: none");
      }
    }), //
    "Toggle anti-aliasing (MSAA at raster level / change downsampling FSAA factor) / toggle VSync / FPS limit (none, refresh rate, fpsRate)");

  mFrontend->add_key_binding(
    KEY_LAMBDA('F', mCfgR, fullScreen, val ^= 1; SetInfo("Toggling full screen (%d)", (int32_t)mCfgR.fullScreen);),    //
    KEY_LAMBDA('_', mCfgR, maximized, val ^= 1; SetInfo("Toggling Maximized window (%d)", (int32_t)mCfgR.maximized);), //
    KEY_LAMBDA('b', mCfgR, fov, {
      if ((val += 5) > 175) {
        val = 5;
      }
      SetInfo("Set FOV to %f", val);
    }),
    KEY_LAMBDA('m', mCfgR, rendererCompatSmoothing, {
      if (displayOpenGLCore && backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
        SetInfo("OpenGL compat profile not available, using core profile", 1);
      } else {
        val ^= 1;
        SetInfo("Using renderer path for %s profile", val ? "compat" : "core (shader)");
      }
    }),
    KEY_LAMBDA('I', mCfgR, useGLIndirectDraw, val ^= true; SetInfo("OpenGL Indirect Draw %s", mCfgR.useGLIndirectDraw ? "enabled" : "disabled");), //
    "Switch mFullScreen / Maximized window / Change FOV (field of view) / Switch between core / compat renderer path (OpenGL profile, Vulkan smooth lines) / Enable / disable GL indirect draw");

  mFrontend->add_key_binding(
    KEY_LAMBDA('o', mCfgC, dummy, {
      FILE* ftmp = fopen("glpos.tmp", "w+b");
      if (ftmp) {
        int32_t retval = fwrite(&mViewMatrix, sizeof(mViewMatrix), 1, ftmp);
        if (retval != 1) {
          GPUError("Error writing position to file");
        } else {
          GPUInfo("Position stored to file");
        }
        fclose(ftmp);
      } else {
        GPUError("Error opening file");
      }
      SetInfo("Camera position stored to file", 1);
    }),
    KEY_LAMBDA('p', mCfgC, dummy, {
      FILE* ftmp = fopen("glpos.tmp", "rb");
      if (ftmp) {
        int32_t retval = fread(&mViewMatrix, 1, sizeof(mViewMatrix), ftmp);
        if (retval == sizeof(mViewMatrix)) {
          GPUInfo("Position read from file");
        } else {
          GPUError("Error reading position from file");
        }
        fclose(ftmp);
      } else {
        GPUError("Error opening file");
      }
      SetInfo("Camera position loaded from file", 1);
    }),
    KEY_LAMBDA('O', mCfgC, dummy, {
      FILE* ftmp = fopen("glanimation.tmp", "w+b");
      if (ftmp) {
        fwrite(&mCfgL, sizeof(mCfgL), 1, ftmp);
        int32_t size = mAnimateVectors[0].size();
        fwrite(&size, sizeof(size), 1, ftmp);
        for (int32_t i = 0; i < 9; i++) {
          fwrite(mAnimateVectors[i].data(), sizeof(mAnimateVectors[i][0]), size, ftmp);
        }
        fwrite(mAnimateConfig.data(), sizeof(mAnimateConfig[0]), size, ftmp);
        fclose(ftmp);
      } else {
        GPUError("Error opening file");
      }
      SetInfo("Animation path stored to file %s", "glanimation.tmp");
    }),
    KEY_LAMBDA('P', mCfgC, dummy, {
      FILE* ftmp = fopen("glanimation.tmp", "rb");
      if (ftmp) {
        int32_t retval = fread(&mCfgL, sizeof(mCfgL), 1, ftmp);
        int32_t size;
        retval += fread(&size, sizeof(size), 1, ftmp);
        for (int32_t i = 0; i < 9; i++) {
          mAnimateVectors[i].resize(size);
          retval += fread(mAnimateVectors[i].data(), sizeof(mAnimateVectors[i][0]), size, ftmp);
        }
        mAnimateConfig.resize(size);
        retval += fread(mAnimateConfig.data(), sizeof(mAnimateConfig[0]), size, ftmp);
        (void)retval; // disable unused warning
        fclose(ftmp);
        updateConfig();
      } else {
        GPUError("Error opening file");
      }
      SetInfo("Animation path loaded from file %s", "glanimation.tmp");
    }), //
    "Save / restore current camera position / Animation path");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_F1, mCfgL, drawTPC, val ^= 1; SetInfo("Showing TPC Clusters: %d", (int32_t)mCfgL.drawTPC);),
    KEY_LAMBDA(mFrontend->KEY_F2, mCfgL, drawTRD, val ^= 1; SetInfo("Showing TRD Tracklets: %d", (int32_t)mCfgL.drawTRD);),
    KEY_LAMBDA(mFrontend->KEY_F3, mCfgL, drawTOF, val ^= 1; SetInfo("Showing TOF Hits: %d", (int32_t)mCfgL.drawTOF);),
    KEY_LAMBDA(mFrontend->KEY_F4, mCfgL, drawITS, val ^= 1; SetInfo("Showing ITS Clusters: %d", (int32_t)mCfgL.drawITS);), //
    "Enable / disable drawing of TPC / TRD / TOF / ITS");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_F1 + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgH, drawTPCTracks, val ^= 1; SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int32_t)mCfgH.drawTPCTracks, (int32_t)mCfgH.drawTRDTracks, (int32_t)mCfgH.drawTOFTracks, (int32_t)mCfgH.drawITSTracks);),
    KEY_LAMBDA(mFrontend->KEY_F2 + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgH, drawTRDTracks, val ^= 1; SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int32_t)mCfgH.drawTPCTracks, (int32_t)mCfgH.drawTRDTracks, (int32_t)mCfgH.drawTOFTracks, (int32_t)mCfgH.drawITSTracks);),
    KEY_LAMBDA(mFrontend->KEY_F3 + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgH, drawTOFTracks, val ^= 1; SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int32_t)mCfgH.drawTPCTracks, (int32_t)mCfgH.drawTRDTracks, (int32_t)mCfgH.drawTOFTracks, (int32_t)mCfgH.drawITSTracks);),
    KEY_LAMBDA(mFrontend->KEY_F4 + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgH, drawITSTracks, val ^= 1; SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int32_t)mCfgH.drawTPCTracks, (int32_t)mCfgH.drawTRDTracks, (int32_t)mCfgH.drawTOFTracks, (int32_t)mCfgH.drawITSTracks);),
    KEY_LAMBDA(mFrontend->KEY_F12 + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgH, drawTracksAndFilter, val ^= 1; SetInfo("Track filter: %s", mCfgH.drawTracksAndFilter ? "AND" : "OR");), //
    "Enable / disable track detector filter / SHIFT F12: Switch track detector filter between AND and OR mode",
    "[SHIFT] + [F1] to [F4] / [F12]");

  mFrontend->add_key_binding(
    KEY_LAMBDA('h', mCfgC, mPrintInfoTextAlways, {
      if (val) {
        val = false;
        SetInfo("Showing help text disabled", 1);
      } else if (mInfoHelpTimer.IsRunning()) {
        val = true;
        mInfoHelpTimer.Reset();
        SetInfo("Showing help text until disabled", 1);
      } else {
        PrintHelp();
        SetInfo("Showing help text", 1);
      }
    }),
    KEY_LAMBDA('H', mCfgC, mPrintInfoText, {
      val += 1;
      val &= 3;
      SetInfo("Info text display - console: %s, onscreen %s", (val & 2) ? "enabled" : "disabled", (val & 1) ? "enabled" : "disabled");
    }), //
    "Print Help / Show info texts");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP('w', FORWARD),
    KEY_DOWN_MAP('s', BACKWARD),
    KEY_DOWN_MAP('a', STRAFE_LEFT),
    KEY_DOWN_MAP('d', STRAFE_RIGHT), //
    "Zoom / Strafe Left and Right");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_PAGEUP, STRAFE_UP),
    KEY_DOWN_MAP(mFrontend->KEY_PAGEDOWN, STRAFE_DOWN), //
    "Strafe up / down");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP('e', ROTATE_LEFT),
    KEY_DOWN_MAP('f', ROTATE_RIGHT), //
    "Rotate left / right");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_LEFT, ROTATE_YAW_LEFT),
    KEY_DOWN_MAP(mFrontend->KEY_RIGHT, ROTATE_YAW_RIGHT),
    KEY_DOWN_MAP(mFrontend->KEY_UP, ROTATE_PITCH_UP),
    KEY_DOWN_MAP(mFrontend->KEY_DOWN, ROTATE_PITCH_DOWN), //
    "Turn camera left / right / up / down");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP('+', INCREASE_WIDTH),
    KEY_DOWN_MAP('-', DECREASE_WIDTH), //
    "Increase / decrease point size (Hold SHIFT for lines)");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_ALT, FOCUS_ORIGIN),
    KEY_DOWN_MAP(mFrontend->KEY_CTRL, FOCUS_ORIENT_UP), //
    KEY_LAMBDA(mFrontend->KEY_ALT + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgR, camLookOrigin, {
      val ^= 1;
      mCfgR.cameraMode = val + 2 * mCfgR.camYUp;
      SetInfo("Camera locked on origin: %s", val ? "enabled" : "disabled");
    }),
    KEY_LAMBDA(mFrontend->KEY_CTRL + mFrontend->SPECIAL_KEYS_SHIFTED, mCfgR, camYUp, {
      val ^= 1;
      mCfgR.cameraMode = mCfgR.camLookOrigin + 2 * val;
      SetInfo("Camera locked on y-axis facing upwards: %s", val ? "enabled" : "disabled");
    }),
    KEY_LAMBDA(mFrontend->KEY_ENTER, mCfgR, cameraMode, {
      val++;
      if (val == 4) {
        val = 0;
      }
      mCfgR.camLookOrigin = val & 1;
      mCfgR.camYUp = val & 2;
      const char* modeText[] = {"Descent (free movement)", "Focus locked on origin (y-axis forced upwards)", "Spectator (y-axis forced upwards)", "Focus locked on origin (with free rotation)"};
      SetInfo("Camera mode %d: %s", val, modeText[val]);
    }),
    "Focus camera on origin / orient y-axis upwards (combine with [SHIFT] to lock) / Cycle through modes", //
    "[ALT] / [CTRL] / [ENTER]");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_RCTRL, ROTATE_MODEL),
    KEY_DOWN_MAP(mFrontend->KEY_RALT, ROTATE_AROUND_Z), //
    "Rotate model instead of camera / rotate TPC around beamline");

  mFrontend->add_key_binding(
    KEY_LAMBDA('>', mCfgC, mAnimationChangeConfig, val ^= 1; SetInfo("Interpolating visualization settings during Animation %s", mCfgC.mAnimationChangeConfig ? "enabled" : "disabled");),
    KEY_DOWN_MAP('<', CHANGE_ANIMATION_INTERVAL), //
    "Toggle config interpolation during Animation / change Animation interval (via movement)");

  mFrontend->mConfigRTTI->helpTextEntries.emplace_back("[MOUSE 1] / [MOUSE 2] / [MOUSE 1+2] / [SHIFT]", "Look around / Strafe camera / Zoom / Rotate / Slow Zoom / Move / Rotate");
}

#endif // GPUDISPLAYFRONTENDCONFIGKEYS_INC
