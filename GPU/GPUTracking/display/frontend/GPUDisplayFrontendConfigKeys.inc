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

#define KEY_DOWN_MAP(character, id) GPUDisplayFrontend::GPUDisplayConfiguration::keyDownMapEntry(character, GPUDisplayFrontend::KEY_MAP_##id) // Key queried as held key: GPUDisplay uses mKeys[mKeyDownMap[KEY_MAP_...]]
#define KEY_LAMBDA(character, actionName, obj, value, ...) GPUDisplayFrontend::GPUDisplayConfiguration::keyLambdaEntry(character, actionName, mFrontend->mConfiguration->getRTTIIndex(*this, &std::remove_cvref_t<decltype(obj)>::value), configKeyCallback([this]() { [[maybe_unused]] auto& val = obj.value; __VA_ARGS__ }))
#define KEY_ACTION(character, actionName, ...) GPUDisplayFrontend::GPUDisplayConfiguration::keyLambdaEntry(character, actionName, GPUDisplayFrontend::GPUDisplayConfiguration::INVALID_INDEX, configKeyCallback([this]() { __VA_ARGS__ })) // Key function without config option

void GPUDisplay::buildConfigKeys()
{
  mFrontend->mConfiguration->helpTextEntries.emplace_back("[MOUSE 1] / [MOUSE 2] / [MOUSE 1+2]", "Look around / Strafe camera / Zoom / Rotate", std::vector<int32_t>());
  mFrontend->mConfiguration->helpTextEntries.emplace_back("[SHIFT]", "Slow Mouse Zoom / Move / Rotate", std::vector<int32_t>());

  mFrontend->add_key_binding(
    KEY_DOWN_MAP('w', FORWARD),
    KEY_DOWN_MAP('s', BACKWARD),
    KEY_DOWN_MAP('a', STRAFE_LEFT),
    KEY_DOWN_MAP('d', STRAFE_RIGHT), //
    "Zoom / Strafe Left and Right");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_PAGEUP, STRAFE_UP),
    KEY_DOWN_MAP(mFrontend->KEY_PAGEDOWN, STRAFE_DOWN),
    KEY_DOWN_MAP('e', ROTATE_LEFT),
    KEY_DOWN_MAP('f', ROTATE_RIGHT), //
    "Strafe up / down / Rotate left / right");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_ESCAPE, "QUIT", mCfgC, displayControl, val.store(2);),
    KEY_LAMBDA('n', "NEXT_EVENT", mCfgC, displayControl, val.store(1);),
    KEY_LAMBDA('r', "RESET_SCENE", mCfgC, resetScene, val.store(true);), //
    KEY_ACTION('q', "TOGGLE_GUI", {
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
    KEY_DOWN_MAP(mFrontend->KEY_LEFT, ROTATE_YAW_LEFT),
    KEY_DOWN_MAP(mFrontend->KEY_RIGHT, ROTATE_YAW_RIGHT),
    KEY_DOWN_MAP(mFrontend->KEY_UP, ROTATE_PITCH_UP),
    KEY_DOWN_MAP(mFrontend->KEY_DOWN, ROTATE_PITCH_DOWN), //
    "Turn camera left / right / up / down");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_ALT, FOCUS_ORIGIN),
    KEY_DOWN_MAP(mFrontend->KEY_CTRL, FOCUS_ORIENT_UP), //
    KEY_LAMBDA(mFrontend->KEY_ALT + mFrontend->SPECIAL_KEYS_SHIFTED, "CAM_LOOK_ORIGIN", mCfgR, camLookOrigin, {
      val ^= 1;
      mCfgR.cameraMode = val + 2 * mCfgR.camYUp;
    }),
    KEY_LAMBDA(mFrontend->KEY_CTRL + mFrontend->SPECIAL_KEYS_SHIFTED, "CAM_YUP", mCfgR, camYUp, {
      val ^= 1;
      mCfgR.cameraMode = mCfgR.camLookOrigin + 2 * val;
    }),
    KEY_LAMBDA(mFrontend->KEY_ENTER, "CAMERA_MODE", mCfgR, cameraMode, {
      val++;
      if (val == 4) {
        val = 0;
      }
      mCfgR.camLookOrigin = val & 1;
      mCfgR.camYUp = val & 2;
    }),
    "Focus camera on origin / orient y-axis upwards (combine with [SHIFT] to lock) / Cycle through modes", //
    "[ALT] / [CTRL] / [ENTER]");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP(mFrontend->KEY_RCTRL, ROTATE_MODEL),
    KEY_DOWN_MAP(mFrontend->KEY_RALT, ROTATE_AROUND_Z), //
    "Rotate model instead of camera / rotate TPC around beamline");

  mFrontend->add_key_binding(
    KEY_LAMBDA('F', "FULL_SCREEN", mCfgR, fullScreen, val ^= 1;), //
    KEY_LAMBDA('_', "MAXIMIZED", mCfgR, maximized, val ^= 1;),    //
    KEY_LAMBDA('b', "FOV", mCfgR, fov, {
      if ((val += 5) > 175) {
        val = 5;
      }
    }),
    KEY_LAMBDA('I', "USE_GLINDIRECT_DRAW", mCfgR, useGLIndirectDraw, val ^= true;), //
    "Switch Full Screen / Maximized window / Change FOV (field of view) / Enable / disable GL indirect draw");

  mFrontend->add_key_binding(
    KEY_LAMBDA('S', "SMOOTH_POINTS", mCfgL, smoothPoints, val ^= true;), //
    KEY_LAMBDA('A', "SMOOTH_LINES", mCfgL, smoothLines, val ^= true;),   //
    KEY_LAMBDA('D', "DEPTH_BUFFER", mCfgL, depthBuffer, val ^= true;),   //
    "Enable or disable smoothing of points / smoothing of lines / depth buffer");

  mFrontend->add_key_binding(
    KEY_LAMBDA('W', "DRAW_QUALITY_MSAA", mCfgR, drawQualityMSAA, {
      val *= 2;
      if (val < 2) {
        val = 2;
      }
      if (val > 16 || val > mBackend->getMaxMSAA()) {
        val = 0;
      }
    }),
    KEY_LAMBDA('U', "DRAW_QUALITY_DOWNSAMPLE_FSAA", mCfgR, drawQualityDownsampleFSAA, {
      val++;
      if (val == 1) {
        val = 2;
      }
      if (val == 5) {
        val = 0;
      }
    }),
    KEY_LAMBDA('V', "DRAW_QUALITY_VSYNC", mCfgR, drawQualityVSync, val ^= true;), //
    KEY_LAMBDA('R', "FPS_LIMIT", mCfgR, fpsLimit, val = (val + 1) % 3;),          //
    "Toggle anti-aliasing (MSAA at raster level / change downsampling FSAA factor) / toggle VSync / FPS limit (none, refresh rate, fpsRate)");

  mFrontend->add_key_binding(
    KEY_LAMBDA('#', "INVERT_COLORS", mCfgL, invertColors, val ^= 1;),
    KEY_LAMBDA('g', "DRAW_GRID", mCfgL, drawGrid, val ^= 1;),
    KEY_LAMBDA('i', "PROJECT_XY", mCfgH, projectXY, val ^= 1;), //
    "Invert colors / Draw Grid / Project onto XY-plane");

  mFrontend->add_key_binding(
    KEY_DOWN_MAP('+', INCREASE_WIDTH),
    KEY_DOWN_MAP('-', DECREASE_WIDTH), //
    "Increase / decrease point size (Hold SHIFT for lines)");

  mFrontend->add_key_binding(
    KEY_LAMBDA('l', "DRAW_SECTOR_NEXT", mCfgL, drawSector, {
      if (val >= (mCfgL.drawRelatedSectors ? (NSECTORS / 4 - 1) : (NSECTORS - 1))) {
        val = -1;
      } else {
        val++;
      }
    }),
    KEY_LAMBDA('k', "DRAW_SECTOR_PREVIOUS", mCfgL, drawSector, {
      if (val <= -1) {
        val = mCfgL.drawRelatedSectors ? (NSECTORS / 4 - 1) : (NSECTORS - 1);
      } else {
        val--;
      }
    }),
    KEY_LAMBDA('J', "DRAW_RELATED_SECTORS", mCfgL, drawRelatedSectors, val ^= 1;), //
    KEY_LAMBDA(';', "XZ_ADD", mCfgH, xzAdd, val = std::min(val + 60, 300.f);),     // Up to the option maximum, reset with the scene
    "Draw single sector (next  / previous sector), draw related sectors (same plane in phi) / Show splitting of TPC in sectors by extruding volume");

  mFrontend->add_key_binding(
    KEY_LAMBDA('x', "EXCLUDE_CLUSTERS", mCfgL, excludeClusters, val = (val + 1) % 3;),
    KEY_LAMBDA('.', "HIDE_REJECTED_TRACKS", mCfgH, hideRejectedTracks, val ^= 1;), //
    "Exclude Clusters used in the tracking steps enabled for visualization ([1]-[8]) / Exclude rejected tracks");

  mFrontend->add_key_binding(
    KEY_LAMBDA('c', "MARK_CLUSTERS", mCfgH, markClusters, {
      if (val == 0) {
        val = 1;
      } else if (val >= 0x20) {
        val = 0;
      } else {
        val <<= 1;
      }
    }),
    KEY_LAMBDA('z', "MARK_FAKE_CLUSTERS", mCfgH, markFakeClusters, val ^= 1;), //
    KEY_LAMBDA('B', "MARK_ADJACENT_CLUSTERS", mCfgH, markAdjacentClusters, {
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
    }), //
    "Mark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20 / Mark fake attached clusters / Mark adjacent clusters");

  mFrontend->add_key_binding(
    KEY_LAMBDA('v', "HIDE_REJECTED_CLUSTERS", mCfgH, hideRejectedClusters, val ^= 1;),
    KEY_LAMBDA('j', "SEPARATE_EXTRAPOLATED_TRACKS", mCfgH, separateExtrapolatedTracks, val ^= 1;),
    KEY_LAMBDA('m', "SPLIT_CETRACKS", mCfgH, splitCETracks, val ^= 1;),
    KEY_LAMBDA('u', "TRACK_FILTER", mCfgH, trackFilter, val = (val + 1) % (mConfig.filterMacros.size() + 1);), //
    "Hide rejected clusters from tracks / Show tracks segments propagated to adjacent sector in different color / splt CE tracks / Cycle through track filter");

  mFrontend->add_key_binding(
    KEY_LAMBDA('E', "PROPAGATE_TRACKS", mCfgL, propagateTracks, {
      val += 1;
      if (val == 4) {
        val = 0;
      }
    }),
    KEY_LAMBDA('G', "PROPAGATE_LOOPERS", mCfgH, propagateLoopers, val ^= 1;), //
    "Extrapolate tracks / loopers");

  mFrontend->add_key_binding(
    KEY_LAMBDA('L', "SHOW_COLLISION_NEXT", mCfgL, showCollision, {
      if (val >= mNCollissions - 1) {
        val = -1;
      } else {
        val++;
      }
    }),
    KEY_LAMBDA('K', "SHOW_COLLISION_PREVIOUS", mCfgL, showCollision, {
      if (val <= -1) {
        val = mNCollissions - 1;
      } else {
        val--;
      }
    }),
    KEY_LAMBDA('C', "COLOR_COLLISIONS", mCfgL, colorCollisions, val ^= 1;), //
    "Draw single collisions (next / previous) / Colorcode clusters of different collisions");

  mFrontend->add_key_binding(
    KEY_LAMBDA('t', "REQUEST_SCREENSHOT", mCfgC, requestScreenshot, takeScreenshot();),
    KEY_LAMBDA('T', "ANIMATE_SCREENSHOT", mCfgO, animateScreenshot, val ^= 1;), //
    KEY_LAMBDA('Z', "SCREENSHOT_SCALE_FACTOR", mCfgR, screenshotScaleFactor, {
      val += 1;
      if (val > 4) {
        val = 1;
      }
    }), //
    "Take Screenshot / Record Animation to pictures / Change screenshot resolution (scaling factor)");

  mFrontend->add_key_binding(
    KEY_ACTION('y', "START_ANIMATION", runAnimation();),
    KEY_ACTION('Y', "ADD_ANIMATION_POINT", setAnimationPoint(); SetInfo("Added Animation point (%d points, %6.2f seconds)", (int32_t)mAnimationVectors.size(), mCfgC.animationPoints.values.back());), //
    KEY_ACTION('\'', "REMOVE_ANIMATION_POINT", removeAnimationPoint(); SetInfo("Removed Animation point", 1);),                                                                                        //
    KEY_ACTION('X', "RESET_ANIMATION", resetAnimation(); SetInfo("Reset Animation points", 1);),                                                                                                       //
    KEY_LAMBDA('M', "ANIMATION_MODE", mCfgL, animationMode, {
      val++;
      if (val == 7) {
        val = 0;
      }
    }), //
    "Start Animation, Add / remove Animation point, Reset Points, Cycle animation camera mode");

  mFrontend->add_key_binding(
    KEY_LAMBDA('>', "ANIMATION_CHANGE_CONFIG", mCfgO, animationChangeConfig, val ^= 1;),
    KEY_DOWN_MAP('<', CHANGE_ANIMATION_INTERVAL), //
    "Toggle config interpolation during Animation / change Animation interval (via movement)");

  mFrontend->add_key_binding(
    KEY_ACTION('o', "SAVE_CAMERA", SaveGLPos();),
    KEY_ACTION('p', "LOAD_CAMERA", LoadGLPos();),
    KEY_ACTION('O', "SAVE_ANIMATION", SaveGLAnimation();),
    KEY_ACTION('P', "LOAD_ANIMATION", LoadGLAnimation();),
    "Save / restore current camera position / Animation path");

  mFrontend->add_key_binding(
    KEY_LAMBDA('1', "DRAW_CLUSTERS", mCfgL, drawClusters, val ^= 1;),
    KEY_LAMBDA('2', "DRAW_INIT_LINKS", mCfgL, drawInitLinks, val ^= 1;),
    KEY_LAMBDA('3', "DRAW_LINKS", mCfgL, drawLinks, val ^= 1;),
    KEY_LAMBDA('4', "DRAW_SEEDS", mCfgL, drawSeeds, val ^= 1;),
    KEY_LAMBDA('5', "DRAW_TRACKLETS", mCfgL, drawTracklets, val ^= 1;),
    KEY_LAMBDA('6', "DRAW_TRACKS", mCfgL, drawTracks, val ^= 1;),
    KEY_LAMBDA('7', "DRAW_EXTRAPOLATED_TRACKS", mCfgL, drawExtrapolatedTracks, val ^= 1;),
    KEY_LAMBDA('8', "DRAW_FINAL", mCfgL, drawFinal, val ^= 1;),
    KEY_LAMBDA('N', "COLOR_CLUSTERS", mCfgL, colorClusters, val ^= 1;), //
    "Enable display of clusters, preseeds, seeds, starthits, tracklets, tracks, extrapolated tracks, merged tracks / Show assigned clusters in colors",
    "[1] ... [8] / [N]");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_F1, "DRAW_TPC", mCfgL, drawTPC, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F2, "DRAW_TRD", mCfgL, drawTRD, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F3, "DRAW_TOF", mCfgL, drawTOF, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F4, "DRAW_ITS", mCfgL, drawITS, val ^= 1;), //
    "Enable / disable drawing of TPC / TRD / TOF / ITS");

  mFrontend->add_key_binding(
    KEY_LAMBDA(mFrontend->KEY_F1 + mFrontend->SPECIAL_KEYS_SHIFTED, "DRAW_TPCTRACKS", mCfgH, drawTPCTracks, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F2 + mFrontend->SPECIAL_KEYS_SHIFTED, "DRAW_TRDTRACKS", mCfgH, drawTRDTracks, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F3 + mFrontend->SPECIAL_KEYS_SHIFTED, "DRAW_TOFTRACKS", mCfgH, drawTOFTracks, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F4 + mFrontend->SPECIAL_KEYS_SHIFTED, "DRAW_ITSTRACKS", mCfgH, drawITSTracks, val ^= 1;),
    KEY_LAMBDA(mFrontend->KEY_F12 + mFrontend->SPECIAL_KEYS_SHIFTED, "DRAW_TRACKS_AND_FILTER", mCfgH, drawTracksAndFilter, val ^= 1;), //
    "Enable / disable track detector filter / SHIFT F12: Switch track detector filter between AND and OR mode",
    "[SHIFT] + [F1] to [F4] / [F12]");

  mFrontend->add_key_binding(
    KEY_LAMBDA('h', "PRINT_INFO_TEXT_ALWAYS", mCfgO, printInfoTextAlways, {
      if (val) {
        val = false;
      } else if (mInfoHelpTimer.IsRunning()) {
        val = true;
        mInfoHelpTimer.Reset();
      } else {
        PrintHelp();
      }
    }),
    KEY_LAMBDA('H', "PRINT_INFO_TEXT", mCfgO, printInfoText, {
      val += 1;
      val &= 3;
    }), //
    "Print Help / Show info texts");

  mFrontend->add_key_binding(
    KEY_LAMBDA('^', "SHOW_CONSOLE", mCfgC, showConsole, val ^= 1;), //
    "Show / hide the console");
}

#endif // GPUDISPLAYFRONTENDCONFIGKEYS_INC
