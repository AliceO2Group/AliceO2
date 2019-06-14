// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayKeys.cpp
/// \author David Rohr

#include "GPUDisplay.h"
#ifdef BUILD_EVENT_DISPLAY

using namespace GPUCA_NAMESPACE::gpu;

const char* HelpText[] = {
  "[n] / [SPACE]                 Next event",
  "[q] / [Q] / [ESC]             Quit",
  "[r]                           Reset Display Settings",
  "[l] / [k] / [J]               Draw single slice (next  / previous slice), draw related slices (same plane in phi)",
  "[;] / [:]                     Show splitting of TPC in slices by extruding volume, [:] resets",
  "[#]                           Invert colors",
  "[y] / [Y] / ['] / [X] / [M]   Start Animation, Add / remove Animation point, Reset, Cycle mode",
  "[>] / [<]                     Toggle config interpolation during Animation / change Animation interval (via movement)",
  "[g]                           Draw Grid",
  "[i]                           Project onto XY-plane",
  "[x]                           Exclude Clusters used in the tracking steps enabled for visualization ([1]-[8])",
  "[.]                           Exclude rejected tracks",
  "[c]                           Mark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20",
  "[B]                           Mark clusters attached as adjacent",
  "[L] / [K]                     Draw single collisions (next / previous)",
  "[C]                           Colorcode clusters of different collisions",
  "[v]                           Hide rejected clusters from tracks",
  "[b]                           Hide all clusters not belonging or related to matched tracks",
  "[j]                           Show global tracks as additional segments of final tracks",
  "[E] / [G]                     Extrapolate tracks / loopers",
  "[t] / [T]                     Take Screenshot / Record Animation to pictures",
  "[Z]                           Change screenshot resolution (scaling factor)",
  "[S] / [A] / [D]               Enable or disable smoothing of points / smoothing of lines / depth buffer",
  "[W] / [U] / [V]               Toggle anti-aliasing (MSAA at raster level / change downsampling FSAA facot / toggle VSync",
  "[F] / [_] / [R]               Switch mFullScreen / Maximized window / FPS rate limiter",
  "[I]                           Enable / disable GL indirect draw",
  "[o] / [p] / [O] / [P]         Save / restore current camera position / Animation path",
  "[h]                           Print Help",
  "[H]                           Show info texts",
  "[w] / [s] / [a] / [d]         Zoom / Strafe Left and Right",
  "[pgup] / [pgdn]               Strafe Up and Down",
  "[e] / [f]                     Rotate",
  "[+] / [-]                     Make points thicker / fainter (Hold SHIFT for lines)",
  "[MOUSE 1]                     Look around",
  "[MOUSE 2]                     Shift c3amera",
  "[MOUSE 1+2]                   Zoom / Rotate",
  "[SHIFT]                       Slow Zoom / Move / Rotate",
  "[ALT] / [CTRL] / [m]          Focus camera on origin / orient y-axis upwards (combine with [SHIFT] to lock) / Cycle through modes",
  "[1] ... [8] / [N]             Enable display of clusters, preseeds, seeds, starthits, tracklets, tracks, global tracks, merged tracks / Show assigned clusters in colors"
  "[F1] / [F2]                   Enable / disable drawing of TPC / TRD"
  // FREE: u z
};

void GPUDisplay::PrintHelp()
{
  mInfoHelpTimer.ResetStart();
  for (unsigned int i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    printf("%s\n", HelpText[i]);
  }
}

void GPUDisplay::HandleKeyRelease(unsigned char key)
{
  if (key == mBackend->KEY_ENTER || key == 'n') {
    mBackend->mDisplayControl = 1;
    SetInfo("Showing next event", 1);
  } else if (key == 27 || key == 'q' || key == 'Q' || key == mBackend->KEY_ESCAPE) {
    mBackend->mDisplayControl = 2;
    SetInfo("Exiting", 1);
  } else if (key == 'r') {
    mResetScene = 1;
    SetInfo("View reset", 1);
  } else if (key == mBackend->KEY_ALT && mBackend->mKeysShift[mBackend->KEY_ALT]) {
    mCamLookOrigin ^= 1;
    mCameraMode = mCamLookOrigin + 2 * mCamYUp;
    SetInfo("Camera locked on origin: %s", mCamLookOrigin ? "enabled" : "disabled");
  } else if (key == mBackend->KEY_CTRL && mBackend->mKeysShift[mBackend->KEY_CTRL]) {
    mCamYUp ^= 1;
    mCameraMode = mCamLookOrigin + 2 * mCamYUp;
    SetInfo("Camera locked on y-axis facing upwards: %s", mCamYUp ? "enabled" : "disabled");
  } else if (key == 'm') {
    mCameraMode++;
    if (mCameraMode == 4) {
      mCameraMode = 0;
    }
    mCamLookOrigin = mCameraMode & 1;
    mCamYUp = mCameraMode & 2;
    const char* modeText[] = { "Descent (free movement)", "Focus locked on origin (y-axis forced upwards)", "Spectator (y-axis forced upwards)", "Focus locked on origin (with free rotation)" };
    SetInfo("Camera mode %d: %s", mCameraMode, modeText[mCameraMode]);
  } else if (key == mBackend->KEY_ALT) {
    mBackend->mKeys[mBackend->KEY_CTRL] = false; // Release CTRL with alt, to avoid orienting along y automatically!
  } else if (key == 'l') {
    if (mCfg.drawSlice >= (mCfg.drawRelatedSlices ? (NSLICES / 4 - 1) : (NSLICES - 1))) {
      mCfg.drawSlice = -1;
      SetInfo("Showing all slices", 1);
    } else {
      mCfg.drawSlice++;
      SetInfo("Showing slice %d", mCfg.drawSlice);
    }
  } else if (key == 'k') {
    if (mCfg.drawSlice <= -1) {
      mCfg.drawSlice = mCfg.drawRelatedSlices ? (NSLICES / 4 - 1) : (NSLICES - 1);
    } else {
      mCfg.drawSlice--;
    }
    if (mCfg.drawSlice == -1) {
      SetInfo("Showing all slices", 1);
    } else {
      SetInfo("Showing slice %d", mCfg.drawSlice);
    }
  } else if (key == 'J') {
    mCfg.drawRelatedSlices ^= 1;
    SetInfo("Drawing of related slices %s", mCfg.drawRelatedSlices ? "enabled" : "disabled");
  } else if (key == 'L') {
    if (mCfg.showCollision >= mNCollissions - 1) {
      mCfg.showCollision = -1;
      SetInfo("Showing all collisions", 1);
    } else {
      mCfg.showCollision++;
      SetInfo("Showing collision %d", mCfg.showCollision);
    }
  } else if (key == 'K') {
    if (mCfg.showCollision <= -1) {
      mCfg.showCollision = mNCollissions - 1;
    } else {
      mCfg.showCollision--;
    }
    if (mCfg.showCollision == -1) {
      SetInfo("Showing all collisions", 1);
    } else {
      SetInfo("Showing collision %d", mCfg.showCollision);
    }
  } else if (key == 'F') {
    mFullScreen ^= 1;
    mBackend->SwitchFullscreen(mFullScreen);
    SetInfo("Toggling full screen (%d)", (int)mFullScreen);
  } else if (key == '_') {
    mMaximized ^= 1;
    mBackend->ToggleMaximized(mMaximized);
    SetInfo("Toggling mMaximized window (%d)", (int)mMaximized);
  } else if (key == 'R') {
    mBackend->mMaxFPSRate ^= 1;
    SetInfo("FPS rate %s", mBackend->mMaxFPSRate ? "not limited" : "limited");
  } else if (key == 'H') {
    mPrintInfoText += 1;
    mPrintInfoText &= 3;
    SetInfo("Info text display - console: %s, onscreen %s", (mPrintInfoText & 2) ? "enabled" : "disabled", (mPrintInfoText & 1) ? "enabled" : "disabled");
  } else if (key == 'j') {
    mSeparateGlobalTracks ^= 1;
    SetInfo("Seperated display of global tracks %s", mSeparateGlobalTracks ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == 'c') {
    if (mMarkClusters == 0) {
      mMarkClusters = 1;
    } else if (mMarkClusters >= 0x20) {
      mMarkClusters = 0;
    } else {
      mMarkClusters <<= 1;
    }
    SetInfo("Cluster flag highlight mask set to %d (%s)", mMarkClusters,
            mMarkClusters == 0 ? "off" : mMarkClusters == 1 ? "split pad" : mMarkClusters == 2 ? "split time" : mMarkClusters == 4 ? "edge" : mMarkClusters == 8 ? "singlePad" : mMarkClusters == 0x10 ? "reject distance" : "reject error");
    mUpdateDLList = true;
  } else if (key == 'B') {
    markAdjacentClusters++;
    if (markAdjacentClusters == 5) {
      markAdjacentClusters = 7;
    }
    if (markAdjacentClusters == 9) {
      markAdjacentClusters = 15;
    }
    if (markAdjacentClusters == 18) {
      markAdjacentClusters = 0;
    }
    if (markAdjacentClusters == 17) {
      SetInfo("Marking protected clusters (%d)", markAdjacentClusters);
    } else if (markAdjacentClusters == 16) {
      SetInfo("Marking removable clusters (%d)", markAdjacentClusters);
    } else {
      SetInfo("Marking adjacent clusters (%d): rejected %s, tube %s, looper leg %s, low Pt %s", markAdjacentClusters, (markAdjacentClusters & 1) ? "yes" : " no", (markAdjacentClusters & 2) ? "yes" : " no", (markAdjacentClusters & 4) ? "yes" : " no", (markAdjacentClusters & 8) ? "yes" : " no");
    }
    mUpdateDLList = true;
  } else if (key == 'C') {
    mCfg.colorCollisions ^= 1;
    SetInfo("Color coding of collisions %s", mCfg.colorCollisions ? "enabled" : "disabled");
  } else if (key == 'N') {
    mCfg.colorClusters ^= 1;
    SetInfo("Color coding for seed / trrack attachmend %s", mCfg.colorClusters ? "enabled" : "disabled");
  } else if (key == 'E') {
    mCfg.propagateTracks += 1;
    if (mCfg.propagateTracks == 4) {
      mCfg.propagateTracks = 0;
    }
    const char* infoText[] = { "Hits connected", "Hits connected and propagated to vertex", "Reconstructed track propagated inwards and outwards", "Monte Carlo track" };
    SetInfo("Display of propagated tracks: %s", infoText[mCfg.propagateTracks]);
  } else if (key == 'G') {
    mPropagateLoopers ^= 1;
    SetInfo("Propagation of loopers %s", mPropagateLoopers ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == 'v') {
    mHideRejectedClusters ^= 1;
    SetInfo("Rejected clusters are %s", mHideRejectedClusters ? "hidden" : "shown");
    mUpdateDLList = true;
  } else if (key == 'b') {
    mHideUnmatchedClusters ^= 1;
    SetInfo("Unmatched clusters are %s", mHideRejectedClusters ? "hidden" : "shown");
    mUpdateDLList = true;
  } else if (key == 'i') {
    mProjectXY ^= 1;
    SetInfo("Projection onto xy plane %s", mProjectXY ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == 'S') {
    mCfg.smoothPoints ^= true;
    SetInfo("Smoothing of points %s", mCfg.smoothPoints ? "enabled" : "disabled");
  } else if (key == 'A') {
    mCfg.smoothLines ^= true;
    SetInfo("Smoothing of lines %s", mCfg.smoothLines ? "enabled" : "disabled");
  } else if (key == 'D') {
    mCfg.depthBuffer ^= true;
    GLint depthBits;
    glGetIntegerv(GL_DEPTH_BITS, &depthBits);
    SetInfo("Depth buffer (z-buffer, %d bits) %s", depthBits, mCfg.depthBuffer ? "enabled" : "disabled");
    setDepthBuffer();
  } else if (key == 'W') {
    mDrawQualityMSAA *= 2;
    if (mDrawQualityMSAA < 2) {
      mDrawQualityMSAA = 2;
    }
    if (mDrawQualityMSAA > 16) {
      mDrawQualityMSAA = 0;
    }
    UpdateOffscreenBuffers();
    SetInfo("Multisampling anti-aliasing factor set to %d", mDrawQualityMSAA);
  } else if (key == 'U') {
    mDrawQualityDownsampleFSAA++;
    if (mDrawQualityDownsampleFSAA == 1) {
      mDrawQualityDownsampleFSAA = 2;
    }
    if (mDrawQualityDownsampleFSAA == 5) {
      mDrawQualityDownsampleFSAA = 0;
    }
    UpdateOffscreenBuffers();
    SetInfo("Downsampling anti-aliasing factor set to %d", mDrawQualityDownsampleFSAA);
  } else if (key == 'V') {
    mDrawQualityVSync ^= true;
    mBackend->SetVSync(mDrawQualityVSync);
    SetInfo("VSync: %s", mDrawQualityVSync ? "enabled" : "disabled");
  } else if (key == 'I') {
    mUseGLIndirectDraw ^= true;
    SetInfo("OpenGL Indirect Draw %s", mUseGLIndirectDraw ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == ';') {
    mUpdateDLList = true;
    mXadd += 60;
    mZadd += 60;
    SetInfo("TPC sector separation: %f %f", mXadd, mZadd);
  } else if (key == ':') {
    mUpdateDLList = true;
    mXadd -= 60;
    mZadd -= 60;
    if (mZadd < 0 || mXadd < 0) {
      mZadd = mXadd = 0;
    }
    SetInfo("TPC sector separation: %f %f", mXadd, mZadd);
  } else if (key == '#') {
    mInvertColors ^= 1;
  } else if (key == 'g') {
    mCfg.drawGrid ^= 1;
    SetInfo("Fast Cluster Search Grid %s", mCfg.drawGrid ? "shown" : "hidden");
  } else if (key == 'x') {
    mCfg.excludeClusters ^= 1;
    SetInfo(mCfg.excludeClusters ? "Clusters of selected category are excluded from display" : "Clusters are shown", 1);
  } else if (key == '.') {
    mHideRejectedTracks ^= 1;
    SetInfo("Rejected tracks are %s", mHideRejectedTracks ? "hidden" : "shown");
    mUpdateDLList = true;
  } else if (key == '1') {
    mCfg.drawClusters ^= 1;
  } else if (key == '2') {
    mCfg.drawInitLinks ^= 1;
  } else if (key == '3') {
    mCfg.drawLinks ^= 1;
  } else if (key == '4') {
    mCfg.drawSeeds ^= 1;
  } else if (key == '5') {
    mCfg.drawTracklets ^= 1;
  } else if (key == '6') {
    mCfg.drawTracks ^= 1;
  } else if (key == '7') {
    mCfg.drawGlobalTracks ^= 1;
  } else if (key == '8') {
    mCfg.drawFinal ^= 1;
  } else if (key == mBackend->KEY_F1) {
    mCfg.drawTPC ^= 1;
  } else if (key == mBackend->KEY_F2) {
    mCfg.drawTRD ^= 1;
  } else if (key == 't') {
    printf("Taking screenshot\n");
    static int nScreenshot = 1;
    char fname[32];
    sprintf(fname, "screenshot%d.bmp", nScreenshot++);
    DoScreenshot(fname);
    SetInfo("Taking screenshot (%s)", fname);
  } else if (key == 'Z') {
    screenshot_scale += 1;
    if (screenshot_scale == 5) {
      screenshot_scale = 1;
    }
    SetInfo("Screenshot scaling factor set to %d", screenshot_scale);
  } else if (key == 'y' || key == 'T') {
    if ((mAnimateScreenshot = (key == 'T'))) {
      mAnimationExport++;
    }
    if (mAnimateVectors[0].size() > 1) {
      startAnimation();
      SetInfo("Starting Animation", 1);
    } else {
      SetInfo("Insufficient Animation points to start Animation", 1);
    }
  } else if (key == '>') {
    mAnimationChangeConfig ^= 1;
    SetInfo("Interpolating visualization settings during Animation %s", mAnimationChangeConfig ? "enabled" : "disabled");
  } else if (key == 'Y') {
    setAnimationPoint();
    SetInfo("Added Animation point (%d points, %6.2f seconds)", (int)mAnimateVectors[0].size(), mAnimateVectors[0].back());
  } else if (key == 'X') {
    resetAnimation();
    SetInfo("Reset Animation points", 1);
  } else if (key == '\'') {
    removeAnimationPoint();
    SetInfo("Removed Animation point", 1);
  } else if (key == 'M') {
    mCfg.animationMode++;
    if (mCfg.animationMode == 7) {
      mCfg.animationMode = 0;
    }
    resetAnimation();
    if (mCfg.animationMode == 6) {
      SetInfo("Animation mode %d - Centered on origin", mCfg.animationMode);
    } else {
      SetInfo("Animation mode %d - Position: %s, Direction: %s", mCfg.animationMode, (mCfg.animationMode & 2) ? "Spherical (spherical rotation)" : (mCfg.animationMode & 4) ? "Spherical (Euler angles)" : "Cartesian", (mCfg.animationMode & 1) ? "Euler angles" : "Quaternion");
    }
  } else if (key == 'o') {
    FILE* ftmp = fopen("glpos.tmp", "w+b");
    if (ftmp) {
      int retval = fwrite(&mCurrentMatrix[0], sizeof(mCurrentMatrix[0]), 16, ftmp);
      if (retval != 16) {
        printf("Error writing position to file\n");
      } else {
        printf("Position stored to file\n");
      }
      fclose(ftmp);
    } else {
      printf("Error opening file\n");
    }
    SetInfo("Camera position stored to file", 1);
  } else if (key == 'p') {
    GLfloat tmp[16];
    FILE* ftmp = fopen("glpos.tmp", "rb");
    if (ftmp) {
      int retval = fread(&tmp[0], sizeof(tmp[0]), 16, ftmp);
      if (retval == 16) {
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(tmp);
        glGetFloatv(GL_MODELVIEW_MATRIX, mCurrentMatrix);
        printf("Position read from file\n");
      } else {
        printf("Error reading position from file\n");
      }
      fclose(ftmp);
    } else {
      printf("Error opening file\n");
    }
    SetInfo("Camera position loaded from file", 1);
  } else if (key == 'O') {
    FILE* ftmp = fopen("glanimation.tmp", "w+b");
    if (ftmp) {
      fwrite(&mCfg, sizeof(mCfg), 1, ftmp);
      int size = mAnimateVectors[0].size();
      fwrite(&size, sizeof(size), 1, ftmp);
      for (int i = 0; i < 9; i++) {
        fwrite(mAnimateVectors[i].data(), sizeof(mAnimateVectors[i][0]), size, ftmp);
      }
      fwrite(mAnimateConfig.data(), sizeof(mAnimateConfig[0]), size, ftmp);
      fclose(ftmp);
    } else {
      printf("Error opening file\n");
    }
    SetInfo("Animation path stored to file %s", "glanimation.tmp");
  } else if (key == 'P') {
    FILE* ftmp = fopen("glanimation.tmp", "rb");
    if (ftmp) {
      int retval = fread(&mCfg, sizeof(mCfg), 1, ftmp);
      int size;
      retval += fread(&size, sizeof(size), 1, ftmp);
      for (int i = 0; i < 9; i++) {
        mAnimateVectors[i].resize(size);
        retval += fread(mAnimateVectors[i].data(), sizeof(mAnimateVectors[i][0]), size, ftmp);
      }
      mAnimateConfig.resize(size);
      retval += fread(mAnimateConfig.data(), sizeof(mAnimateConfig[0]), size, ftmp);
      (void)retval; // disable unused warning
      fclose(ftmp);
      updateConfig();
    } else {
      printf("Error opening file\n");
    }
    SetInfo("Animation path loaded from file %s", "glanimation.tmp");
  } else if (key == 'h') {
    PrintHelp();
    SetInfo("Showing help text", 1);
  }
  /*else if (key == '#')
        {
            mTestSetting++;
            SetInfo("Debug test variable set to %d", mTestSetting);
            mUpdateDLList = true;
        }*/
}

void GPUDisplay::HandleSendKey(int key)
{
  // fprintf(stderr, "key %d '%c'\n", key, (char) key);

  bool shifted = key >= 'A' && key <= 'Z';
  int press = key;
  if (press >= 'a' && press <= 'z') {
    press += 'A' - 'a';
  }
  bool oldShift = mBackend->mKeysShift[press];
  mBackend->mKeysShift[press] = shifted;
  HandleKeyRelease(key);
  mBackend->mKeysShift[press] = oldShift;
}

void GPUDisplay::PrintGLHelpText(float colorValue)
{
  for (unsigned int i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    mBackend->OpenGLPrint(HelpText[i], 40.f, 35 + 20 * (1 + i), colorValue, colorValue, colorValue, mInfoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - mInfoHelpTimer.GetCurrentElapsedTime()) : 1, false);
  }
}

#endif
