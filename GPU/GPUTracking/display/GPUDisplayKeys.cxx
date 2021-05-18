// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayKeys.cxx
/// \author David Rohr

#include "GPUDisplay.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY

using namespace GPUCA_NAMESPACE::gpu;

const char* HelpText[] = {
  "[ESC]                         Quit",
  "[n]                           Next event",
  "[r]                           Reset Display Settings",
  "[l] / [k] / [J]               Draw single slice (next  / previous slice), draw related slices (same plane in phi)",
  "[;] / [:]                     Show splitting of TPC in slices by extruding volume, [:] resets",
  "[#]                           Invert colors",
  "[y] / [Y] / ['] / [X] / [M]   Start Animation, Add / remove Animation point, Reset Points, Cycle animation camera mode (resets)",
  "[>] / [<]                     Toggle config interpolation during Animation / change Animation interval (via movement)",
  "[g]                           Draw Grid",
  "[i]                           Project onto XY-plane",
  "[x]                           Exclude Clusters used in the tracking steps enabled for visualization ([1]-[8])",
  "[.]                           Exclude rejected tracks",
  "[c]                           Mark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20",
  "[z]                           Mark fake attached clusters",
  "[B]                           Mark clusters attached as adjacent",
  "[L] / [K]                     Draw single collisions (next / previous)",
  "[C]                           Colorcode clusters of different collisions",
  "[v]                           Hide rejected clusters from tracks",
  "[j]                           Show global tracks as additional segments of final tracks",
  "[u]                           Cycle through track filter",
  "[E] / [G]                     Extrapolate tracks / loopers",
  "[t] / [T]                     Take Screenshot / Record Animation to pictures",
  "[Z]                           Change screenshot resolution (scaling factor)",
  "[S] / [A] / [D]               Enable or disable smoothing of points / smoothing of lines / depth buffer",
  "[W] / [U] / [V]               Toggle anti-aliasing (MSAA at raster level / change downsampling FSAA factor) / toggle VSync",
  "[F] / [_] / [R]               Switch mFullScreen / Maximized window / FPS rate limiter",
  "[I]                           Enable / disable GL indirect draw",
  "[o] / [p] / [O] / [P]         Save / restore current camera position / Animation path",
  "[h]                           Print Help",
  "[H]                           Show info texts",
  "[w] / [s] / [a] / [d]         Zoom / Strafe Left and Right",
  "[pgup] / [pgdn]               Strafe up / down",
  "[e] / [f]                     Rotate left / right",
  "[+] / [-]                     Increase / decrease point size (Hold SHIFT for lines)",
  "[b]                           Change FOV (field of view)",
  "[']                           Switch between OpenGL core / compat code path",
  "[MOUSE 1]                     Look around",
  "[MOUSE 2]                     Strafe camera",
  "[MOUSE 1+2]                   Zoom / Rotate",
  "[SHIFT]                       Slow Zoom / Move / Rotate",
  "[ALT] / [CTRL] / [ENTER]      Focus camera on origin / orient y-axis upwards (combine with [SHIFT] to lock) / Cycle through modes",
  "[RCTRL] / [RALT]              Rotate model instead of camera / rotate TPC around beamline",
  "[1] ... [8] / [N]             Enable display of clusters, preseeds, seeds, starthits, tracklets, tracks, global tracks, merged tracks / Show assigned clusters in colors",
  "[F1] / [F2] / [F3] / [F4]     Enable / disable drawing of TPC / TRD / TOF / ITS",
  "[SHIFT] + [F1] to [F4]        Enable / disable track detector filter",
  "[SHIFT] + [F12]               Switch track detector filter between AND and OR mode"
  // FREE: [m] [SPACE] [q] [Q]
  // Test setting: ^ --> mHideUnmatchedClusters
};

void GPUDisplay::PrintHelp()
{
  mInfoHelpTimer.ResetStart();
  for (unsigned int i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    GPUInfo("%s", HelpText[i]);
  }
}

void GPUDisplay::HandleKey(unsigned char key)
{
  if (key == 'n') {
    mBackend->mDisplayControl = 1;
    SetInfo("Showing next event", 1);
  } else if (key == 27 || key == mBackend->KEY_ESCAPE) {
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
  } else if (key == mBackend->KEY_ENTER) {
    mCameraMode++;
    if (mCameraMode == 4) {
      mCameraMode = 0;
    }
    mCamLookOrigin = mCameraMode & 1;
    mCamYUp = mCameraMode & 2;
    const char* modeText[] = {"Descent (free movement)", "Focus locked on origin (y-axis forced upwards)", "Spectator (y-axis forced upwards)", "Focus locked on origin (with free rotation)"};
    SetInfo("Camera mode %d: %s", mCameraMode, modeText[mCameraMode]);
  } else if (key == mBackend->KEY_ALT) {
    mBackend->mKeys[mBackend->KEY_CTRL] = false; // Release CTRL with alt, to avoid orienting along y automatically!
  } else if (key == 'l') {
    if (mCfgL.drawSlice >= (mCfgL.drawRelatedSlices ? (NSLICES / 4 - 1) : (NSLICES - 1))) {
      mCfgL.drawSlice = -1;
      SetInfo("Showing all slices", 1);
    } else {
      mCfgL.drawSlice++;
      SetInfo("Showing slice %d", mCfgL.drawSlice);
    }
  } else if (key == 'k') {
    if (mCfgL.drawSlice <= -1) {
      mCfgL.drawSlice = mCfgL.drawRelatedSlices ? (NSLICES / 4 - 1) : (NSLICES - 1);
    } else {
      mCfgL.drawSlice--;
    }
    if (mCfgL.drawSlice == -1) {
      SetInfo("Showing all slices", 1);
    } else {
      SetInfo("Showing slice %d", mCfgL.drawSlice);
    }
  } else if (key == 'J') {
    mCfgL.drawRelatedSlices ^= 1;
    SetInfo("Drawing of related slices %s", mCfgL.drawRelatedSlices ? "enabled" : "disabled");
  } else if (key == 'L') {
    if (mCfgL.showCollision >= mNCollissions - 1) {
      mCfgL.showCollision = -1;
      SetInfo("Showing all collisions", 1);
    } else {
      mCfgL.showCollision++;
      SetInfo("Showing collision %d / %d", mCfgL.showCollision, mNCollissions);
    }
  } else if (key == 'K') {
    if (mCfgL.showCollision <= -1) {
      mCfgL.showCollision = mNCollissions - 1;
    } else {
      mCfgL.showCollision--;
    }
    if (mCfgL.showCollision == -1) {
      SetInfo("Showing all collisions", 1);
    } else {
      SetInfo("Showing collision %d", mCfgL.showCollision);
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
  } else if (key == 'z') {
    mMarkFakeClusters ^= 1;
    SetInfo("Marking fake clusters: %s", mMarkFakeClusters ? "on" : "off");
    mUpdateDLList = true;
  } else if (key == 'b') {
    if ((mFOV += 5) > 175) {
      mFOV = 5;
    }
    SetInfo("Set FOV to %f", mFOV);
  } else if (key == 39) { // character = "'"
#ifdef GPUCA_DISPLAY_OPENGL_CORE
    SetInfo("OpenGL compat profile not available, using core profile", 1);
#else
    mOpenGLCore ^= 1;
    SetInfo("Using renderer path for OpenGL %s profile", mOpenGLCore ? "core" : "compat");
#endif
  } else if (key == 'B') {
    mMarkAdjacentClusters++;
    if (mMarkAdjacentClusters == 5) {
      mMarkAdjacentClusters = 7;
    }
    if (mMarkAdjacentClusters == 9) {
      mMarkAdjacentClusters = 15;
    }
    if (mMarkAdjacentClusters == 17) {
      mMarkAdjacentClusters = 31;
    }
    if (mMarkAdjacentClusters == 34) {
      mMarkAdjacentClusters = 0;
    }
    if (mMarkAdjacentClusters == 33) {
      SetInfo("Marking protected clusters (%d)", mMarkAdjacentClusters);
    } else if (mMarkAdjacentClusters == 32) {
      SetInfo("Marking removable clusters (%d)", mMarkAdjacentClusters);
    } else {
      SetInfo("Marking adjacent clusters (%d): rejected %s, tube %s, looper leg %s, low Pt %s, high incl %s", mMarkAdjacentClusters, (mMarkAdjacentClusters & 1) ? "yes" : " no", (mMarkAdjacentClusters & 2) ? "yes" : " no", (mMarkAdjacentClusters & 4) ? "yes" : " no", (mMarkAdjacentClusters & 8) ? "yes" : " no", (mMarkAdjacentClusters & 16) ? "yes" : " no");
    }
    mUpdateDLList = true;
  } else if (key == 'C') {
    mCfgL.colorCollisions ^= 1;
    SetInfo("Color coding of collisions %s", mCfgL.colorCollisions ? "enabled" : "disabled");
  } else if (key == 'N') {
    mCfgL.colorClusters ^= 1;
    SetInfo("Color coding for seed / trrack attachmend %s", mCfgL.colorClusters ? "enabled" : "disabled");
  } else if (key == 'E') {
    mCfgL.propagateTracks += 1;
    if (mCfgL.propagateTracks == 4) {
      mCfgL.propagateTracks = 0;
    }
    const char* infoText[] = {"Hits connected", "Hits connected and propagated to vertex", "Reconstructed track propagated inwards and outwards", "Monte Carlo track"};
    SetInfo("Display of propagated tracks: %s", infoText[mCfgL.propagateTracks]);
  } else if (key == 'G') {
    mPropagateLoopers ^= 1;
    SetInfo("Propagation of loopers %s", mPropagateLoopers ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == 'v') {
    mHideRejectedClusters ^= 1;
    SetInfo("Rejected clusters are %s", mHideRejectedClusters ? "hidden" : "shown");
    mUpdateDLList = true;
  } else if (key == 'i') {
    mProjectXY ^= 1;
    SetInfo("Projection onto xy plane %s", mProjectXY ? "enabled" : "disabled");
    mUpdateDLList = true;
  } else if (key == 'S') {
    mCfgL.smoothPoints ^= true;
    SetInfo("Smoothing of points %s", mCfgL.smoothPoints ? "enabled" : "disabled");
  } else if (key == 'A') {
    mCfgL.smoothLines ^= true;
    SetInfo("Smoothing of lines %s", mCfgL.smoothLines ? "enabled" : "disabled");
  } else if (key == 'D') {
    mCfgL.depthBuffer ^= true;
    GLint depthBits = 0;
#ifndef GPUCA_DISPLAY_OPENGL_CORE
    glGetIntegerv(GL_DEPTH_BITS, &depthBits);
#endif
    SetInfo("Depth buffer (z-buffer, %d bits) %s", depthBits, mCfgL.depthBuffer ? "enabled" : "disabled");
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
    mCfgL.drawGrid ^= 1;
    SetInfo("Fast Cluster Search Grid %s", mCfgL.drawGrid ? "shown" : "hidden");
  } else if (key == 'x') {
    mCfgL.excludeClusters ^= 1;
    SetInfo(mCfgL.excludeClusters ? "Clusters of selected category are excluded from display" : "Clusters are shown", 1);
  } else if (key == '.') {
    mHideRejectedTracks ^= 1;
    SetInfo("Rejected tracks are %s", mHideRejectedTracks ? "hidden" : "shown");
    mUpdateDLList = true;
  } else if (key == '1') {
    mCfgL.drawClusters ^= 1;
  } else if (key == '2') {
    mCfgL.drawInitLinks ^= 1;
  } else if (key == '3') {
    mCfgL.drawLinks ^= 1;
  } else if (key == '4') {
    mCfgL.drawSeeds ^= 1;
  } else if (key == '5') {
    mCfgL.drawTracklets ^= 1;
  } else if (key == '6') {
    mCfgL.drawTracks ^= 1;
  } else if (key == '7') {
    mCfgL.drawGlobalTracks ^= 1;
  } else if (key == '8') {
    mCfgL.drawFinal ^= 1;
  } else if (key == mBackend->KEY_F1) {
    if (mBackend->mKeysShift[mBackend->KEY_F1]) {
      mCfgH.drawTPCTracks ^= 1;
      SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int)mCfgH.drawTPCTracks, (int)mCfgH.drawTRDTracks, (int)mCfgH.drawTOFTracks, (int)mCfgH.drawITSTracks);
      mUpdateDLList = true;
    } else {
      mCfgL.drawTPC ^= 1;
      SetInfo("Showing TPC Clusters: %d", (int)mCfgL.drawTPC);
    }
  } else if (key == mBackend->KEY_F2) {
    if (mBackend->mKeysShift[mBackend->KEY_F2]) {
      mCfgH.drawTRDTracks ^= 1;
      SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int)mCfgH.drawTPCTracks, (int)mCfgH.drawTRDTracks, (int)mCfgH.drawTOFTracks, (int)mCfgH.drawITSTracks);
      mUpdateDLList = true;
    } else {
      mCfgL.drawTRD ^= 1;
      SetInfo("Showing TRD Tracklets: %d", (int)mCfgL.drawTRD);
    }
  } else if (key == mBackend->KEY_F3) {
    if (mBackend->mKeysShift[mBackend->KEY_F3]) {
      mCfgH.drawTOFTracks ^= 1;
      SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int)mCfgH.drawTPCTracks, (int)mCfgH.drawTRDTracks, (int)mCfgH.drawTOFTracks, (int)mCfgH.drawITSTracks);
      mUpdateDLList = true;
    } else {
      mCfgL.drawTOF ^= 1;
      SetInfo("Showing TOF Hits: %d", (int)mCfgL.drawTOF);
    }
  } else if (key == mBackend->KEY_F4) {
    if (mBackend->mKeysShift[mBackend->KEY_F4]) {
      mCfgH.drawITSTracks ^= 1;
      SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int)mCfgH.drawTPCTracks, (int)mCfgH.drawTRDTracks, (int)mCfgH.drawTOFTracks, (int)mCfgH.drawITSTracks);
      mUpdateDLList = true;
    } else {
      mCfgL.drawITS ^= 1;
      SetInfo("Showing ITS Clusters: %d", (int)mCfgL.drawITS);
    }
  } else if (key == mBackend->KEY_F12 && mBackend->mKeysShift[mBackend->KEY_F12]) {
    mCfgH.drawTracksAndFilter ^= 1;
    SetInfo("Track filter: %s", mCfgH.drawTracksAndFilter ? "AND" : "OR");
  } else if (key == 't') {
    GPUInfo("Taking screenshot");
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
    mCfgL.animationMode++;
    if (mCfgL.animationMode == 7) {
      mCfgL.animationMode = 0;
    }
    resetAnimation();
    if (mCfgL.animationMode == 6) {
      SetInfo("Animation mode %d - Centered on origin", mCfgL.animationMode);
    } else {
      SetInfo("Animation mode %d - Position: %s, Direction: %s", mCfgL.animationMode, (mCfgL.animationMode & 2) ? "Spherical (spherical rotation)" : (mCfgL.animationMode & 4) ? "Spherical (Euler angles)" : "Cartesian", (mCfgL.animationMode & 1) ? "Euler angles" : "Quaternion");
    }
  } else if (key == 'u') {
    mTrackFilter = (mTrackFilter + 1) % 3;
    mUpdateDLList = true;
    SetInfo("Track filter: %s", mTrackFilter == 2 ? "TRD Track candidates" : mTrackFilter ? "TRD Tracks only" : "None");
  } else if (key == 'o') {
    FILE* ftmp = fopen("glpos.tmp", "w+b");
    if (ftmp) {
      int retval = fwrite(&mViewMatrix, sizeof(mViewMatrix), 1, ftmp);
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
  } else if (key == 'p') {
    FILE* ftmp = fopen("glpos.tmp", "rb");
    if (ftmp) {
      int retval = fread(&mViewMatrix, 1, sizeof(mViewMatrix), ftmp);
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
  } else if (key == 'O') {
    FILE* ftmp = fopen("glanimation.tmp", "w+b");
    if (ftmp) {
      fwrite(&mCfgL, sizeof(mCfgL), 1, ftmp);
      int size = mAnimateVectors[0].size();
      fwrite(&size, sizeof(size), 1, ftmp);
      for (int i = 0; i < 9; i++) {
        fwrite(mAnimateVectors[i].data(), sizeof(mAnimateVectors[i][0]), size, ftmp);
      }
      fwrite(mAnimateConfig.data(), sizeof(mAnimateConfig[0]), size, ftmp);
      fclose(ftmp);
    } else {
      GPUError("Error opening file");
    }
    SetInfo("Animation path stored to file %s", "glanimation.tmp");
  } else if (key == 'P') {
    FILE* ftmp = fopen("glanimation.tmp", "rb");
    if (ftmp) {
      int retval = fread(&mCfgL, sizeof(mCfgL), 1, ftmp);
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
      GPUError("Error opening file");
    }
    SetInfo("Animation path loaded from file %s", "glanimation.tmp");
  } else if (key == 'h') {
    PrintHelp();
    SetInfo("Showing help text", 1);
  }
  /*
  else if (key == '^')
  {
    mTestSetting++;
    SetInfo("Debug test variable set to %d", mTestSetting);
    // mHideUnmatchedClusters ^= 1;
    mUpdateDLList = true;
  }
  */
}

void GPUDisplay::HandleSendKey(int key)
{
  // GPUError("key %d '%c'", key, (char) key);

  bool shifted = key >= 'A' && key <= 'Z';
  int press = key;
  if (press >= 'a' && press <= 'z') {
    press += 'A' - 'a';
  }
  bool oldShift = mBackend->mKeysShift[press];
  mBackend->mKeysShift[press] = shifted;
  HandleKey(key);
  mBackend->mKeysShift[press] = oldShift;
}

void GPUDisplay::PrintGLHelpText(float colorValue)
{
  for (unsigned int i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    mBackend->OpenGLPrint(HelpText[i], 40.f, 35 + 20 * (1 + i), colorValue, colorValue, colorValue, mInfoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - mInfoHelpTimer.GetCurrentElapsedTime()) : 1, false);
  }
}

#endif
