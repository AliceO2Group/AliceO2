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

/// \file GPUDisplayFrontendConfigInfo.inc
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDCONFIGINFO_INC
#define GPUDISPLAYFRONTENDCONFIGINFO_INC

#define INFO_LAMBDA(obj, value, ...) std::pair(mFrontend->mConfigRTTI->getRTTIIndex(*this, &std::remove_cvref_t<decltype(obj)>::value), configKeyCallback([=, this]() { [[maybe_unused]] auto& val = obj.value; __VA_ARGS__ })) // Info printed after the callback of a key bound to the option, same val as in KEY_LAMBDA, local helpers captured by copy

void GPUDisplay::buildConfigInfo()
{
  auto addInfo = [this](std::pair<uint32_t, configKeyCallback>&& info) { mFrontend->mConfigRTTI->configInfos[info.first] = std::move(info.second); };
  addInfo(INFO_LAMBDA(mCfgL, colorClusters, SetInfo("Color coding for seed / trrack attachmend %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgL, drawSector, {
    if (val == -1) {
      SetInfo("Showing all sectors", 1);
    } else {
      SetInfo("Showing sector %d", val);
    }
  }));
  addInfo(INFO_LAMBDA(mCfgL, drawRelatedSectors, SetInfo("Drawing of related sectors %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgH, xAdd, SetInfo("TPC sector separation: %f %f", mCfgH.xAdd, mCfgH.zAdd);));
  addInfo(INFO_LAMBDA(mCfgL, animationMode, {
    if (val == 6) {
      SetInfo("Animation mode %d - Centered on origin", val);
    } else {
      SetInfo("Animation mode %d - Position: %s, Direction: %s", val, (val & 2) ? "Spherical (spherical rotation)" : ((val & 4) ? "Spherical (Euler angles)" : "Cartesian"), (val & 1) ? "Euler angles" : "Quaternion");
    }
  }));
  addInfo(INFO_LAMBDA(mCfgL, drawGrid, SetInfo("Fast Cluster Search Grid %s", val ? "shown" : "hidden");));
  addInfo(INFO_LAMBDA(mCfgH, projectXY, SetInfo("Projection onto xy plane %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgL, excludeClusters, SetInfo(val ? "Clusters of selected category are excluded from display (%d)" : "Clusters are shown", val);));
  addInfo(INFO_LAMBDA(mCfgH, hideRejectedTracks, SetInfo("Rejected tracks are %s", val ? "hidden" : "shown");));
  addInfo(INFO_LAMBDA(mCfgH, markClusters, {
    SetInfo("Cluster flag highlight mask set to %d (%s)", val, val == 0 ? "off" : (val == 1 ? "split pad" : (val == 2 ? "split time" : (val == 4 ? "edge" : (val == 8 ? "singlePad" : (val == 0x10 ? "reject distance" : "reject error"))))));
  }));
  addInfo(INFO_LAMBDA(mCfgH, markFakeClusters, SetInfo("Marking fake clusters: %s", val ? "on" : "off");));
  addInfo(INFO_LAMBDA(mCfgH, markAdjacentClusters, {
    if (val == 33) {
      SetInfo("Marking protected clusters (%d)", val);
    } else if (val == 32) {
      SetInfo("Marking removable clusters (%d)", val);
    } else {
      SetInfo("Marking adjacent clusters (%d): rejected %s, tube %s, looper leg %s, low Pt %s, high incl %s", val, (val & 1) ? "yes" : " no", (val & 2) ? "yes" : " no", (val & 4) ? "yes" : " no", (val & 8) ? "yes" : " no", (val & 16) ? "yes" : " no");
    }
  }));
  addInfo(INFO_LAMBDA(mCfgL, showCollision, {
    if (val == -1) {
      SetInfo("Showing all collisions", 1);
    } else {
      SetInfo("Showing collision %d / %d", val, mNCollissions);
    }
  }));
  addInfo(INFO_LAMBDA(mCfgL, colorCollisions, SetInfo("Color coding of collisions %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgH, hideRejectedClusters, SetInfo("Rejected clusters are %s", val ? "hidden" : "shown");));
  addInfo(INFO_LAMBDA(mCfgH, separateExtrapolatedTracks, SetInfo("Seperated display of tracks propagated to adjacent sectors %s / of CE tracks %s", val ? "enabled" : "disabled", mCfgH.splitCETracks ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgH, trackFilter, SetInfo("Track filter: %s", val == 0 ? "None" : mConfig.filterMacros[val - 1].c_str());));
  addInfo(INFO_LAMBDA(mCfgL, propagateTracks, {
    const char* infoText[] = {"Hits connected", "Hits connected and propagated to vertex", "Reconstructed track propagated inwards and outwards", "Monte Carlo track"};
    SetInfo("Display of propagated tracks: %s", infoText[val]);
  }));
  addInfo(INFO_LAMBDA(mCfgH, propagateLoopers, SetInfo("Propagation of loopers %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgR, screenshotScaleFactor, SetInfo("Screenshot scaling factor set to %d", val);));
  addInfo(INFO_LAMBDA(mCfgL, smoothPoints, SetInfo("Smoothing of points %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgL, smoothLines, SetInfo("Smoothing of lines %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgL, depthBuffer, SetInfo("Depth buffer (z-buffer) %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgR, drawQualityMSAA, SetInfo("Multisampling anti-aliasing factor set to %d", val);));
  addInfo(INFO_LAMBDA(mCfgR, drawQualityDownsampleFSAA, SetInfo("Downsampling anti-aliasing factor set to %d", val);));
  addInfo(INFO_LAMBDA(mCfgR, drawQualityVSync, SetInfo("VSync: %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgR, fpsLimit, {
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
  }));
  addInfo(INFO_LAMBDA(mCfgR, fullScreen, SetInfo("Toggling full screen (%d)", (int32_t)val);));
  addInfo(INFO_LAMBDA(mCfgR, maximized, SetInfo("Toggling Maximized window (%d)", (int32_t)val);));
  addInfo(INFO_LAMBDA(mCfgR, fov, SetInfo("Set FOV to %f", val);));
  addInfo(INFO_LAMBDA(mCfgR, useGLIndirectDraw, SetInfo("OpenGL Indirect Draw %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgL, drawTPC, SetInfo("Showing TPC Clusters: %d", (int32_t)val);));
  addInfo(INFO_LAMBDA(mCfgL, drawTRD, SetInfo("Showing TRD Tracklets: %d", (int32_t)val);));
  addInfo(INFO_LAMBDA(mCfgL, drawTOF, SetInfo("Showing TOF Hits: %d", (int32_t)val);));
  addInfo(INFO_LAMBDA(mCfgL, drawITS, SetInfo("Showing ITS Clusters: %d", (int32_t)val);));
  auto trackFilterMaskInfo = [this]() { SetInfo("Track Filter Mask: TPC:%d TRD:%d TOF:%d ITS:%d", (int32_t)mCfgH.drawTPCTracks, (int32_t)mCfgH.drawTRDTracks, (int32_t)mCfgH.drawTOFTracks, (int32_t)mCfgH.drawITSTracks); };
  addInfo(INFO_LAMBDA(mCfgH, drawTPCTracks, trackFilterMaskInfo();));
  addInfo(INFO_LAMBDA(mCfgH, drawTRDTracks, trackFilterMaskInfo();));
  addInfo(INFO_LAMBDA(mCfgH, drawTOFTracks, trackFilterMaskInfo();));
  addInfo(INFO_LAMBDA(mCfgH, drawITSTracks, trackFilterMaskInfo();));
  addInfo(INFO_LAMBDA(mCfgH, drawTracksAndFilter, SetInfo("Track filter: %s", val ? "AND" : "OR");));
  addInfo(INFO_LAMBDA(mCfgR, camLookOrigin, SetInfo("Camera locked on origin: %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgR, camYUp, SetInfo("Camera locked on y-axis facing upwards: %s", val ? "enabled" : "disabled");));
  addInfo(INFO_LAMBDA(mCfgR, cameraMode, {
    const char* modeText[] = {"Descent (free movement)", "Focus locked on origin (y-axis forced upwards)", "Spectator (y-axis forced upwards)", "Focus locked on origin (with free rotation)"};
    SetInfo("Camera mode %d: %s", val, modeText[val]);
  }));
}

#endif // GPUDISPLAYFRONTENDCONFIGINFO_INC
