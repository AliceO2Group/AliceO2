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

/// \file GPUDisplayControl.h
/// \author David Rohr

#ifndef GPUDISPLAYCONTROL_H
#define GPUDISPLAYCONTROL_H

#include <atomic>
#include <cstdint>
#include <string>

namespace o2::gpu
{
template <class T>
struct GPUDisplayCopyableAtomic : std::atomic<T> { // std::atomic deletes the copy, here it copies the value
  using std::atomic<T>::atomic;
  GPUDisplayCopyableAtomic(const GPUDisplayCopyableAtomic& o) : std::atomic<T>(o.load()) {}
  GPUDisplayCopyableAtomic& operator=(const GPUDisplayCopyableAtomic& o)
  {
    this->store(o.load());
    return *this;
  }
};

struct GPUDisplayControl {                               // Settings of the display without RTTI, copied to the Qt GUI
  GPUDisplayCopyableAtomic<int32_t> mDisplayControl = 0; // Control for next event (=1) or quit (=2)
  GPUDisplayCopyableAtomic<bool> mResetScene = false;
  int32_t mPrintInfoText = 1;
  bool mPrintInfoTextAlways = false;
  bool mRequestScreenshot = false;
  int32_t mAnimateScreenshot = 0;
  bool mAnimationChangeConfig = true;
  float mAnimationDelay = 2.f;
  bool mPrintInfoTextGUI = true;                    // Send the status and info text to the Qt GUI, shown at its bottom
  bool mGuiPrintInfo = false;                       // Show the info of options changed with Qt GUI elements (not for key presses in the GUI)
  bool mRotateModel = false;                        // Rotate the model instead of the camera, like holding RCTRL (set in the Qt GUI)
  bool mRotateAroundZ = false;                      // Rotate the TPC around the beamline, like holding RALT (set in the Qt GUI)
  std::string mFilterMacroCustom;                   // Code of the custom track filter ROOT macro (trackFilter = filterMacros.size() + 1), set in the Qt GUI
  std::string mScreenShotname = "screenshot";       // Name of screenshot and animation frame files, followed by numbers
  std::string mGLPosFile = "glpos.tmp";             // File for saving / loading the camera position
  std::string mGLAnimationFile = "glanimation.tmp"; // File for saving / loading the animation path
  int8_t mGuiDarkMode = -1;                         // Dark mode of the Qt GUI, -1 = not yet set: the GUI reports the system scheme when it starts
  float mGuiScale = 1.f;                            // Scale factor of text and elements of the Qt GUI (on top of the system scaling), Ctrl + mouse wheel / Ctrl + 0 in the GUI
  bool dummy = false;                               // dummy value to reference for keys that do not change variables
};
} // namespace o2::gpu

#endif
