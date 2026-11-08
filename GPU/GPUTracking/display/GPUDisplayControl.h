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

// No include guard: included multiple times by qconfig with different macro definitions

#include <atomic>
#include <cstdint>
#include <string>
#include "GPUDisplayTypes.h"

// clang-format off
BeginNamespace(o2::gpu)
BeginHiddenConfig(GPUDisplayControl, control)
  AddVariable(mDisplayControl, GPUDisplayCopyableAtomic<int32_t>, 0) // Control for next event (=1) or quit (=2)
  AddVariable(mResetScene, GPUDisplayCopyableAtomic<bool>, false)
  AddVariable(mPrintInfoText, int32_t, 1)
  AddVariable(mPrintInfoTextAlways, bool, false)
  AddVariable(mRequestScreenshot, bool, false)
  AddVariable(mAnimateScreenshot, int32_t, 0)
  AddVariable(mAnimationChangeConfig, bool, true)
  AddVariable(mAnimationDelay, float, 2.f)
  AddVariable(mPrintInfoTextGUI, bool, true)                     // Send the status and info text to the Qt GUI, shown at its bottom
  AddVariable(mGuiPrintInfo, bool, false)                        // Show the info of options changed with Qt GUI elements (not for key presses in the GUI)
  AddVariable(mRotateModel, bool, false)                         // Rotate the model instead of the camera, like holding RCTRL (set in the Qt GUI)
  AddVariable(mRotateAroundZ, bool, false)                       // Rotate the TPC around the beamline, like holding RALT (set in the Qt GUI)
  AddVariable(mFilterMacroCustom, std::string, "")               // Code of the custom track filter ROOT macro (trackFilter = filterMacros.size() + 1), set in the Qt GUI
  AddVariable(mScreenShotname, std::string, "screenshot")        // Name of screenshot and animation frame files, followed by numbers
  AddVariable(mGLPosFile, std::string, "glpos.tmp")              // File for saving / loading the camera position
  AddVariable(mGLAnimationFile, std::string, "glanimation.tmp")  // File for saving / loading the animation path
  AddVariable(mGuiDarkMode, int8_t, -1)                          // Dark mode of the Qt GUI, -1 = not yet set: the GUI reports the system scheme when it starts
  AddVariable(mGuiScale, float, 1.f)                             // Scale factor of text and elements of the Qt GUI (on top of the system scaling), Ctrl + mouse wheel / Ctrl + 0 in the GUI
  AddVariable(showConsole, bool, false)                          // Console shown (^ key): all other keys go to the console
  AddVariable(dummy, bool, false)                                // dummy value to reference for keys that do not change variables
EndConfig()
EndNamespace() // o2::gpu
