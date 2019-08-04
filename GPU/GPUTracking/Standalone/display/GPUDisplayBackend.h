// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackend.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKEND_H
#define GPUDISPLAYBACKEND_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUReconstruction;
class GPUDisplay;

class GPUDisplayBackend
{
  friend class GPUDisplay;

 public:
  GPUDisplayBackend() = default;
  virtual ~GPUDisplayBackend() = default;

  virtual int StartDisplay() = 0;                                                                                            // Start the display. This function returns, and should spawn a thread that runs the display, and calls InitGL
  virtual void DisplayExit() = 0;                                                                                            // Stop the display. Display thread should call ExitGL and the function returns after the thread has terminated
  virtual void SwitchFullscreen(bool set) = 0;                                                                               // Toggle full-screen mode
  virtual void ToggleMaximized(bool set) = 0;                                                                                // Maximize window
  virtual void SetVSync(bool enable) = 0;                                                                                    // Enable / disable vsync
  virtual bool EnableSendKey();                                                                                              // Request external keys (e.g. from terminal)
  virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) = 0; // Print text on the display (needs the backend to build the font)

  // volatile variables to exchange control informations between display and backend
  volatile int mDisplayControl = 0; // Control for next event (=1) or quit (=2)
  volatile int mSendKey = 0;        // Key sent by external entity (usually console), may be ignored by backend.
  volatile int mNeedUpdate = 0;     // flag that backend shall update the GL window, and call DrawGLScene

 protected:
  virtual int OpenGLMain() = 0;
  static void* OpenGLWrapper(void*);

  static constexpr int INIT_WIDTH = 1024, INIT_HEIGHT = 768;                           // Initial window size, before maximizing
  static constexpr const char* GL_WINDOW_NAME = "GPU CA TPC Standalone Event Display"; // Title of event display set by backend
  // Constant key codes for special mKeys (to unify different treatment in X11 / Windows / GLUT / etc.)
  static constexpr int KEY_UP = 1;
  static constexpr int KEY_DOWN = 2;
  static constexpr int KEY_LEFT = 3;
  static constexpr int KEY_RIGHT = 4;
  static constexpr int KEY_PAGEUP = 5;
  static constexpr int KEY_PAGEDOWN = 6;
  static constexpr int KEY_SPACE = 7;
  static constexpr int KEY_SHIFT = 8;
  static constexpr int KEY_ALT = 9;
  static constexpr int KEY_CTRL = 10;
  static constexpr int KEY_F1 = 11;
  static constexpr int KEY_F2 = 12;
  static constexpr int KEY_F3 = 26;
  static constexpr int KEY_F4 = 14;
  static constexpr int KEY_F5 = 15;
  static constexpr int KEY_F6 = 16;
  static constexpr int KEY_F7 = 17;
  static constexpr int KEY_F8 = 18;
  static constexpr int KEY_F9 = 19;
  static constexpr int KEY_F10 = 20;
  static constexpr int KEY_F11 = 21;
  static constexpr int KEY_F12 = 22;
  static constexpr int KEY_HOME = 23;
  static constexpr int KEY_END = 24;
  static constexpr int KEY_INSERT = 25;
  static constexpr int KEY_ESCAPE = 27;
  static constexpr int KEY_ENTER = 13;

  // Keyboard / Mouse actions
  bool mMouseDn = false;          // Mouse button down
  bool mMouseDnR = false;         // Right mouse button down
  float mMouseDnX, mMouseDnY;     // X/Y position where mouse button was pressed
  float mouseMvX, mouseMvY;       // Current mouse pointer position
  int mMouseWheel = 0;            // Incremental value of mouse wheel, ca +/- 100 per wheel tick
  bool mKeys[256] = {false};      // Array of mKeys currently pressed
  bool mKeysShift[256] = {false}; // Array whether shift was held during key-press
  int mDisplayHeight = INIT_HEIGHT;
  int mDisplayWidth = INIT_WIDTH;

  int mMaxFPSRate = 0; // run at highest possible frame rate, do not sleep in between frames

  GPUDisplay* mDisplay; // Ptr to display, not owning, set by display when it connects to backend

  void HandleKeyRelease(unsigned char key);                             // Callback for handling key presses
  int DrawGLScene(bool mixAnimation = false, float animateTime = -1.f); // Callback to draw the GL scene
  void HandleSendKey();                                                 // Optional callback to handle key press from external source (e.g. stdin by default)
  void ReSizeGLScene(int width, int height);                            // Callback when GL window is resized
  int InitGL(bool initFailure = false);                                 // Callback to initialize the GL Display (to be called in StartDisplay)
  void ExitGL();                                                        // Callback to clean up the GL Display
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
