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

/// \file GPUDisplayFrontend.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTEND_H
#define GPUDISPLAYFRONTEND_H

#include "GPUCommonDef.h"
#include "GPUDisplayInterface.h"
#include <memory>
#include <chrono>
#include <mutex>
#include <vector>

namespace o2::gpu
{
class GPUReconstruction;
class GPUDisplay;
class GPUDisplayBackend;
class GPUDisplayGUIWrapper;
struct GPUDisplayGUIChannel;

class GPUDisplayFrontend : public GPUDisplayFrontendInterface
{
  friend class GPUDisplay;
  friend class GPUDisplayGUI; // Key codes

 public:
  GPUDisplayFrontend();
  ~GPUDisplayFrontend() override;

  enum frontendTypes {
    TYPE_INVALID = -1,
    TYPE_WIN32 = 0,
    TYPE_X11 = 1,
    TYPE_GLUT = 2,
    TYPE_GLFW = 3,
    TYPE_WAYLAND = 4,
    TYPE_NONE = 5
  };

  // Compile time minimum version checked in display/CMakeLists.txt (OpenGL 4.6 headers for SPIR-V shaders), keep in sync!
  static constexpr int32_t GL_MIN_VERSION_MAJOR = 4;
  static constexpr int32_t GL_MIN_VERSION_MINOR = 6;

  int32_t StartDisplay();                                                                                                    // Start the display. This function returns, and should spawn a thread that runs the display, and calls InitDisplay
  void StopDisplay() override;                                                                                               // Stop the display. Display thread should call ExitDisplay. Functions waits for display thread to terminate
  virtual void SwitchFullscreen(bool set) = 0;                                                                               // Toggle full-screen mode
  virtual void ToggleMaximized(bool set) = 0;                                                                                // Maximize window
  virtual void SetVSync(bool enable) = 0;                                                                                    // Enable / disable vsync
  virtual bool canSetVSync() { return false; }                                                                               // OpenGL: swap interval can be set
  virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) = 0; // Print text on the display (needs the backend to build the font)
  GPUDisplayBackend* backend();
  static GPUDisplayFrontend* getFrontend(const char* type);
  virtual void getSize(int32_t& width, int32_t& height) { width = height = 0; }
  virtual float getRefreshRate() { return 0.f; }        // Refresh rate of the display the window is on in Hz, 0 = unknown
  virtual float getDisplayScale() const { return 1.f; } // HiDPI scale factor: physical pixels per logical pixel, text positions and sizes are scaled by it
  virtual int32_t getVulkanSurface(void* instance, void* surface) { return 1; }
  virtual uint32_t getReqVulkanExtensions(const char**& p) { return 0; };

  int32_t getDisplayControl() const override { return mDisplayControl; }
  void setDisplayControl(int32_t v) override { mDisplayControl = v; }
  void sendExternalGUIKey(int32_t v) override;

  frontendTypes frontendType() const { return mFrontendType; }
  const char* frontendName() const override { return mFrontendName; }

  int32_t startGUI();
  void stopGUI();
  bool isGUIRunning();

 protected:
  virtual int32_t FrontendMain() = 0;
  virtual void StopDisplayA();
  static void* FrontendThreadWrapper(void*);
  void waitForFPSLimit(); // Sleep until the next frame is due according to fpsRate (only in FPS_LIMIT_RATE mode, see GPUDisplayBackend::fpsLimitMode)

  static constexpr int32_t INIT_WIDTH = 1024, INIT_HEIGHT = 768;                        // Initial window size, before maximizing
  static constexpr int32_t MIN_WIDTH = 200, MIN_HEIGHT = 150;                           // Minimum window size
  static constexpr const char* DISPLAY_WINDOW_NAME = "GPU CA Standalone Event Display"; // Title of event display set by backend
  // Constant key codes for special mKeys (to unify different treatment in X11 / Windows / GLUT / etc.)
  static constexpr int32_t KEY_UP = 1;
  static constexpr int32_t KEY_DOWN = 2;
  static constexpr int32_t KEY_LEFT = 3;
  static constexpr int32_t KEY_RIGHT = 4;
  static constexpr int32_t KEY_PAGEUP = 5;
  static constexpr int32_t KEY_PAGEDOWN = 6;
  static constexpr int32_t KEY_SHIFT = 8;
  static constexpr int32_t KEY_ALT = 9;
  static constexpr int32_t KEY_RALT = 10;
  static constexpr int32_t KEY_CTRL = 11;
  static constexpr int32_t KEY_RCTRL = 12;
  static constexpr int32_t KEY_ENTER = 13; // fixed at 13
  static constexpr int32_t KEY_F1 = 14;
  static constexpr int32_t KEY_F2 = 15;
  static constexpr int32_t KEY_F3 = 16;
  static constexpr int32_t KEY_F4 = 17;
  static constexpr int32_t KEY_F5 = 18;
  static constexpr int32_t KEY_F6 = 19;
  static constexpr int32_t KEY_F7 = 20;
  static constexpr int32_t KEY_F8 = 21;
  static constexpr int32_t KEY_F9 = 22;
  static constexpr int32_t KEY_F10 = 23;
  static constexpr int32_t KEY_F11 = 24;
  static constexpr int32_t KEY_F12 = 25;
  static constexpr int32_t KEY_INSERT = 26;
  static constexpr int32_t KEY_ESCAPE = 27; // fixed at 27
  static constexpr int32_t KEY_HOME = 28;
  static constexpr int32_t KEY_END = 29;
  static constexpr int32_t KEY_SPACE = 32; // fixed at 32

  // volatile variables to exchange control informations between display and backend
  volatile int32_t mDisplayControl = 0; // Control for next event (=1) or quit (=2)
  volatile int32_t mNeedUpdate = 0;     // flag that backend shall update the GL window, and call DrawGLScene
  std::mutex mExternalGUIKeyMutex;
  std::vector<int32_t> mExternalGUIKeys; // Keys sent by external entity (usually console), may be ignored by backend.
  std::mutex mExitStateMutex;

  // Keyboard / Mouse actions
  bool mMouseDn = false;          // Mouse button down
  bool mMouseDnR = false;         // Right mouse button down
  bool mMouseDnM = false;         // Middle mouse button down (not used yet)
  float mMouseDnX, mMouseDnY;     // X/Y position where mouse button was pressed
  float mMouseMvX, mMouseMvY;     // Current mouse pointer position
  int32_t mMouseWheel = 0;        // Incremental value of mouse wheel, ca +/- 100 per wheel tick
  bool mKeys[256] = {false};      // Array of mKeys currently pressed
  bool mKeysShift[256] = {false}; // Array whether shift was held during key-press
  int32_t mDisplayHeight = INIT_HEIGHT;
  int32_t mDisplayWidth = INIT_WIDTH;
  int32_t mCanDrawText = 0; // 1 = in compat mode, 2 = with shader

  std::chrono::steady_clock::time_point mNextFrameTime;

  GPUDisplay* mDisplay = nullptr;        // Ptr to display, not owning, set by display when it connects to backend
  GPUDisplayBackend* mBackend = nullptr; // Ptr to backend, not owning

  frontendTypes mFrontendType = TYPE_INVALID;
  const char* mFrontendName = nullptr;
  volatile bool mDisplayRunning = false;

  std::unique_ptr<GPUDisplayGUIChannel> mGUIChannel; // Must outlive mGUI
  std::unique_ptr<GPUDisplayGUIWrapper> mGUI;

  void HandleKey(uint8_t key);                     // Callback for handling key presses
  int32_t DrawGLScene();                           // Callback to draw the GL scene
  void HandleExternalGUISignals();                 // Callback to handle signals from external GUIs
  bool HasExternalGUISignals() const;              // Returns whether there are currently pending signals from external GUIs to be handled
  void ResizeScene(int32_t width, int32_t height); // Callback when GL window is resized
  int32_t InitDisplay(bool initFailure = false);   // Callback to initialize the GL Display (to be called in StartDisplay)
  void ExitDisplay();                              // Callback to clean up the GL Display
  int32_t& drawTextFontSize();
  bool vsyncEnabled() const;                            // VSync setting (drawQualityVSync)
  void setWindowState(bool fullScreen, bool maximized); // Sync the settings if the window manager changed the window state
  void focusLost();                                     // Release all keys and mouse buttons when the window loses focus, since the release events go to another window
  void applyInitialWindowState();                       // Switch to fullscreen / maximize at startup according to the settings
 private:
  bool HandleExternalGUIKeys(); // Returns whether keys were handled, config changes are not handled
};
} // namespace o2::gpu

#endif
