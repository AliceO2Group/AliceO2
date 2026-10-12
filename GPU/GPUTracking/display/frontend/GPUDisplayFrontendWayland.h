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

/// \file GPUDisplayFrontenWayland1.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDWAYLAND_H
#define GPUDISPLAYFRONTENDWAYLAND_H

#include "GPUDisplayFrontend.h"
#include <pthread.h>
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#include <vector>
#include <string>

struct zxdg_decoration_manager_v1;
struct zxdg_toplevel_decoration_v1;
struct xkb_context;
struct xkb_keymap;
struct xkb_state;

namespace o2::gpu
{
class GPUDisplayFrontendWayland : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendWayland();
  ~GPUDisplayFrontendWayland() override = default;

  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  void getSize(int32_t& width, int32_t& height) override;
  float getDisplayScale() const override { return mScale; }
  int32_t getVulkanSurface(void* instance, void* surface) override;
  uint32_t getReqVulkanExtensions(const char**& p) override;

 private:
  int32_t FrontendMain() override;
  int32_t GetKey(uint32_t key, uint32_t state);
  void updateScale();     // Buffer scale from wl_surface.preferred_buffer_scale, or max scale of the outputs the surface is on (older compositors)
  void updateSize();      // Physical size = logical size * scale, resizes the scene if changed
  void handleKeyRepeat(); // Call HandleKey for the held key according to wl_keyboard.repeat_info
  void applyScroll();     // Apply scroll events of one wl_pointer.frame: high-resolution wheel (120 = 1 notch = 100 mMouseWheel units) if available, otherwise continuous axis (touchpad, ~15 units per notch)

  struct OutputInfo {
    wl_output* output;
    uint32_t globalName;
    uint32_t version;
    int32_t scale = 1;
    std::string name;
  };
  OutputInfo* findOutput(wl_output* output);

  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;

  wl_display* mWayland = nullptr;
  wl_registry* mRegistry = nullptr;

  wl_compositor* mCompositor = nullptr;
  wl_seat* mSeat = nullptr;
  wl_pointer* mPointer = nullptr;
  wl_keyboard* mKeyboard = nullptr;
  xdg_wm_base* mXdgBase = nullptr;

  wl_surface* mSurface = nullptr;
  xdg_surface* mXdgSurface = nullptr;
  xdg_toplevel* mXdgToplevel = nullptr;

  wl_registry_listener mRegistryListener{};
  wl_seat_listener mSeatListener{};
  wl_pointer_listener mPointerListener{};
  wl_keyboard_listener mKeyboardListener{};
  wl_output_listener mOutputListener{};
  wl_surface_listener mSurfaceListener{};
  xdg_wm_base_listener mXdgWmBaseListener{};
  xdg_surface_listener mXdgSurfaceListener{};
  xdg_toplevel_listener mXdgToplevelListener{};

  std::vector<OutputInfo> mOutputs;
  std::vector<wl_output*> mSurfaceOutputs; // Outputs the surface is on, most recently entered last
  uint32_t mSeatVersion = 0;

  xkb_context* mXKBcontext = nullptr;
  xkb_keymap* mXKBkeymap = nullptr;
  xkb_state* mXKBstate = nullptr;

  zxdg_decoration_manager_v1* mDecManager = nullptr;
  zxdg_toplevel_decoration_v1* mXdgDecoration = nullptr;

  int32_t mWidthRequested = 0;
  int32_t mHeightRequested = 0;
  int32_t mWidthLogical = INIT_WIDTH;
  int32_t mHeightLogical = INIT_HEIGHT;
  int32_t mWidthWindowed = INIT_WIDTH; // Last size when neither maximized nor fullscreen
  int32_t mHeightWindowed = INIT_HEIGHT;
  bool mStateFullScreenOrMaximized = false;
  int32_t mBoundsWidth = 0;
  int32_t mBoundsHeight = 0;
  int32_t mScale = 1;
  int32_t mPreferredScale = 0; // 0 = not sent by compositor
  bool mSuspended = false;
  bool mCanFullscreen = true;
  bool mCanMaximize = true;

  int32_t mRepeatRate = 0;     // Keys per second, 0 = no repeat
  int32_t mRepeatDelay = 0;    // ms before the first repeat
  int64_t mRepeatKeyCode = -1; // Evdev key code of the held key to repeat, -1 = none
  int32_t mRepeatSymbol = 0;
  std::chrono::steady_clock::time_point mRepeatNext;

  double mAxisValue = 0; // Accumulated scroll values of one wl_pointer.frame
  int32_t mAxisValue120 = 0;
  bool mHasAxisValue120 = false;
};
GPUDisplayFrontend* createGPUDisplayFrontendWayland();
} // namespace o2::gpu

#endif
