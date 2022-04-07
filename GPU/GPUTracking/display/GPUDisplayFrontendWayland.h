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

struct xdg_wm_base;
struct xdg_toplevel;
struct xdg_surface;
struct zxdg_decoration_manager_v1;
struct zxdg_toplevel_decoration_v1;
struct xkb_context;
struct xkb_keymap;
struct xkb_state;

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayFrontendWayland : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendWayland();
  ~GPUDisplayFrontendWayland() override = default;

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  void getSize(int& width, int& height) override;
  int getVulkanSurface(void* instance, void* surface) override;
  unsigned int getReqVulkanExtensions(const char**& p) override;

 private:
  int FrontendMain() override;
  int GetKey(unsigned int key, unsigned int state);
  void createBuffer(unsigned int width, unsigned int height);
  void recreateBuffer(unsigned int width, unsigned int height);

  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;
  volatile bool mDisplayRunning = false;

  wl_display* mWayland = nullptr;
  wl_registry* mRegistry = nullptr;

  wl_compositor* mCompositor = nullptr;
  wl_seat* mSeat = nullptr;
  wl_pointer* mPointer = nullptr;
  wl_keyboard* mKeyboard = nullptr;
  xdg_wm_base* mXdgBase = nullptr;
  wl_shm* mShm = nullptr;

  wl_surface* mSurface = nullptr;
  xdg_surface* mXdgSurface = nullptr;
  xdg_toplevel* mXdgToplevel = nullptr;

  wl_output* mOutput = nullptr;

  int mFd = 0;
  wl_shm_pool* mPool;
  wl_buffer* mBuffer;

  xkb_context* mXKBcontext = nullptr;
  xkb_keymap* mXKBkeymap = nullptr;
  xkb_state* mXKBstate = nullptr;

  zxdg_decoration_manager_v1* mDecManager = nullptr;
  // zxdg_toplevel_decoration_v1* mXdgDecoration = nullptr;

  int mWidthRequested = 0;
  int mHeightRequested = 0;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
