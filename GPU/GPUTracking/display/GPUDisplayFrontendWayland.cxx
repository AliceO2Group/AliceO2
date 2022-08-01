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

/// \file GPUDisplayFrontendWayland.cxx
/// \author David Rohr

// Now the other headers
#include "GPUDisplayFrontendWayland.h"
#include "GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPUDisplay.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <functional>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#include "xdg-decoration-client-protocol.h"
#include <xkbcommon/xkbcommon.h>
#include <linux/input-event-codes.h>

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayFrontendWayland::GPUDisplayFrontendWayland()
{
  mFrontendType = TYPE_WAYLAND;
  mFrontendName = "Wayland";
}

void GPUDisplayFrontendWayland::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
}

template <class T, class... Args>
struct CCallWrapper {
  std::function<T(Args...)> func;
  static T callback(void* context, Args... args)
  {
    const CCallWrapper* funcwrap = reinterpret_cast<const CCallWrapper*>(context);
    return funcwrap->func(std::forward<Args>(args)...);
  }
};

int GPUDisplayFrontendWayland::GetKey(unsigned int key, unsigned int state)
{
  int retVal = 0;
  if (mXKBkeymap) {
    xkb_keysym_t sym = xkb_state_key_get_one_sym(mXKBstate, key + 8);
    if (sym == 65453) {
      return ('-');
    } else if (sym == 65451) {
      return ('+');
    } else if (sym == XKB_KEY_Shift_L || sym == XKB_KEY_Shift_R) {
      return (KEY_SHIFT);
    } else if (sym == XKB_KEY_Alt_L) {
      return (KEY_ALT);
    } else if (sym == XKB_KEY_ISO_Level3_Shift || sym == XKB_KEY_Alt_R) {
      return (KEY_RALT);
    } else if (sym == XKB_KEY_Control_L) {
      return (KEY_CTRL);
    } else if (sym == XKB_KEY_Control_R) {
      return (KEY_RCTRL);
    } else if (sym == XKB_KEY_Up) {
      return (KEY_UP);
    } else if (sym == XKB_KEY_Down) {
      return (KEY_DOWN);
    } else if (sym == XKB_KEY_Left) {
      return (KEY_LEFT);
    } else if (sym == XKB_KEY_Right) {
      return (KEY_RIGHT);
    } else if (sym == XKB_KEY_Page_Up) {
      return (KEY_PAGEUP);
    } else if (sym == XKB_KEY_Page_Down) {
      return (KEY_PAGEDOWN);
    } else if (sym == XKB_KEY_Escape) {
      return (KEY_ESCAPE);
    } else if (sym == XKB_KEY_Return) {
      return (KEY_ENTER);
    } else if (sym == XKB_KEY_End) {
      return (KEY_END);
    } else if (sym == XKB_KEY_Home) {
      return (KEY_HOME);
    } else if (sym == XKB_KEY_Insert) {
      return (KEY_INSERT);
    } else if (sym == XKB_KEY_F1) {
      return (KEY_F1);
    } else if (sym == XKB_KEY_F2) {
      return (KEY_F2);
    } else if (sym == XKB_KEY_F3) {
      return (KEY_F3);
    } else if (sym == XKB_KEY_F4) {
      return (KEY_F4);
    } else if (sym == XKB_KEY_F5) {
      return (KEY_F5);
    } else if (sym == XKB_KEY_F6) {
      return (KEY_F6);
    } else if (sym == XKB_KEY_F7) {
      return (KEY_F7);
    } else if (sym == XKB_KEY_F8) {
      return (KEY_F8);
    } else if (sym == XKB_KEY_F9) {
      return (KEY_F9);
    } else if (sym == XKB_KEY_F10) {
      return (KEY_F10);
    } else if (sym == XKB_KEY_F11) {
      return (KEY_F11);
    } else if (sym == XKB_KEY_F12) {
      return (KEY_F12);
    } else if (sym == 32) {
      return (KEY_SPACE);
    } else if (sym > 255) {
      return (0);
    } else {
      retVal = xkb_keysym_to_utf32(sym);
    }
  }
  if (retVal > 255) {
    return (0);
  }
  return retVal;
}

void GPUDisplayFrontendWayland::createBuffer(unsigned int width, unsigned int height)
{
  const unsigned int stride = width * 4;
  const unsigned int size = stride * height;
  if (ftruncate(mFd, size) < 0) {
    throw std::runtime_error("Error setting waysland shm file size");
  }
  void* shm_data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0);
  if (shm_data == MAP_FAILED) {
    throw std::runtime_error("wayland mmap failed");
  }
  memset(shm_data, 0, size);
  munmap(shm_data, size);

  mPool = wl_shm_create_pool(mShm, mFd, size);
  mBuffer = wl_shm_pool_create_buffer(mPool, 0, width, height, stride, WL_SHM_FORMAT_XRGB8888);
  mDisplayWidth = width;
  mDisplayHeight = height;
  wl_surface_attach(mSurface, mBuffer, 0, 0);
  wl_surface_commit(mSurface);
}

void GPUDisplayFrontendWayland::recreateBuffer(unsigned int width, unsigned int height)
{
  wl_surface_attach(mSurface, nullptr, 0, 0);
  wl_surface_commit(mSurface);
  wl_buffer_destroy(mBuffer);
  wl_shm_pool_destroy(mPool);
  createBuffer(width, height);
}

int GPUDisplayFrontendWayland::FrontendMain()
{
  if (backend()->backendType() != GPUDisplayBackend::TYPE_VULKAN) {
    fprintf(stderr, "Only Vulkan backend supported\n");
    return 1;
  }

  mXKBcontext = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
  if (mXKBcontext == nullptr) {
    throw std::runtime_error("Error initializing xkb context");
  }

  mWayland = wl_display_connect(nullptr);
  if (mWayland == nullptr) {
    throw std::runtime_error("Could not connect to wayland display");
  }
  mRegistry = wl_display_get_registry(mWayland);
  if (mRegistry == nullptr) {
    throw std::runtime_error("Could not create wayland registry");
  }

  mFd = memfd_create("/tmp/ca-gpu-display-wayland-memfile", 0);
  if (mFd < 0) {
    throw std::runtime_error("Error creating wayland shm segment file descriptor");
  }

  auto pointer_enter = [](void* data, wl_pointer* wl_pointer, uint32_t serial, wl_surface* surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
  };
  auto pointer_leave = [](void* data, wl_pointer* wl_pointer, uint32_t serial, wl_surface* wl_surface) {
  };
  auto pointer_motion = [](void* data, wl_pointer* wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mMouseMvX = wl_fixed_to_double(surface_x);
    me->mMouseMvY = wl_fixed_to_double(surface_y);
  };
  auto pointer_button = [](void* data, wl_pointer* wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
      if (button == BTN_RIGHT) {
        me->mMouseDnR = true;
      } else if (button == BTN_LEFT) {
        me->mMouseDn = true;
      }
      me->mMouseDnX = me->mMouseMvX;
      me->mMouseDnY = me->mMouseMvY;
    } else if (state == WL_POINTER_BUTTON_STATE_RELEASED) {
      if (button == BTN_RIGHT) {
        me->mMouseDnR = false;
      } else if (button == BTN_LEFT) {
        me->mMouseDn = false;
      }
    }
  };
  auto pointer_axis = [](void* data, wl_pointer* wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
    if (axis == 0) {
      ((GPUDisplayFrontendWayland*)data)->mMouseWheel += wl_fixed_to_double(value) * (-100.f / 15.f);
    }
  };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  const wl_pointer_listener pointer_listener = {.enter = pointer_enter, .leave = pointer_leave, .motion = pointer_motion, .button = pointer_button, .axis = pointer_axis, .frame = nullptr, .axis_source = nullptr, .axis_stop = nullptr, .axis_discrete = nullptr};
#pragma GCC diagnostic pop

  auto keyboard_keymap = [](void* data, wl_keyboard* wl_keyboard, uint format, int fd, uint size) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (me->mXKBkeymap) {
      xkb_state_unref(me->mXKBstate);
      xkb_keymap_unref(me->mXKBkeymap);
    }
    char* keymap_string = (char*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    me->mXKBkeymap = xkb_keymap_new_from_string(me->mXKBcontext, keymap_string, XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
    me->mXKBstate = xkb_state_new(me->mXKBkeymap);
    munmap(keymap_string, size);
    close(fd);
  };
  auto keyboard_enter = [](void* data, wl_keyboard* wl_keyboard, uint serial, wl_surface* surface, wl_array* keys) {};
  auto keyboard_leave = [](void* data, wl_keyboard* wl_keyboard, uint serial, wl_surface* surface) {};
  auto keyboard_key = [](void* data, wl_keyboard* wl_keyboard, uint serial, uint time, uint key, uint state) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    int symbol = me->GetKey(key, state);
    int keyPress = (symbol >= 'a' && symbol <= 'z') ? symbol + 'A' - 'a' : symbol;
    if (state == XKB_KEY_DOWN) {
      me->mKeys[keyPress] = true;
      me->mKeysShift[keyPress] = me->mKeys[KEY_SHIFT];
      me->HandleKey(symbol);
    } else {
      me->mKeys[keyPress] = false;
      me->mKeysShift[keyPress] = false;
    }
  };
  auto keyboard_modifiers = [](void* data, wl_keyboard* wl_keyboard, uint serial, uint mods_depressed, uint mods_latched, uint mods_locked, uint group) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    xkb_state_update_mask(me->mXKBstate, mods_depressed, mods_latched, mods_locked, 0, 0, group);
  };
  auto keyboard_repat = [](void* data, wl_keyboard* wl_keyboard, int rate, int delay) {};
  const wl_keyboard_listener keyboard_listener = {.keymap = keyboard_keymap, .enter = keyboard_enter, .leave = keyboard_leave, .key = keyboard_key, .modifiers = keyboard_modifiers, .repeat_info = keyboard_repat};

  auto xdg_wm_base_ping = [](void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial) {
    xdg_wm_base_pong(xdg_wm_base, serial);
  };
  const xdg_wm_base_listener xdg_wm_base_listener = {
    .ping = xdg_wm_base_ping,
  };

  auto seat_capabilities = [&](struct wl_seat* seat, uint32_t capabilities) {
    if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
      mPointer = wl_seat_get_pointer(mSeat);
      wl_pointer_add_listener(mPointer, &pointer_listener, this);
    }
    if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
      mKeyboard = wl_seat_get_keyboard(mSeat);
      wl_keyboard_add_listener(mKeyboard, &keyboard_listener, this);
    }
  };
  auto seat_capabilities_c = CCallWrapper<void, wl_seat*, uint32_t>{[seat_capabilities](wl_seat* seat, uint32_t capabilities) { seat_capabilities(seat, capabilities); }};

  auto seat_name = [](void* data, struct wl_seat* seat, const char* name) {
    if (((GPUDisplayFrontendWayland*)data)->mDisplay->param()->par.debugLevel >= 2) {
      GPUInfo("Wayland seat: %s", name);
    }
  };
  const wl_seat_listener seat_listener = {
    .capabilities = seat_capabilities_c.callback,
    .name = seat_name,
  };

  auto registry_global = [&](wl_registry* registry, uint32_t name, const char* interface, uint32_t version) {
    if (mDisplay->param()->par.debugLevel >= 3) {
      GPUInfo("Available interface %s", interface);
    }
    if (strcmp(interface, wl_output_interface.name) == 0) {
      mOutput = (wl_output*)wl_registry_bind(registry, name, &wl_output_interface, 1);
      // wl_output_add_listener(mOutput, &output_listener, this);
    } else if (strcmp(interface, wl_compositor_interface.name) == 0) {
      mCompositor = (wl_compositor*)wl_registry_bind(registry, name, &wl_compositor_interface, 1);
    } else if (strcmp(interface, wl_shm_interface.name) == 0) {
      mShm = (wl_shm*)wl_registry_bind(registry, name, &wl_shm_interface, 1);
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
      mXdgBase = (xdg_wm_base*)wl_registry_bind(registry, name, &xdg_wm_base_interface, 1);
      xdg_wm_base_add_listener(mXdgBase, &xdg_wm_base_listener, this);
    } else if (strcmp(interface, wl_seat_interface.name) == 0) {
      mSeat = (wl_seat*)wl_registry_bind(registry, name, &wl_seat_interface, 1);
      wl_seat_add_listener(mSeat, &seat_listener, &seat_capabilities_c);
    } else if (strcmp(interface, zxdg_toplevel_decoration_v1_interface.name) == 0) {
      mDecManager = (zxdg_decoration_manager_v1*)wl_registry_bind(registry, name, &zxdg_toplevel_decoration_v1_interface, 1);
    }
  };

  auto registry_global_c = CCallWrapper<void, wl_registry*, uint32_t, const char*, uint32_t>{[registry_global](wl_registry* registry, uint32_t name, const char* interface, uint32_t version) { registry_global(registry, name, interface, version); }};
  auto registry_global_remove = [](void* a, wl_registry* b, uint32_t c) {};
  const wl_registry_listener registry_listener = {.global = &registry_global_c.callback, .global_remove = registry_global_remove};

  wl_registry_add_listener(mRegistry, &registry_listener, &registry_global_c);
  wl_display_roundtrip(mWayland);

  if (mCompositor == nullptr || mShm == nullptr || mXdgBase == nullptr || mSeat == nullptr || mOutput == nullptr) {
    throw std::runtime_error("Error getting wayland objects");
  }

  mSurface = wl_compositor_create_surface(mCompositor);
  if (mSurface == nullptr) {
    throw std::runtime_error("Error creating wayland surface");
  }
  mXdgSurface = xdg_wm_base_get_xdg_surface(mXdgBase, mSurface);
  if (mXdgSurface == nullptr) {
    throw std::runtime_error("Error creating wayland xdg surface");
  }
  mXdgToplevel = xdg_surface_get_toplevel(mXdgSurface);

  auto xdg_toplevel_handle_configure = [](void* data, xdg_toplevel* toplevel, int32_t width, int32_t height, wl_array* states) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (me->mDisplay->param()->par.debugLevel >= 3) {
      GPUInfo("Wayland surface resized to %d %d", width, height);
    }
    me->mWidthRequested = width;
    me->mHeightRequested = height;
  };

  auto xdg_surface_handle_configure = [](void* data, xdg_surface* surface, uint32_t serial) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    xdg_surface_ack_configure(me->mXdgSurface, serial);
    if (me->mWidthRequested && me->mHeightRequested && (me->mWidthRequested != me->mDisplayWidth || me->mHeightRequested != me->mDisplayHeight)) {
      me->recreateBuffer(me->mWidthRequested, me->mHeightRequested);
      me->ResizeScene(me->mDisplayWidth, me->mDisplayHeight);
    } else {
      wl_surface_commit(((GPUDisplayFrontendWayland*)data)->mSurface);
    }
    me->mWidthRequested = me->mHeightRequested = 0;
  };

  auto xdg_toplevel_handle_close = [](void* data, xdg_toplevel* toplevel) {
    ((GPUDisplayFrontendWayland*)data)->mDisplayControl = 2;
  };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  const xdg_surface_listener xdg_surface_listener = {
    .configure = xdg_surface_handle_configure,
  };

  const xdg_toplevel_listener xdg_toplevel_listener = {
    .configure = xdg_toplevel_handle_configure,
    .close = xdg_toplevel_handle_close,
    .configure_bounds = nullptr};
#pragma GCC diagnostic pop

  xdg_surface_add_listener(mXdgSurface, &xdg_surface_listener, this);
  xdg_toplevel_add_listener(mXdgToplevel, &xdg_toplevel_listener, this);

  xdg_toplevel_set_title(mXdgToplevel, DISPLAY_WINDOW_NAME);

  if (mDecManager) {
    printf("Enabling decoration\n");
  }

  wl_surface_commit(mSurface);
  wl_display_roundtrip(mWayland);

  createBuffer(INIT_WIDTH, INIT_HEIGHT);

  if (InitDisplay()) {
    return (1);
  }

  while (wl_display_dispatch(mWayland) != -1 && mDisplayControl != 2) {
    HandleSendKey();
    DrawGLScene();
    wl_surface_damage(mSurface, 0, 0, mDisplayWidth, mDisplayHeight);
    usleep(10000);
  }

  ExitDisplay();

  if (mXKBkeymap) {
    xkb_state_unref(mXKBstate);
    xkb_keymap_unref(mXKBkeymap);
    mXKBkeymap = nullptr;
  }
  xkb_context_unref(mXKBcontext);

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  if (mPointer) {
    wl_pointer_release(mPointer);
    mPointer = nullptr;
  }
  if (mKeyboard) {
    wl_keyboard_release(mKeyboard);
    mKeyboard = nullptr;
  }
  xdg_toplevel_destroy(mXdgToplevel);
  xdg_surface_destroy(mXdgSurface);
  wl_surface_destroy(mSurface);
  wl_buffer_destroy(mBuffer);
  wl_shm_pool_destroy(mPool);
  wl_registry_destroy(mRegistry);
  wl_display_disconnect(mWayland);
  close(mFd);

  return (0);
}

void GPUDisplayFrontendWayland::DisplayExit()
{
  pthread_mutex_lock(&mSemLockExit);
  if (mDisplayRunning) {
    mDisplayControl = 2;
  }
  pthread_mutex_unlock(&mSemLockExit);
  while (mDisplayRunning) {
    usleep(10000);
  }
}

void GPUDisplayFrontendWayland::SwitchFullscreen(bool set)
{
  if (set) {
    xdg_toplevel_set_fullscreen(mXdgToplevel, mOutput);
  } else {
    xdg_toplevel_unset_fullscreen(mXdgToplevel);
  }
}

void GPUDisplayFrontendWayland::ToggleMaximized(bool set)
{
  if (set) {
    xdg_toplevel_set_maximized(mXdgToplevel);
  } else {
    xdg_toplevel_unset_maximized(mXdgToplevel);
  }
}

void GPUDisplayFrontendWayland::SetVSync(bool enable)
{
}

int GPUDisplayFrontendWayland::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, FrontendThreadWrapper, this)) {
    GPUError("Coult not Create frontend Thread...");
    return (1);
  }
  return (0);
}

void GPUDisplayFrontendWayland::getSize(int& width, int& height)
{
  width = mDisplayWidth;
  height = mDisplayHeight;
}

int GPUDisplayFrontendWayland::getVulkanSurface(void* instance, void* surface)
{
  VkWaylandSurfaceCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
  info.flags = 0;
  info.display = mWayland;
  info.surface = mSurface;
  return vkCreateWaylandSurfaceKHR(*(VkInstance*)instance, &info, nullptr, (VkSurfaceKHR*)surface) != VK_SUCCESS;
}

unsigned int GPUDisplayFrontendWayland::getReqVulkanExtensions(const char**& p)
{
  static const char* exts[] = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
  p = exts;
  return 2;
}
