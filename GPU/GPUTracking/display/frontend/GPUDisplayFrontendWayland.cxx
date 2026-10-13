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
#include "backend/GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPUDisplay.h"
#include "GPULogging.h"
#include "GPUParam.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cerrno>
#include <chrono>
#include <algorithm>
#include <cmath>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>

#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#include "xdg-decoration-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "cursor-shape-client-protocol.h"
#include <xkbcommon/xkbcommon.h>
#include <linux/input-event-codes.h>

using namespace o2::gpu;

GPUDisplayFrontend* o2::gpu::createGPUDisplayFrontendWayland() { return new GPUDisplayFrontendWayland; }

GPUDisplayFrontendWayland::GPUDisplayFrontendWayland()
{
  mFrontendType = TYPE_WAYLAND;
  mFrontendName = "Wayland";
}

void GPUDisplayFrontendWayland::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
}

int32_t GPUDisplayFrontendWayland::GetKey(uint32_t key, uint32_t state)
{
  int32_t retVal = 0;
  if (mXKBkeymap) {
    xkb_keysym_t sym = xkb_state_key_get_one_sym(mXKBstate, key + 8);
    if (sym == XKB_KEY_KP_Subtract) {
      return ('-');
    } else if (sym == XKB_KEY_KP_Add) {
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

int32_t GPUDisplayFrontendWayland::FrontendMain()
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

  auto pointer_enter = [](void* data, wl_pointer* wl_pointer, uint32_t serial, wl_surface* surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (!me->mCursorShapeDevice && me->mCursorShapeManager) {
      me->mCursorShapeDevice = wp_cursor_shape_manager_v1_get_pointer(me->mCursorShapeManager, wl_pointer);
    }
    if (me->mCursorShapeDevice) {
      wp_cursor_shape_device_v1_set_shape(me->mCursorShapeDevice, serial, WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_DEFAULT);
    }
  };
  auto pointer_leave = [](void* data, wl_pointer* wl_pointer, uint32_t serial, wl_surface* wl_surface) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mMouseDn = me->mMouseDnR = me->mMouseDnM = false;
  };
  auto pointer_motion = [](void* data, wl_pointer* wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mMouseMvX = wl_fixed_to_double(surface_x) * me->mScale;
    me->mMouseMvY = wl_fixed_to_double(surface_y) * me->mScale;
  };
  auto pointer_button = [](void* data, wl_pointer* wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
      if (button == BTN_RIGHT) {
        me->mMouseDnR = true;
      } else if (button == BTN_LEFT) {
        me->mMouseDn = true;
      } else if (button == BTN_MIDDLE) {
        me->mMouseDnM = true;
      }
      me->mMouseDnX = me->mMouseMvX;
      me->mMouseDnY = me->mMouseMvY;
    } else if (state == WL_POINTER_BUTTON_STATE_RELEASED) {
      if (button == BTN_RIGHT) {
        me->mMouseDnR = false;
      } else if (button == BTN_LEFT) {
        me->mMouseDn = false;
      } else if (button == BTN_MIDDLE) {
        me->mMouseDnM = false;
      }
    }
  };
  auto pointer_frame = [](void* data, wl_pointer* wl_pointer) { ((GPUDisplayFrontendWayland*)data)->applyScroll(); };
  auto pointer_axis = [](void* data, wl_pointer* wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
      me->mAxisValue += wl_fixed_to_double(value);
      if (me->mSeatVersion < WL_POINTER_FRAME_SINCE_VERSION) {
        me->applyScroll();
      }
    }
  };
  auto pointer_axis_source = [](void* data, wl_pointer* wl_pointer, uint32_t axis_source) {};
  auto pointer_axis_stop = [](void* data, wl_pointer* wl_pointer, uint32_t time, uint32_t axis) {};
  auto pointer_axis_discrete = [](void* data, wl_pointer* wl_pointer, uint32_t axis, int32_t discrete) {};
#ifdef WL_POINTER_AXIS_VALUE120_SINCE_VERSION
  auto pointer_axis_value120 = [](void* data, wl_pointer* wl_pointer, uint32_t axis, int32_t value120) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
      me->mAxisValue120 += value120;
      me->mHasAxisValue120 = true;
    }
  };
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  mPointerListener = wl_pointer_listener{.enter = pointer_enter, .leave = pointer_leave, .motion = pointer_motion, .button = pointer_button, .axis = pointer_axis, .frame = pointer_frame, .axis_source = pointer_axis_source, .axis_stop = pointer_axis_stop, .axis_discrete = pointer_axis_discrete,
#ifdef WL_POINTER_AXIS_VALUE120_SINCE_VERSION
                                         .axis_value120 = pointer_axis_value120
#endif
  };
#pragma GCC diagnostic pop
  auto keyboard_keymap = [](void* data, wl_keyboard* wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
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
  auto keyboard_enter = [](void* data, wl_keyboard* wl_keyboard, uint32_t serial, wl_surface* surface, wl_array* keys) {};
  auto keyboard_leave = [](void* data, wl_keyboard* wl_keyboard, uint32_t serial, wl_surface* surface) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->focusLost();
    me->mRepeatKeyCode = -1;
  };
  auto keyboard_key = [](void* data, wl_keyboard* wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    int32_t symbol = me->GetKey(key, state);
    int32_t keyPress = (symbol >= 'a' && symbol <= 'z') ? symbol + 'A' - 'a' : symbol;
    if (state == WL_KEYBOARD_KEY_STATE_PRESSED) {
      me->mKeys[keyPress] = true;
      me->mKeysShift[keyPress] = me->mKeys[KEY_SHIFT];
      me->HandleKey(symbol);
      if (symbol && me->mRepeatRate > 0 && me->mXKBkeymap && xkb_keymap_key_repeats(me->mXKBkeymap, key + 8)) { // Wayland compositors do not repeat keys, the client must do it
        me->mRepeatKeyCode = key;
        me->mRepeatSymbol = symbol;
        me->mRepeatNext = std::chrono::steady_clock::now() + std::chrono::milliseconds(me->mRepeatDelay);
      }
    } else {
      me->mKeys[keyPress] = false;
      me->mKeysShift[keyPress] = false;
      if ((int64_t)key == me->mRepeatKeyCode) {
        me->mRepeatKeyCode = -1;
      }
    }
  };
  auto keyboard_modifiers = [](void* data, wl_keyboard* wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    xkb_state_update_mask(me->mXKBstate, mods_depressed, mods_latched, mods_locked, 0, 0, group);
  };
  auto keyboard_repat = [](void* data, wl_keyboard* wl_keyboard, int32_t rate, int32_t delay) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mRepeatRate = rate;
    me->mRepeatDelay = delay;
    if (rate <= 0) {
      me->mRepeatKeyCode = -1;
    }
  };
  mKeyboardListener = wl_keyboard_listener{.keymap = keyboard_keymap, .enter = keyboard_enter, .leave = keyboard_leave, .key = keyboard_key, .modifiers = keyboard_modifiers, .repeat_info = keyboard_repat};

  auto xdg_wm_base_ping = [](void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial) {
    xdg_wm_base_pong(xdg_wm_base, serial);
  };
  mXdgWmBaseListener = xdg_wm_base_listener{
    .ping = xdg_wm_base_ping,
  };

  auto seat_capabilities = [](void* data, struct wl_seat* seat, uint32_t capabilities) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if ((capabilities & WL_SEAT_CAPABILITY_POINTER) && !me->mPointer) {
      me->mPointer = wl_seat_get_pointer(seat);
      wl_pointer_add_listener(me->mPointer, &me->mPointerListener, me);
    }
    if ((capabilities & WL_SEAT_CAPABILITY_KEYBOARD) && !me->mKeyboard) {
      me->mKeyboard = wl_seat_get_keyboard(seat);
      wl_keyboard_add_listener(me->mKeyboard, &me->mKeyboardListener, me);
    }
  };
  auto seat_name = [](void* data, struct wl_seat* seat, const char* name) {
    if (((GPUDisplayFrontendWayland*)data)->mDisplay->GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Wayland seat: %s", name);
    }
  };
  mSeatListener = wl_seat_listener{
    .capabilities = seat_capabilities,
    .name = seat_name,
  };

  auto output_geometry = [](void* data, wl_output* output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char* make, const char* model, int32_t transform) {};
  auto output_mode = [](void* data, wl_output* output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {};
  auto output_done = [](void* data, wl_output* output) {};
  auto output_scale = [](void* data, wl_output* output, int32_t factor) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (auto* o = me->findOutput(output)) {
      o->scale = factor;
      me->updateScale();
    }
  };
#ifdef WL_OUTPUT_NAME_SINCE_VERSION
  auto output_name = [](void* data, wl_output* output, const char* name) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (auto* o = me->findOutput(output)) {
      o->name = name;
    }
    if (me->mDisplay->GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Wayland output: %s", name);
    }
  };
  auto output_description = [](void* data, wl_output* output, const char* description) {};
#endif
  mOutputListener = wl_output_listener{
    .geometry = output_geometry,
    .mode = output_mode,
    .done = output_done,
    .scale = output_scale,
#ifdef WL_OUTPUT_NAME_SINCE_VERSION
    .name = output_name,
    .description = output_description,
#endif
  };

  auto registry_global = [](void* data, wl_registry* registry, uint32_t name, const char* interface, uint32_t version) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (me->mDisplay->GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Available interface %s (version %u)", interface, version);
    }
    auto bindVersion = [version](const wl_interface& iface, uint32_t maxVersion) { return std::min<uint32_t>({version, (uint32_t)iface.version, maxVersion}); }; // Highest version supported by compositor, generated headers, and this code
    if (strcmp(interface, wl_output_interface.name) == 0) {
      uint32_t v = bindVersion(wl_output_interface, 4);
      wl_output* output = (wl_output*)wl_registry_bind(registry, name, &wl_output_interface, v);
      me->mOutputs.emplace_back(OutputInfo{output, name, v, 1, ""});
      wl_output_add_listener(output, &me->mOutputListener, me);
    } else if (strcmp(interface, wl_compositor_interface.name) == 0) {
      me->mCompositor = (wl_compositor*)wl_registry_bind(registry, name, &wl_compositor_interface, bindVersion(wl_compositor_interface, 6));
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
      me->mXdgBase = (xdg_wm_base*)wl_registry_bind(registry, name, &xdg_wm_base_interface, bindVersion(xdg_wm_base_interface, 6));
      xdg_wm_base_add_listener(me->mXdgBase, &me->mXdgWmBaseListener, me);
    } else if (strcmp(interface, wl_seat_interface.name) == 0 && !me->mSeat) {
      me->mSeatVersion = bindVersion(wl_seat_interface, 8);
      me->mSeat = (wl_seat*)wl_registry_bind(registry, name, &wl_seat_interface, me->mSeatVersion);
      wl_seat_add_listener(me->mSeat, &me->mSeatListener, me);
    } else if (strcmp(interface, zxdg_decoration_manager_v1_interface.name) == 0) {
      me->mDecManager = (zxdg_decoration_manager_v1*)wl_registry_bind(registry, name, &zxdg_decoration_manager_v1_interface, 1);
    } else if (strcmp(interface, wp_cursor_shape_manager_v1_interface.name) == 0) {
      me->mCursorShapeManager = (wp_cursor_shape_manager_v1*)wl_registry_bind(registry, name, &wp_cursor_shape_manager_v1_interface, 1);
    } else if (strcmp(interface, wp_viewporter_interface.name) == 0) {
      me->mViewporter = (wp_viewporter*)wl_registry_bind(registry, name, &wp_viewporter_interface, 1);
    } else if (strcmp(interface, wp_fractional_scale_manager_v1_interface.name) == 0) {
      me->mFractionalScaleManager = (wp_fractional_scale_manager_v1*)wl_registry_bind(registry, name, &wp_fractional_scale_manager_v1_interface, 1);
    }
  };
  auto registry_global_remove = [](void* data, wl_registry* registry, uint32_t name) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    for (auto it = me->mOutputs.begin(); it != me->mOutputs.end(); it++) {
      if (it->globalName == name) {
        std::erase(me->mSurfaceOutputs, it->output);
        if (it->version >= WL_OUTPUT_RELEASE_SINCE_VERSION) {
          wl_output_release(it->output);
        } else {
          wl_output_destroy(it->output);
        }
        me->mOutputs.erase(it);
        me->updateScale();
        break;
      }
    }
  };
  mRegistryListener = wl_registry_listener{.global = registry_global, .global_remove = registry_global_remove};

  wl_registry_add_listener(mRegistry, &mRegistryListener, this);
  if (wl_display_roundtrip(mWayland) == -1) {
    reportError("initial roundtrip");
    throw std::runtime_error("Wayland initialization failed");
  }

  if (mCompositor == nullptr || mXdgBase == nullptr || mSeat == nullptr) {
    throw std::runtime_error("Error getting wayland objects");
  }

  mSurface = wl_compositor_create_surface(mCompositor);
  if (mSurface == nullptr) {
    throw std::runtime_error("Error creating wayland surface");
  }

  auto surface_enter = [](void* data, wl_surface* surface, wl_output* output) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mSurfaceOutputs.emplace_back(output);
    me->updateScale();
  };
  auto surface_leave = [](void* data, wl_surface* surface, wl_output* output) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    std::erase(me->mSurfaceOutputs, output);
    me->updateScale();
  };
#ifdef WL_SURFACE_PREFERRED_BUFFER_SCALE_SINCE_VERSION
  auto surface_preferred_buffer_scale = [](void* data, wl_surface* surface, int32_t factor) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mPreferredScale = factor;
    me->updateScale();
  };
  auto surface_preferred_buffer_transform = [](void* data, wl_surface* surface, uint32_t transform) {};
#endif
  mSurfaceListener = wl_surface_listener{
    .enter = surface_enter,
    .leave = surface_leave,
#ifdef WL_SURFACE_PREFERRED_BUFFER_SCALE_SINCE_VERSION
    .preferred_buffer_scale = surface_preferred_buffer_scale,
    .preferred_buffer_transform = surface_preferred_buffer_transform,
#endif
  };
  wl_surface_add_listener(mSurface, &mSurfaceListener, this);

  if (mViewporter && mFractionalScaleManager) {
    auto fractional_scale_preferred_scale = [](void* data, wp_fractional_scale_v1* fractional_scale, uint32_t scale) {
      GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
      me->mFractionalScale120 = scale;
      me->updateScale();
    };
    mFractionalScaleListener = wp_fractional_scale_v1_listener{.preferred_scale = fractional_scale_preferred_scale};
    mViewport = wp_viewporter_get_viewport(mViewporter, mSurface);
    mFractionalScale = wp_fractional_scale_manager_v1_get_fractional_scale(mFractionalScaleManager, mSurface);
    wp_fractional_scale_v1_add_listener(mFractionalScale, &mFractionalScaleListener, this);
  }

  mXdgSurface = xdg_wm_base_get_xdg_surface(mXdgBase, mSurface);
  if (mXdgSurface == nullptr) {
    throw std::runtime_error("Error creating wayland xdg surface");
  }
  mXdgToplevel = xdg_surface_get_toplevel(mXdgSurface);

  auto xdg_toplevel_handle_configure = [](void* data, xdg_toplevel* toplevel, int32_t width, int32_t height, wl_array* states) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    if (me->mDisplay->GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Wayland surface resized to %d %d", width, height);
    }
    me->mWidthRequested = width;
    me->mHeightRequested = height;
    bool fullScreen = false, maximized = false;
    me->mSuspended = false;
    const uint32_t* state = (const uint32_t*)states->data;
    for (size_t i = 0; i < states->size / sizeof(uint32_t); i++) {
      if (state[i] == XDG_TOPLEVEL_STATE_FULLSCREEN) {
        fullScreen = true;
      } else if (state[i] == XDG_TOPLEVEL_STATE_MAXIMIZED) {
        maximized = true;
#ifdef XDG_TOPLEVEL_STATE_SUSPENDED_SINCE_VERSION
      } else if (state[i] == XDG_TOPLEVEL_STATE_SUSPENDED) {
        me->mSuspended = true;
#endif
      }
    }
    me->setWindowState(fullScreen, maximized);
    me->mStateFullScreenOrMaximized = fullScreen || maximized;
  };

  auto xdg_surface_handle_configure = [](void* data, xdg_surface* surface, uint32_t serial) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    xdg_surface_ack_configure(me->mXdgSurface, serial);
    if (me->mWidthRequested && me->mHeightRequested) {
      me->mWidthLogical = me->mWidthRequested;
      me->mHeightLogical = me->mHeightRequested;
      if (!me->mStateFullScreenOrMaximized) {
        me->mWidthWindowed = me->mWidthRequested;
        me->mHeightWindowed = me->mHeightRequested;
      }
    } else if (!me->mStateFullScreenOrMaximized) {
      me->mWidthLogical = me->mWidthWindowed;
      me->mHeightLogical = me->mHeightWindowed;
      if (me->mBoundsWidth && me->mBoundsHeight) {
        me->mWidthLogical = std::min(me->mWidthLogical, me->mBoundsWidth);
        me->mHeightLogical = std::min(me->mHeightLogical, me->mBoundsHeight);
      }
    }
    me->updateSize();
    me->mWidthRequested = me->mHeightRequested = 0;
  };

  auto xdg_toplevel_handle_close = [](void* data, xdg_toplevel* toplevel) {
    ((GPUDisplayFrontendWayland*)data)->mDisplayControl = 2;
  };
#ifdef XDG_TOPLEVEL_CONFIGURE_BOUNDS_SINCE_VERSION
  auto xdg_toplevel_handle_configure_bounds = [](void* data, xdg_toplevel* toplevel, int32_t width, int32_t height) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mBoundsWidth = width;
    me->mBoundsHeight = height;
  };
#endif
#ifdef XDG_TOPLEVEL_WM_CAPABILITIES_SINCE_VERSION
  auto xdg_toplevel_handle_wm_capabilities = [](void* data, xdg_toplevel* toplevel, wl_array* capabilities) {
    GPUDisplayFrontendWayland* me = (GPUDisplayFrontendWayland*)data;
    me->mCanFullscreen = me->mCanMaximize = false;
    const uint32_t* cap = (const uint32_t*)capabilities->data;
    for (size_t i = 0; i < capabilities->size / sizeof(uint32_t); i++) {
      me->mCanFullscreen |= cap[i] == XDG_TOPLEVEL_WM_CAPABILITIES_FULLSCREEN;
      me->mCanMaximize |= cap[i] == XDG_TOPLEVEL_WM_CAPABILITIES_MAXIMIZE;
    }
  };
#endif

  mXdgSurfaceListener = xdg_surface_listener{
    .configure = xdg_surface_handle_configure,
  };

  mXdgToplevelListener = xdg_toplevel_listener{
    .configure = xdg_toplevel_handle_configure,
    .close = xdg_toplevel_handle_close,
#ifdef XDG_TOPLEVEL_CONFIGURE_BOUNDS_SINCE_VERSION
    .configure_bounds = xdg_toplevel_handle_configure_bounds,
#endif
#ifdef XDG_TOPLEVEL_WM_CAPABILITIES_SINCE_VERSION
    .wm_capabilities = xdg_toplevel_handle_wm_capabilities,
#endif
  };

  xdg_surface_add_listener(mXdgSurface, &mXdgSurfaceListener, this);
  xdg_toplevel_add_listener(mXdgToplevel, &mXdgToplevelListener, this);

  xdg_toplevel_set_title(mXdgToplevel, DISPLAY_WINDOW_NAME);
  xdg_toplevel_set_min_size(mXdgToplevel, MIN_WIDTH, MIN_HEIGHT); // Logical size, avoids degenerate swapchains / projections
  xdg_toplevel_set_app_id(mXdgToplevel, "o2-gpu-event-display");  // Used by the compositor for window grouping, window rules, and matching a .desktop file
  applyInitialWindowState();                                      // Must be requested before the first commit

  if (mDecManager) { // Must be requested before the first commit. Without the protocol (e.g. GNOME), the window has no decorations
    mXdgDecoration = zxdg_decoration_manager_v1_get_toplevel_decoration(mDecManager, mXdgToplevel);
    zxdg_toplevel_decoration_v1_set_mode(mXdgDecoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
  } else if (mDisplay->GetProcessingSettings().debugLevel >= 0) {
    GPUInfo("Wayland compositor does not support server-side decorations");
  }

  wl_surface_commit(mSurface); // Initial commit without buffer, the compositor answers with the first configure
  if (wl_display_roundtrip(mWayland) == -1) {
    reportError("initial roundtrip");
    throw std::runtime_error("Wayland initialization failed");
  }

  if (InitDisplay()) {
    return (1);
  }

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

  while (mDisplayControl != 2) {
    bool error = false;
    while (!error && wl_display_prepare_read(mWayland) != 0) {
      error = wl_display_dispatch_pending(mWayland) == -1;
    }
    if (error) {
      reportError("dispatching events");
      break;
    }
    wl_display_flush(mWayland);
    pollfd pfd = {wl_display_get_fd(mWayland), POLLIN, 0};
    if (poll(&pfd, 1, 0) > 0) {
      if (wl_display_read_events(mWayland) == -1) {
        reportError("reading events");
        break;
      }
    } else {
      wl_display_cancel_read(mWayland);
    }
    if (wl_display_dispatch_pending(mWayland) == -1) {
      reportError("dispatching events");
      break;
    }
    if (mDisplayControl == 2) {
      break;
    }
    HandleSendKeys();
    handleKeyRepeat();
    if (!mSuspended) { // Window not visible, compositor would not release swapchain images
      DrawGLScene();
    }
    waitForFPSLimit();
  }

  ExitDisplay();

  if (mXKBkeymap) {
    xkb_state_unref(mXKBstate);
    xkb_keymap_unref(mXKBkeymap);
    mXKBkeymap = nullptr;
  }
  xkb_context_unref(mXKBcontext);

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  if (mCursorShapeDevice) {
    wp_cursor_shape_device_v1_destroy(mCursorShapeDevice);
    mCursorShapeDevice = nullptr;
  }
  if (mCursorShapeManager) {
    wp_cursor_shape_manager_v1_destroy(mCursorShapeManager);
    mCursorShapeManager = nullptr;
  }
  if (mPointer) {
    mSeatVersion >= WL_POINTER_RELEASE_SINCE_VERSION ? wl_pointer_release(mPointer) : wl_pointer_destroy(mPointer);
    mPointer = nullptr;
  }
  if (mKeyboard) {
    mSeatVersion >= WL_KEYBOARD_RELEASE_SINCE_VERSION ? wl_keyboard_release(mKeyboard) : wl_keyboard_destroy(mKeyboard);
    mKeyboard = nullptr;
  }
  if (mXdgDecoration) {
    zxdg_toplevel_decoration_v1_destroy(mXdgDecoration);
    mXdgDecoration = nullptr;
  }
  if (mFractionalScale) {
    wp_fractional_scale_v1_destroy(mFractionalScale);
    mFractionalScale = nullptr;
  }
  if (mViewport) {
    wp_viewport_destroy(mViewport);
    mViewport = nullptr;
  }
  xdg_toplevel_destroy(mXdgToplevel);
  xdg_surface_destroy(mXdgSurface);
  wl_surface_destroy(mSurface);
  if (mFractionalScaleManager) {
    wp_fractional_scale_manager_v1_destroy(mFractionalScaleManager);
    mFractionalScaleManager = nullptr;
  }
  if (mViewporter) {
    wp_viewporter_destroy(mViewporter);
    mViewporter = nullptr;
  }
  if (mDecManager) {
    zxdg_decoration_manager_v1_destroy(mDecManager);
    mDecManager = nullptr;
  }
  for (auto& o : mOutputs) {
    if (o.version >= WL_OUTPUT_RELEASE_SINCE_VERSION) {
      wl_output_release(o.output);
    } else {
      wl_output_destroy(o.output);
    }
  }
  mOutputs.clear();
  mSurfaceOutputs.clear();
  if (mSeat) {
    if (mSeatVersion >= WL_SEAT_RELEASE_SINCE_VERSION) {
      wl_seat_release(mSeat);
    } else {
      wl_seat_destroy(mSeat);
    }
    mSeat = nullptr;
  }
  xdg_wm_base_destroy(mXdgBase);
  wl_compositor_destroy(mCompositor);
  wl_registry_destroy(mRegistry);
  wl_display_disconnect(mWayland);

  return (0);
}

void GPUDisplayFrontendWayland::DisplayExit()
{
  pthread_mutex_lock(&mSemLockExit);
  if (mDisplayRunning) {
    mDisplayControl = 2;
  }
  pthread_mutex_unlock(&mSemLockExit);
}

void GPUDisplayFrontendWayland::SwitchFullscreen(bool set)
{
  if (!mCanFullscreen) {
    GPUWarning("Wayland compositor does not support fullscreen");
  } else if (set) {
    xdg_toplevel_set_fullscreen(mXdgToplevel, mSurfaceOutputs.size() ? mSurfaceOutputs.back() : nullptr);
  } else {
    xdg_toplevel_unset_fullscreen(mXdgToplevel);
  }
}

void GPUDisplayFrontendWayland::ToggleMaximized(bool set)
{
  if (!mCanMaximize) {
    GPUWarning("Wayland compositor does not support maximizing windows");
  } else if (set) {
    xdg_toplevel_set_maximized(mXdgToplevel);
  } else {
    xdg_toplevel_unset_maximized(mXdgToplevel);
  }
}

GPUDisplayFrontendWayland::OutputInfo* GPUDisplayFrontendWayland::findOutput(wl_output* output)
{
  auto it = std::find_if(mOutputs.begin(), mOutputs.end(), [output](const auto& o) { return o.output == output; });
  return it == mOutputs.end() ? nullptr : &*it;
}

void GPUDisplayFrontendWayland::updateScale()
{
  double scale;
  if (mFractionalScale) {
    scale = mFractionalScale120 ? mFractionalScale120 / 120. : 1.;
  } else {
    int32_t intScale = mPreferredScale;
    if (intScale == 0) {
      intScale = 1;
      for (auto* output : mSurfaceOutputs) {
        if (auto* o = findOutput(output)) {
          intScale = std::max(intScale, o->scale);
        }
      }
    }
    scale = intScale;
  }
  if (scale != mScale && mSurface) {
    mScale = scale;
    if (!mFractionalScale) {
      wl_surface_set_buffer_scale(mSurface, (int32_t)mScale); // Applied with the next commit, i.e. the next present, which happens after the swapchain was recreated with the new size by ResizeScene
    }
    updateSize();
  }
}

void GPUDisplayFrontendWayland::updateSize()
{
  if (mViewport) {
    wp_viewport_set_destination(mViewport, mWidthLogical, mHeightLogical); // Surface size in logical coordinates, independent of the buffer size, applied with the next present
  }
  const int32_t width = mFractionalScale ? (int32_t)std::lround(mWidthLogical * mScale) : (mWidthLogical * (int32_t)mScale);
  const int32_t height = mFractionalScale ? (int32_t)std::lround(mHeightLogical * mScale) : (mHeightLogical * (int32_t)mScale);
  if (width != mDisplayWidth || height != mDisplayHeight) {
    if (mDisplayRunning) {
      ResizeScene(width, height); // Swapchain is recreated by the backend, Vulkan WSI commits the new buffer
    } else {
      mDisplayWidth = width; // Initial size, used by InitDisplay via getSize()
      mDisplayHeight = height;
    }
  }
}

void GPUDisplayFrontendWayland::reportError(const char* where)
{
  const int32_t err = wl_display_get_error(mWayland);
  if (err == EPROTO) {
    const wl_interface* iface = nullptr;
    uint32_t id = 0;
    const uint32_t code = wl_display_get_protocol_error(mWayland, &iface, &id);
    GPUError("Wayland protocol error %u on %s@%u while %s", code, iface ? iface->name : "unknown interface", id, where);
  } else {
    GPUError("Wayland error while %s: %s", where, strerror(err));
  }
}

void GPUDisplayFrontendWayland::handleKeyRepeat()
{
  if (mRepeatKeyCode < 0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  while (now >= mRepeatNext && mRepeatKeyCode >= 0) {
    HandleKey(mRepeatSymbol);
    mRepeatNext += std::chrono::microseconds(1000000 / mRepeatRate);
  }
}

void GPUDisplayFrontendWayland::applyScroll()
{
  mMouseWheel += mHasAxisValue120 ? (mAxisValue120 * -100 / 120) : (int32_t)(mAxisValue * (-100. / 15.));
  mAxisValue = 0;
  mAxisValue120 = 0;
  mHasAxisValue120 = false;
}

void GPUDisplayFrontendWayland::SetVSync(bool enable)
{
}

void GPUDisplayFrontendWayland::getSize(int32_t& width, int32_t& height)
{
  width = mDisplayWidth;
  height = mDisplayHeight;
}

int32_t GPUDisplayFrontendWayland::getVulkanSurface(void* instance, void* surface)
{
  VkWaylandSurfaceCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
  info.flags = 0;
  info.display = mWayland;
  info.surface = mSurface;
  return vkCreateWaylandSurfaceKHR(*(VkInstance*)instance, &info, nullptr, (VkSurfaceKHR*)surface) != VK_SUCCESS;
}

uint32_t GPUDisplayFrontendWayland::getReqVulkanExtensions(const char**& p)
{
  static const char* exts[] = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
  p = exts;
  return 2;
}
