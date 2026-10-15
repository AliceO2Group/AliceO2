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

/// \file GPUDisplayFrontendX11.cxx
/// \author David Rohr

// Now the other headers
#include "GPUDisplayFrontendX11.h"
#include "backend/GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <unistd.h>
#include <X11/Xatom.h>
#include <X11/Xresource.h>

#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xlib.h>
#endif

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int32_t*);

using namespace o2::gpu;

GPUDisplayFrontendX11::GPUDisplayFrontendX11()
{
  mFrontendType = TYPE_X11;
  mFrontendName = "X11";
}

int32_t GPUDisplayFrontendX11::GetKey(int32_t key)
{
  if (key == 65453) {
    return ('-');
  }
  if (key == 65451) {
    return ('+');
  }
  if (key == 65505 || key == 65506) {
    return (KEY_SHIFT);
  }
  if (key == 65513 || key == 65511) {
    return (KEY_ALT);
  }
  if (key == 65027) {
    return (KEY_RALT);
  }
  if (key == 65507) {
    return (KEY_CTRL);
  }
  if (key == 65508) {
    return (KEY_RCTRL);
  }
  if (key == 65362) {
    return (KEY_UP);
  }
  if (key == 65364) {
    return (KEY_DOWN);
  }
  if (key == 65361) {
    return (KEY_LEFT);
  }
  if (key == 65363) {
    return (KEY_RIGHT);
  }
  if (key == 65365) {
    return (KEY_PAGEUP);
  }
  if (key == 65366) {
    return (KEY_PAGEDOWN);
  }
  if (key == 65307) {
    return (KEY_ESCAPE);
  }
  if (key == 65293) {
    return (KEY_ENTER);
  }
  if (key == 65367) {
    return (KEY_END);
  }
  if (key == 65360) {
    return (KEY_HOME);
  }
  if (key == 65379) {
    return (KEY_INSERT);
  }
  if (key == 65470) {
    return (KEY_F1);
  }
  if (key == 65471) {
    return (KEY_F2);
  }
  if (key == 65472) {
    return (KEY_F3);
  }
  if (key == 65473) {
    return (KEY_F4);
  }
  if (key == 65474) {
    return (KEY_F5);
  }
  if (key == 65475) {
    return (KEY_F6);
  }
  if (key == 65476) {
    return (KEY_F7);
  }
  if (key == 65477) {
    return (KEY_F8);
  }
  if (key == 65478) {
    return (KEY_F9);
  }
  if (key == 65479) {
    return (KEY_F10);
  }
  if (key == 65480) {
    return (KEY_F11);
  }
  if (key == 65481) {
    return (KEY_F12);
  }
  if (key == 32) {
    return (KEY_SPACE);
  }
  if (key > 255) {
    return (0);
  }
  return 0;
}

void GPUDisplayFrontendX11::GetKey(XEvent& event, int32_t& keyOut, int32_t& keyPressOut)
{
  char tmpString[9];
  KeySym sym;
  if (XLookupString(&event.xkey, tmpString, 8, &sym, nullptr) == 0) {
    tmpString[0] = 0;
  }
  int32_t specialKey = GetKey(sym);
  int32_t localeKey = (uint8_t)tmpString[0];
  // GPUInfo("Key: keycode %d -> sym %d (%c) key %d (%c) special %d (%c)", (int32_t)event.xkey.keycode, (int32_t)sym, (char)sym, (int32_t)localeKey, (char)localeKey, (int32_t)specialKey, (char)specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = localeKey;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayFrontendX11::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    if (!fromBotton) {
      y = mDisplayHeight - y;
    }
    glColor4f(r, g, b, a);
    glRasterPos2f(x, y);
    if (!glIsList(mFontBase)) {
      GPUError("print string: Bad display list.");
      exit(1);
    } else if (s && strlen(s)) {
      glPushAttrib(GL_LIST_BIT);
      glListBase(mFontBase);
      glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte*)s);
      glPopAttrib();
    }
  }
#endif
}

int32_t GPUDisplayFrontendX11::FrontendMain()
{
  XSetWindowAttributes windowAttributes;
  XVisualInfo* visualInfo = nullptr;
  XEvent event;
  Colormap colorMap;
  GLXContext glxContext = nullptr;
  int32_t errorBase;
  int32_t eventBase;

  // Open a connection to the X server
  mXDisplay = XOpenDisplay(nullptr);

  if (mXDisplay == nullptr) {
    GPUError("could not open display");
    return (-1);
  }

  XrmInitialize();
  if (const char* resources = XResourceManagerString(mXDisplay)) {
    XrmDatabase db = XrmGetStringDatabase(resources);
    char* type = nullptr;
    XrmValue value;
    if (db && XrmGetResource(db, "Xft.dpi", "Xft.Dpi", &type, &value) && value.addr && atof(value.addr) > 0) {
      mScale = atof(value.addr) / 96.;
    }
    if (db) {
      XrmDestroyDatabase(db);
    }
  }

  // Make sure OpenGL's GLX extension supported
  if (!glXQueryExtension(mXDisplay, &errorBase, &eventBase)) {
    GPUError("X server has no OpenGL GLX extension");
    return (-1);
  }

  const char* glxExt = glXQueryExtensionsString(mXDisplay, DefaultScreen(mXDisplay));
  if (strstr(glxExt, "GLX_EXT_swap_control") == nullptr) {
    GPUError("No vsync support!");
    vsync_supported = false;
  } else {
    vsync_supported = true;
  }

  // Require MSAA, double buffering, and Depth buffer
  int32_t attribs[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 24, GLX_STENCIL_SIZE, 8, GLX_DOUBLEBUFFER, True,
                       // GLX_SAMPLE_BUFFERS  , 1, //Disable MSAA here, we do it by rendering to offscreenbuffer
                       // GLX_SAMPLES         , MSAA_SAMPLES,
                       None};

  GLXFBConfig fbconfig = nullptr;
  int32_t fbcount;
  GLXFBConfig* fbc = glXChooseFBConfig(mXDisplay, DefaultScreen(mXDisplay), attribs, &fbcount);
  if (fbc == nullptr || fbcount == 0) {
    GPUError("Failed to get MSAA GLXFBConfig");
    return (-1);
  }
  fbconfig = fbc[0];
  XFree(fbc);
  visualInfo = glXGetVisualFromFBConfig(mXDisplay, fbconfig);

  if (visualInfo == nullptr) {
    GPUError("no RGB visual with depth buffer");
    return (-1);
  }

  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    // Create an OpenGL rendering context
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
    if (glXCreateContextAttribsARB) {
      int32_t context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, GL_MIN_VERSION_MAJOR,
        GLX_CONTEXT_MINOR_VERSION_ARB, GL_MIN_VERSION_MINOR,
        GLX_CONTEXT_PROFILE_MASK_ARB, mBackend->CoreProfile() ? GLX_CONTEXT_CORE_PROFILE_BIT_ARB : GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
        None};
      glxContext = glXCreateContextAttribsARB(mXDisplay, fbconfig, nullptr, GL_TRUE, context_attribs);
    } else {
      glxContext = glXCreateContext(mXDisplay, visualInfo, nullptr, GL_TRUE);
    }
    if (glxContext == nullptr) {
      GPUError("could not create rendering context");
      return (-1);
    }
  }

  Window win = RootWindow(mXDisplay, visualInfo->screen);
  colorMap = XCreateColormap(mXDisplay, win, visualInfo->visual, AllocNone);
  windowAttributes.colormap = colorMap;
  windowAttributes.border_pixel = 0;
  windowAttributes.event_mask = ExposureMask | VisibilityChangeMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | SubstructureNotifyMask | FocusChangeMask | PropertyChangeMask;

  // Create an X window with the selected visual
  mWindow = XCreateWindow(mXDisplay, win, 50, 50, INIT_WIDTH * mScale, INIT_HEIGHT * mScale, // Position / Width and height of window
                          0, visualInfo->depth, InputOutput, visualInfo->visual, CWBorderPixel | CWColormap | CWEventMask, &windowAttributes);
  XSetStandardProperties(mXDisplay, mWindow, DISPLAY_WINDOW_NAME, DISPLAY_WINDOW_NAME, None, nullptr, 0, nullptr);
  XSizeHints sizeHints{};
  sizeHints.flags = PMinSize;
  sizeHints.min_width = MIN_WIDTH * mScale;
  sizeHints.min_height = MIN_HEIGHT * mScale;
  XSetWMNormalHints(mXDisplay, mWindow, &sizeHints);
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glXMakeCurrent(mXDisplay, mWindow, glxContext);
  }
  XMapWindow(mXDisplay, mWindow);

  applyInitialWindowState();

  // Receive signal when window closed
  Atom WM_DELETE_WINDOW = XInternAtom(mXDisplay, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(mXDisplay, mWindow, &WM_DELETE_WINDOW, 1);
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  XFontStruct* font_info = nullptr;
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    // Prepare fonts
    mFontBase = glGenLists(256);
    if (!glIsList(mFontBase)) {
      GPUError("Out of display lists.");
      return (-1);
    }
    const char* f = "fixed";
    font_info = XLoadQueryFont(mXDisplay, f);
    if (!font_info) {
      GPUError("XLoadQueryFont failed.");
      return (-1);
    } else {
      int32_t first = font_info->min_char_or_byte2;
      int32_t last = font_info->max_char_or_byte2;
      glXUseXFont(font_info->fid, first, last - first + 1, mFontBase + first);
    }
  }
  mCanDrawText = 1;
  if (drawTextFontSize() == 0) {
    drawTextFontSize() = 12;
  }
#endif

  if (mBackend->ExtInit()) {
    fprintf(stderr, "Error initializing backend\n");
    return (-1);
  }

  XMapWindow(mXDisplay, mWindow);
  XFlush(mXDisplay);

  // Enable vsync
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL && vsync_supported) {
    mGlXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT");
    if (mGlXSwapIntervalEXT == nullptr) {
      GPUError("Cannot enable vsync");
      return (-1);
    }
    mGlXSwapIntervalEXT(mXDisplay, glXGetCurrentDrawable(), 1);
  }

  if (InitDisplay()) {
    return (1);
  }

  {
    std::lock_guard lock(mExitStateMutex);
    mDisplayRunning = true;
  }

  while (1) {
    do {
      if (mDisplayControl == 2) {
        break;
      }
      HandleSendKeys();
      if (!XPending(mXDisplay)) {
        event.type = Expose;
      } else {
        XNextEvent(mXDisplay, &event);
      }
      switch (event.type) {
        case ButtonPress: {
          if (event.xbutton.button == 4) {
            mMouseWheel += 100;
          } else if (event.xbutton.button == 5) {
            mMouseWheel -= 100;
          } else {
            if (event.xbutton.button == 1) {
              mMouseDn = true;
            } else if (event.xbutton.button == 2) {
              mMouseDnM = true;
            } else {
              mMouseDnR = true;
            }
            mMouseDnX = event.xmotion.x;
            mMouseDnY = event.xmotion.y;
          }
          break;
        }

        case ButtonRelease: {
          if (event.xbutton.button != 4 && event.xbutton.button != 5) {
            if (event.xbutton.button == 1) {
              mMouseDn = false;
            } else if (event.xbutton.button == 2) {
              mMouseDnM = false;
            } else {
              mMouseDnR = false;
            }
          }
          break;
        }

        case KeyPress: {
          int32_t handleKey = 0, keyPress = 0;
          GetKey(event, handleKey, keyPress);
          mKeysShift[keyPress] = mKeys[KEY_SHIFT];
          mKeys[keyPress] = true;
          HandleKey(handleKey);
          break;
        }

        case KeyRelease: {
          int32_t handleKey = 0, keyPress = 0;
          GetKey(event, handleKey, keyPress);
          mKeys[keyPress] = false;
          mKeysShift[keyPress] = false;
          break;
        }

        case MotionNotify: {
          mMouseMvX = event.xmotion.x;
          mMouseMvY = event.xmotion.y;
          break;
        }

        case FocusOut: {
          focusLost();
          break;
        }

        case PropertyNotify: {
          if (event.xproperty.atom == XInternAtom(mXDisplay, "_NET_WM_STATE", False)) {
            updateWindowState();
          }
          break;
        }

        case Expose: {
          break;
        }

        case ConfigureNotify: {
          ResizeScene(event.xconfigure.width, event.xconfigure.height);
          break;
        }

        case ClientMessage: {
          if (event.xclient.message_type == XInternAtom(mXDisplay, "_NET_WM_STATE", False)) {
            XFlush(mXDisplay);
          } else {
            mDisplayControl = 2;
          }
          break;
        }
      }
    } while (XPending(mXDisplay)); // Loop to compress events
    if (mDisplayControl == 2) {
      break;
    }

    DrawGLScene();
    if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
      glXSwapBuffers(mXDisplay, mWindow);
    }
    waitForFPSLimit();
  }

#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glDeleteLists(mFontBase, 256);
    XUnloadFont(mXDisplay, font_info->fid);
  }
#endif
  ExitDisplay();
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glXDestroyContext(mXDisplay, glxContext);
  }
  XFree(visualInfo);
  XDestroyWindow(mXDisplay, mWindow);
  XCloseDisplay(mXDisplay);

  {
    std::lock_guard lock(mExitStateMutex);
    mDisplayRunning = false;
  }

  return (0);
}

void GPUDisplayFrontendX11::updateWindowState()
{
  Atom type;
  int32_t format;
  unsigned long nItems, bytesAfter;
  unsigned char* data = nullptr;
  if (XGetWindowProperty(mXDisplay, mWindow, XInternAtom(mXDisplay, "_NET_WM_STATE", False), 0, 1024, False, XA_ATOM, &type, &format, &nItems, &bytesAfter, &data) != Success || data == nullptr) {
    return;
  }
  const Atom fullScreen = XInternAtom(mXDisplay, "_NET_WM_STATE_FULLSCREEN", False), maxHorz = XInternAtom(mXDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False), maxVert = XInternAtom(mXDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
  bool isFullScreen = false, isMaxHorz = false, isMaxVert = false;
  for (unsigned long i = 0; i < nItems; i++) {
    const Atom a = ((Atom*)data)[i];
    isFullScreen |= a == fullScreen;
    isMaxHorz |= a == maxHorz;
    isMaxVert |= a == maxVert;
  }
  XFree(data);
  setWindowState(isFullScreen, isMaxHorz && isMaxVert);
}

void GPUDisplayFrontendX11::SwitchFullscreen(bool set)
{
  XEvent xev;
  memset(&xev, 0, sizeof(xev));
  xev.type = ClientMessage;
  xev.xclient.window = mWindow;
  xev.xclient.message_type = XInternAtom(mXDisplay, "_NET_WM_STATE", False);
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = 2; // _NET_WM_STATE_TOGGLE
  xev.xclient.data.l[1] = XInternAtom(mXDisplay, "_NET_WM_STATE_FULLSCREEN", True);
  xev.xclient.data.l[2] = 0;
  XSendEvent(mXDisplay, DefaultRootWindow(mXDisplay), False, SubstructureNotifyMask, &xev);
}

void GPUDisplayFrontendX11::ToggleMaximized(bool set)
{
  XEvent xev;
  memset(&xev, 0, sizeof(xev));
  xev.type = ClientMessage;
  xev.xclient.window = mWindow;
  xev.xclient.message_type = XInternAtom(mXDisplay, "_NET_WM_STATE", False);
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = set ? 1 : 2; //_NET_WM_STATE_ADD
  xev.xclient.data.l[1] = XInternAtom(mXDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
  xev.xclient.data.l[2] = XInternAtom(mXDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
  XSendEvent(mXDisplay, DefaultRootWindow(mXDisplay), False, SubstructureNotifyMask, &xev);
}

void GPUDisplayFrontendX11::SetVSync(bool enable)
{
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL && vsync_supported) {
    mGlXSwapIntervalEXT(mXDisplay, glXGetCurrentDrawable(), (int32_t)enable);
  }
}

void GPUDisplayFrontendX11::getSize(int32_t& width, int32_t& height)
{
  Window root_return;
  int32_t x_return, y_return;
  uint32_t width_return, height_return, border_width_return, depth_return;
  if (XGetGeometry(mXDisplay, mWindow, &root_return, &x_return, &y_return, &width_return, &height_return, &border_width_return, &depth_return) == 0) {
    throw std::runtime_error("Cannot query X11 window geometry");
  }
  width = width_return;
  height = height_return;
}

int32_t GPUDisplayFrontendX11::getVulkanSurface(void* instance, void* surface)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  VkXlibSurfaceCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  info.flags = 0;
  info.dpy = mXDisplay;
  info.window = mWindow;
  return vkCreateXlibSurfaceKHR(*(VkInstance*)instance, &info, nullptr, (VkSurfaceKHR*)surface) != VK_SUCCESS;
#else
  return 1;
#endif
}

uint32_t GPUDisplayFrontendX11::getReqVulkanExtensions(const char**& p)
{
  static const char* exts[] = {"VK_KHR_surface", "VK_KHR_xlib_surface"};
  p = exts;
  return 2;
}
