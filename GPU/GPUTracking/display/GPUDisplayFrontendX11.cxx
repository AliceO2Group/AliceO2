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
#include "GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <chrono>

#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xlib.h>
#endif

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayFrontendX11::GPUDisplayFrontendX11()
{
  mFrontendType = TYPE_X11;
  mFrontendName = "X11";
}

int GPUDisplayFrontendX11::GetKey(int key)
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

void GPUDisplayFrontendX11::GetKey(XEvent& event, int& keyOut, int& keyPressOut)
{
  char tmpString[9];
  KeySym sym;
  if (XLookupString(&event.xkey, tmpString, 8, &sym, nullptr) == 0) {
    tmpString[0] = 0;
  }
  int specialKey = GetKey(sym);
  int localeKey = (unsigned char)tmpString[0];
  // GPUInfo("Key: keycode %d -> sym %d (%c) key %d (%c) special %d (%c)", (int)event.xkey.keycode, (int)sym, (char)sym, (int)localeKey, (char)localeKey, (int)specialKey, (char)specialKey);

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

int GPUDisplayFrontendX11::FrontendMain()
{
  XSetWindowAttributes windowAttributes;
  XVisualInfo* visualInfo = nullptr;
  XEvent event;
  Colormap colorMap;
  GLXContext glxContext = nullptr;
  int errorBase;
  int eventBase;

  // Open a connection to the X server
  mDisplay = XOpenDisplay(nullptr);

  if (mDisplay == nullptr) {
    GPUError("could not open display");
    return (-1);
  }

  // Make sure OpenGL's GLX extension supported
  if (!glXQueryExtension(mDisplay, &errorBase, &eventBase)) {
    GPUError("X server has no OpenGL GLX extension");
    return (-1);
  }

  const char* glxExt = glXQueryExtensionsString(mDisplay, DefaultScreen(mDisplay));
  if (strstr(glxExt, "GLX_EXT_swap_control") == nullptr) {
    GPUError("No vsync support!");
    vsync_supported = false;
  } else {
    vsync_supported = true;
  }

  // Require MSAA, double buffering, and Depth buffer
  int attribs[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 24, GLX_STENCIL_SIZE, 8, GLX_DOUBLEBUFFER, True,
                   // GLX_SAMPLE_BUFFERS  , 1, //Disable MSAA here, we do it by rendering to offscreenbuffer
                   // GLX_SAMPLES         , MSAA_SAMPLES,
                   None};

  GLXFBConfig fbconfig = nullptr;
  int fbcount;
  GLXFBConfig* fbc = glXChooseFBConfig(mDisplay, DefaultScreen(mDisplay), attribs, &fbcount);
  if (fbc == nullptr || fbcount == 0) {
    GPUError("Failed to get MSAA GLXFBConfig");
    return (-1);
  }
  fbconfig = fbc[0];
  XFree(fbc);
  visualInfo = glXGetVisualFromFBConfig(mDisplay, fbconfig);

  if (visualInfo == nullptr) {
    GPUError("no RGB visual with depth buffer");
    return (-1);
  }

  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    // Create an OpenGL rendering context
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
    if (glXCreateContextAttribsARB) {
      int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, GL_MIN_VERSION_MAJOR,
        GLX_CONTEXT_MINOR_VERSION_ARB, GL_MIN_VERSION_MINOR,
        GLX_CONTEXT_PROFILE_MASK_ARB, mBackend->CoreProfile() ? GLX_CONTEXT_CORE_PROFILE_BIT_ARB : GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
        None};
      glxContext = glXCreateContextAttribsARB(mDisplay, fbconfig, nullptr, GL_TRUE, context_attribs);
    } else {
      glxContext = glXCreateContext(mDisplay, visualInfo, nullptr, GL_TRUE);
    }
    if (glxContext == nullptr) {
      GPUError("could not create rendering context");
      return (-1);
    }
  }

  Window win = RootWindow(mDisplay, visualInfo->screen);
  colorMap = XCreateColormap(mDisplay, win, visualInfo->visual, AllocNone);
  windowAttributes.colormap = colorMap;
  windowAttributes.border_pixel = 0;
  windowAttributes.event_mask = ExposureMask | VisibilityChangeMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | SubstructureNotifyMask | FocusChangeMask;

  // Create an X window with the selected visual
  mWindow = XCreateWindow(mDisplay, win, 50, 50, INIT_WIDTH, INIT_HEIGHT, // Position / Width and height of window
                          0, visualInfo->depth, InputOutput, visualInfo->visual, CWBorderPixel | CWColormap | CWEventMask, &windowAttributes);
  XSetStandardProperties(mDisplay, mWindow, DISPLAY_WINDOW_NAME, DISPLAY_WINDOW_NAME, None, nullptr, 0, nullptr);
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glXMakeCurrent(mDisplay, mWindow, glxContext);
  }
  XMapWindow(mDisplay, mWindow);

  // Maximize window
  ToggleMaximized(true);

  // Receive signal when window closed
  Atom WM_DELETE_WINDOW = XInternAtom(mDisplay, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(mDisplay, mWindow, &WM_DELETE_WINDOW, 1);
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
    font_info = XLoadQueryFont(mDisplay, f);
    if (!font_info) {
      GPUError("XLoadQueryFont failed.");
      return (-1);
    } else {
      int first = font_info->min_char_or_byte2;
      int last = font_info->max_char_or_byte2;
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

  XMapWindow(mDisplay, mWindow);
  XFlush(mDisplay);
  int x11_fd = ConnectionNumber(mDisplay);

  // Enable vsync
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL && vsync_supported) {
    mGlXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT");
    if (mGlXSwapIntervalEXT == nullptr) {
      GPUError("Cannot enable vsync");
      return (-1);
    }
    mGlXSwapIntervalEXT(mDisplay, glXGetCurrentDrawable(), 1);
  }

  if (InitDisplay()) {
    return (1);
  }

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  while (1) {
    int num_ready_fds;
    struct timeval tv;
    fd_set in_fds;
    int waitCount = 0;
    do {
      FD_ZERO(&in_fds);
      FD_SET(x11_fd, &in_fds);
      tv.tv_usec = 10000;
      tv.tv_sec = 0;
      std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
      bool allowMax = mMaxFPSRate && std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() < 0.01;
      t1 = t2;
      num_ready_fds = allowMax || XPending(mDisplay) || select(x11_fd + 1, &in_fds, nullptr, nullptr, &tv);
      if (num_ready_fds < 0) {
        GPUError("Error (num_ready_fds)");
      }
      if (mDisplayControl == 2) {
        break;
      }
      if (mSendKey) {
        mNeedUpdate = 1;
      }
      if (waitCount++ != 100) {
        mNeedUpdate = 1;
      }
    } while (!(num_ready_fds || mNeedUpdate));
    mNeedUpdate = 0;

    do {
      if (mDisplayControl == 2) {
        break;
      }
      HandleSendKey();
      if (!XPending(mDisplay)) {
        event.type = Expose;
      } else {
        XNextEvent(mDisplay, &event);
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
            } else {
              mMouseDnR = false;
            }
          }
          break;
        }

        case KeyPress: {
          int handleKey = 0, keyPress = 0;
          GetKey(event, handleKey, keyPress);
          mKeysShift[keyPress] = mKeys[KEY_SHIFT];
          mKeys[keyPress] = true;
          HandleKey(handleKey);
          break;
        }

        case KeyRelease: {
          int handleKey = 0, keyPress = 0;
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

        case Expose: {
          break;
        }

        case ConfigureNotify: {
          ResizeScene(event.xconfigure.width, event.xconfigure.height);
          break;
        }

        case ClientMessage: {
          if (event.xclient.message_type == XInternAtom(mDisplay, "_NET_WM_STATE", False)) {
            XFlush(mDisplay);
          } else {
            mDisplayControl = 2;
          }
          break;
        }
      }
    } while (XPending(mDisplay)); // Loop to compress events
    if (mDisplayControl == 2) {
      break;
    }

    DrawGLScene();
    if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
      glXSwapBuffers(mDisplay, mWindow);
    }
  }

#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glDeleteLists(mFontBase, 256);
    XUnloadFont(mDisplay, font_info->fid);
  }
#endif
  ExitDisplay();
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glXDestroyContext(mDisplay, glxContext);
  }
  XFree(visualInfo);
  XDestroyWindow(mDisplay, mWindow);
  XCloseDisplay(mDisplay);

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  return (0);
}

void GPUDisplayFrontendX11::DisplayExit()
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

void GPUDisplayFrontendX11::SwitchFullscreen(bool set)
{
  XEvent xev;
  memset(&xev, 0, sizeof(xev));
  xev.type = ClientMessage;
  xev.xclient.window = mWindow;
  xev.xclient.message_type = XInternAtom(mDisplay, "_NET_WM_STATE", False);
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = 2; // _NET_WM_STATE_TOGGLE
  xev.xclient.data.l[1] = XInternAtom(mDisplay, "_NET_WM_STATE_FULLSCREEN", True);
  xev.xclient.data.l[2] = 0;
  XSendEvent(mDisplay, DefaultRootWindow(mDisplay), False, SubstructureNotifyMask, &xev);
}

void GPUDisplayFrontendX11::ToggleMaximized(bool set)
{
  XEvent xev;
  memset(&xev, 0, sizeof(xev));
  xev.type = ClientMessage;
  xev.xclient.window = mWindow;
  xev.xclient.message_type = XInternAtom(mDisplay, "_NET_WM_STATE", False);
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = set ? 1 : 2; //_NET_WM_STATE_ADD
  xev.xclient.data.l[1] = XInternAtom(mDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
  xev.xclient.data.l[2] = XInternAtom(mDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
  XSendEvent(mDisplay, DefaultRootWindow(mDisplay), False, SubstructureNotifyMask, &xev);
}

void GPUDisplayFrontendX11::SetVSync(bool enable)
{
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL && vsync_supported) {
    mGlXSwapIntervalEXT(mDisplay, glXGetCurrentDrawable(), (int)enable);
  }
}

int GPUDisplayFrontendX11::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, FrontendThreadWrapper, this)) {
    GPUError("Coult not Create frontend Thread...");
    return (1);
  }
  return (0);
}

void GPUDisplayFrontendX11::getSize(int& width, int& height)
{
  Window root_return;
  int x_return, y_return;
  unsigned int width_return, height_return, border_width_return, depth_return;
  if (XGetGeometry(mDisplay, mWindow, &root_return, &x_return, &y_return, &width_return, &height_return, &border_width_return, &depth_return) == 0) {
    throw std::runtime_error("Cannot query X11 window geometry");
  }
  width = width_return;
  height = height_return;
}

int GPUDisplayFrontendX11::getVulkanSurface(void* instance, void* surface)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  VkXlibSurfaceCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  info.flags = 0;
  info.dpy = mDisplay;
  info.window = mWindow;
  return vkCreateXlibSurfaceKHR(*(VkInstance*)instance, &info, nullptr, (VkSurfaceKHR*)surface) != VK_SUCCESS;
#else
  return 1;
#endif
}

unsigned int GPUDisplayFrontendX11::getReqVulkanExtensions(const char**& p)
{
  static const char* exts[] = {"VK_KHR_surface", "VK_KHR_xlib_surface"};
  p = exts;
  return 2;
}
