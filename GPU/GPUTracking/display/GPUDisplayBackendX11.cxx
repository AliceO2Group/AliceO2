// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendX11.cxx
/// \author David Rohr

// GL EXT must be the first header
#include "GPUDisplayExt.h"

// Now the other headers
#include "GPUDisplayBackendX11.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

using namespace GPUCA_NAMESPACE::gpu;

int GPUDisplayBackendX11::GetKey(int key)
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
  if (key == 65513 || key == 65027) {
    return (KEY_ALT);
  }
  if (key == 65507 || key == 65508) {
    return (KEY_CTRL);
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

void GPUDisplayBackendX11::GetKey(XEvent& event, int& keyOut, int& keyPressOut)
{
  char tmpString[9];
  KeySym sym;
  if (XLookupString(&event.xkey, tmpString, 8, &sym, nullptr) == 0) {
    tmpString[0] = 0;
  }
  int specialKey = GetKey(sym);
  int localeKey = tmpString[0];
  // GPUInfo("Key: keycode %d -> sym %d (%c) key %d (%c) special %d (%c)", event.xkey.keycode, (int) sym, (char) sym, (int) localeKey, localeKey, specialKey, (char) specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = localeKey;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayBackendX11::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
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
#endif
}

int GPUDisplayBackendX11::OpenGLMain()
{
  XSetWindowAttributes windowAttributes;
  XVisualInfo* visualInfo = nullptr;
  XEvent event;
  Colormap colorMap;
  GLXContext glxContext;
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
                   //		GLX_SAMPLE_BUFFERS  , 1, //Disable MSAA here, we do it by rendering to offscreenbuffer
                   //		GLX_SAMPLES         , MSAA_SAMPLES,
                   None};

  GLXFBConfig fbconfig = 0;
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

  // Create an OpenGL rendering context
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
  if (glXCreateContextAttribsARB) {
    int context_attribs[] = {
      GLX_CONTEXT_MAJOR_VERSION_ARB, GL_MIN_VERSION_MAJOR,
      GLX_CONTEXT_MINOR_VERSION_ARB, GL_MIN_VERSION_MINOR,
      GLX_CONTEXT_PROFILE_MASK_ARB, GPUCA_DISPLAY_OPENGL_CORE_FLAGS ? GLX_CONTEXT_CORE_PROFILE_BIT_ARB : GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
      None};
    glxContext = glXCreateContextAttribsARB(mDisplay, fbconfig, nullptr, GL_TRUE, context_attribs);
  } else {
    glxContext = glXCreateContext(mDisplay, visualInfo, nullptr, GL_TRUE);
  }
  if (glxContext == nullptr) {
    GPUError("could not create rendering context");
    return (-1);
  }

  Window win = RootWindow(mDisplay, visualInfo->screen);
  colorMap = XCreateColormap(mDisplay, win, visualInfo->visual, AllocNone);
  windowAttributes.colormap = colorMap;
  windowAttributes.border_pixel = 0;
  windowAttributes.event_mask = ExposureMask | VisibilityChangeMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | SubstructureNotifyMask | FocusChangeMask;

  // Create an X window with the selected visual
  mWindow = XCreateWindow(mDisplay, win, 50, 50, INIT_WIDTH, INIT_HEIGHT, // Position / Width and height of window
                          0, visualInfo->depth, InputOutput, visualInfo->visual, CWBorderPixel | CWColormap | CWEventMask, &windowAttributes);
  XSetStandardProperties(mDisplay, mWindow, GL_WINDOW_NAME, GL_WINDOW_NAME, None, nullptr, 0, nullptr);
  glXMakeCurrent(mDisplay, mWindow, glxContext);
  XMapWindow(mDisplay, mWindow);

  // Maximize window
  ToggleMaximized(true);

  // Receive signal when window closed
  Atom WM_DELETE_WINDOW = XInternAtom(mDisplay, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(mDisplay, mWindow, &WM_DELETE_WINDOW, 1);
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  // Prepare fonts
  mFontBase = glGenLists(256);
  if (!glIsList(mFontBase)) {
    GPUError("Out of display lists.");
    return (-1);
  }
  const char* f = "fixed";
  XFontStruct* font_info = XLoadQueryFont(mDisplay, f);
  if (!font_info) {
    GPUError("XLoadQueryFont failed.");
    return (-1);
  } else {
    int first = font_info->min_char_or_byte2;
    int last = font_info->max_char_or_byte2;
    glXUseXFont(font_info->fid, first, last - first + 1, mFontBase + first);
  }
#endif

  // Init OpenGL...
  if (GPUDisplayExtInit()) {
    fprintf(stderr, "Error initializing GL extension wrapper\n");
    return (-1);
  }

  XMapWindow(mDisplay, mWindow);
  XFlush(mDisplay);
  int x11_fd = ConnectionNumber(mDisplay);

  // Enable vsync
  if (vsync_supported) {
    mGlXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT");
    if (mGlXSwapIntervalEXT == nullptr) {
      GPUError("Cannot enable vsync");
      return (-1);
    }
    mGlXSwapIntervalEXT(mDisplay, glXGetCurrentDrawable(), 0);
  }

  if (InitGL()) {
    return (1);
  }

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

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
      num_ready_fds = mMaxFPSRate || XPending(mDisplay) || select(x11_fd + 1, &in_fds, nullptr, nullptr, &tv);
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
          break;
        }

        case KeyRelease: {
          int handleKey = 0, keyPress = 0;
          GetKey(event, handleKey, keyPress);
          if (mKeys[keyPress]) {
            HandleKeyRelease(handleKey);
          }
          mKeys[keyPress] = false;
          mKeysShift[keyPress] = false;
          break;
        }

        case MotionNotify: {
          mouseMvX = event.xmotion.x;
          mouseMvY = event.xmotion.y;
          break;
        }

        case Expose: {
          break;
        }

        case ConfigureNotify: {
          ReSizeGLScene(event.xconfigure.width, event.xconfigure.height);
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
    glXSwapBuffers(mDisplay, mWindow); // Buffer swap does implicit glFlush
  }

#ifndef GPUCA_DISPLAY_OPENGL_CORE
  glDeleteLists(mFontBase, 256);
  XUnloadFont(mDisplay, font_info->fid);
#endif
  ExitGL();
  glXDestroyContext(mDisplay, glxContext);
  XFree(visualInfo);
  XDestroyWindow(mDisplay, mWindow);
  XCloseDisplay(mDisplay);

  pthread_mutex_lock(&mSemLockExit);
  mDisplayRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  return (0);
}

void GPUDisplayBackendX11::DisplayExit()
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

void GPUDisplayBackendX11::SwitchFullscreen(bool set)
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

void GPUDisplayBackendX11::ToggleMaximized(bool set)
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

void GPUDisplayBackendX11::SetVSync(bool enable)
{
  if (vsync_supported) {
    mGlXSwapIntervalEXT(mDisplay, glXGetCurrentDrawable(), (int)enable);
  }
}

int GPUDisplayBackendX11::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, OpenGLWrapper, this)) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
  return (0);
}
