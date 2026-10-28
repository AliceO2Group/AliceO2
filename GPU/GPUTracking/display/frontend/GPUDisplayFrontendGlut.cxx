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

/// \file GPUDisplayFrontendGlut.cxx
/// \author David Rohr

// Now the other headers
#include "GPUDisplayFrontendGlut.h"
#include "backend/GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstring>
#include <GL/freeglut.h>
#include <GL/freeglut_ucall.h>
#include <unistd.h>
#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xatom.h>
#endif

#include <pthread.h>
using namespace o2::gpu;

GPUDisplayFrontendGlut::GPUDisplayFrontendGlut()
{
  mFrontendType = TYPE_GLUT;
  mFrontendName = "GLUT";
}

void GPUDisplayFrontendGlut::displayFunc()
{
  mRedisplayPending = false;
  HandleUpdatesAndDrawGLScene();
  glutSwapBuffers();
  waitForFPSLimit();
}

void GPUDisplayFrontendGlut::glutLoopFunc()
{
  if (mRedisplayPending) { // Last redisplay not served (window not visible): handle updates without drawing
    HandleUpdatesAndDrawGLScene(true);
    waitForFPSLimit();
  }
  mRedisplayPending = true;
  glutPostRedisplay(); // Drawing only in displayFunc, so that expose events do not draw twice
}

int32_t GPUDisplayFrontendGlut::GetKey(int32_t key)
{
  switch (key) {
    case GLUT_KEY_UP:
      return KEY_UP;
    case GLUT_KEY_DOWN:
      return KEY_DOWN;
    case GLUT_KEY_LEFT:
      return KEY_LEFT;
    case GLUT_KEY_RIGHT:
      return KEY_RIGHT;
    case GLUT_KEY_PAGE_UP:
      return KEY_PAGEUP;
    case GLUT_KEY_PAGE_DOWN:
      return KEY_PAGEDOWN;
    case GLUT_KEY_HOME:
      return KEY_HOME;
    case GLUT_KEY_END:
      return KEY_END;
    case GLUT_KEY_INSERT:
      return KEY_INSERT;
    case GLUT_KEY_F1:
      return KEY_F1;
    case GLUT_KEY_F2:
      return KEY_F2;
    case GLUT_KEY_F3:
      return KEY_F3;
    case GLUT_KEY_F4:
      return KEY_F4;
    case GLUT_KEY_F5:
      return KEY_F5;
    case GLUT_KEY_F6:
      return KEY_F6;
    case GLUT_KEY_F7:
      return KEY_F7;
    case GLUT_KEY_F8:
      return KEY_F8;
    case GLUT_KEY_F9:
      return KEY_F9;
    case GLUT_KEY_F10:
      return KEY_F10;
    case GLUT_KEY_F11:
      return KEY_F11;
    case GLUT_KEY_F12:
      return KEY_F12;
    case GLUT_KEY_SHIFT_L:
    case GLUT_KEY_SHIFT_R:
      return KEY_SHIFT;
    case GLUT_KEY_CTRL_L:
      return KEY_CTRL;
    case GLUT_KEY_CTRL_R:
      return KEY_RCTRL;
    case GLUT_KEY_ALT_L:
      return KEY_ALT;
    case GLUT_KEY_ALT_R:
      return KEY_RALT;
    default:
      return 0;
  }
}

void GPUDisplayFrontendGlut::GetKey(int32_t key, int32_t& keyOut, int32_t& keyPressOut, bool special)
{
  int32_t specialKey = special ? GetKey(key) : 0;
  // GPUInfo("Key: key %d (%c) (special %d) -> %d (%c) special %d (%c)", key, (char) key, (int32_t) special, (int32_t) key, key, specialKey, (char) specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = key;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayFrontendGlut::keyboardFunc(int32_t key, bool special, bool down)
{
  int32_t handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, special);
  mKeys[keyPress] = down;
  if (down) {
    HandleKey(handleKey);
  }
}

void GPUDisplayFrontendGlut::ResizeSceneWrapper(int32_t width, int32_t height)
{
  if (!mFullScreen) {
    mWidth = width;
    mHeight = height;
  }
  ResizeScene(width, height);
#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
  Display* display = glXGetCurrentDisplay(); // Maximizing / unmaximizing by the window manager changes the size, read the state as freeglut does not report it (fullscreen stays as requested, freeglut tracks it itself)
  Atom type;
  int32_t format;
  unsigned long nItems, bytesAfter;
  unsigned char* data = nullptr;
  if (!display || XGetWindowProperty(display, glXGetCurrentDrawable(), XInternAtom(display, "_NET_WM_STATE", False), 0, 1024, False, XA_ATOM, &type, &format, &nItems, &bytesAfter, &data) != Success || data == nullptr) {
    return;
  }
  const Atom maxHorz = XInternAtom(display, "_NET_WM_STATE_MAXIMIZED_HORZ", False), maxVert = XInternAtom(display, "_NET_WM_STATE_MAXIMIZED_VERT", False);
  bool isMaxHorz = false, isMaxVert = false;
  for (unsigned long i = 0; i < nItems; i++) {
    isMaxHorz |= ((Atom*)data)[i] == maxHorz;
    isMaxVert |= ((Atom*)data)[i] == maxVert;
  }
  XFree(data);
  setWindowState(mFullScreen, isMaxHorz && isMaxVert);
#endif
}

void GPUDisplayFrontendGlut::mouseFunc(int32_t button, int32_t state, int32_t x, int32_t y)
{
  if (button == 3) {
    mMouseWheel += 100;
  } else if (button == 4) {
    mMouseWheel -= 100;
  } else if (state == GLUT_DOWN) {
    if (button == GLUT_LEFT_BUTTON) {
      mMouseDn = true;
    } else if (button == GLUT_RIGHT_BUTTON) {
      mMouseDnR = true;
    } else if (button == GLUT_MIDDLE_BUTTON) {
      mMouseDnM = true;
    }
    mMouseDnX = x;
    mMouseDnY = y;
  } else if (state == GLUT_UP) {
    if (button == GLUT_LEFT_BUTTON) {
      mMouseDn = false;
    } else if (button == GLUT_RIGHT_BUTTON) {
      mMouseDnR = false;
    } else if (button == GLUT_MIDDLE_BUTTON) {
      mMouseDnM = false;
    }
  }
}

void GPUDisplayFrontendGlut::mouseMoveFunc(int32_t x, int32_t y)
{
  mMouseMvX = x;
  mMouseMvY = y;
}

void GPUDisplayFrontendGlut::mMouseWheelFunc(int32_t button, int32_t dir, int32_t x, int32_t y) { mMouseWheel += dir; }

int32_t GPUDisplayFrontendGlut::FrontendMain()
{
  if (backend()->backendType() != GPUDisplayBackend::TYPE_OPENGL) {
    fprintf(stderr, "Only OpenGL backend supported\n");
    return 1;
  }
  mCanDrawText = 1;
  if (drawTextFontSize() == 0) {
    drawTextFontSize() = 12;
  }

  int32_t nopts = 2;
  char opt1[] = "progname";
  char opt2[] = "-direct";
  char* opts[] = {opt1, opt2};
  glutInit(&nopts, opts);
  glutInitContextVersion(GL_MIN_VERSION_MAJOR, GL_MIN_VERSION_MINOR);
  glutInitContextProfile(mBackend->CoreProfile() ? GLUT_CORE_PROFILE : GLUT_COMPATIBILITY_PROFILE);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
  glutCreateWindow(DISPLAY_WINDOW_NAME);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  if (mBackend->ExtInit()) {
    fprintf(stderr, "Error initializing GL extension wrapper\n");
    return (-1);
  }
  if (InitDisplay()) {
    fprintf(stderr, "Error in OpenGL initialization\n");
    return (1);
  }

  auto mouseMoveFunc = [](int32_t x, int32_t y, void* p) { self(p)->mouseMoveFunc(x, y); };
  glutDisplayFuncUcall([](void* p) { self(p)->displayFunc(); }, this); // Callbacks with this as user data, freeglut >= 3.2
  glutIdleFuncUcall([](void* p) { self(p)->glutLoopFunc(); }, this);
  glutReshapeFuncUcall([](int32_t width, int32_t height, void* p) { self(p)->ResizeSceneWrapper(width, height); }, this);
  glutKeyboardFuncUcall([](uint8_t key, int32_t x, int32_t y, void* p) { self(p)->keyboardFunc(key, false, true); }, this);
  glutKeyboardUpFuncUcall([](uint8_t key, int32_t x, int32_t y, void* p) { self(p)->keyboardFunc(key, false, false); }, this);
  glutSpecialFuncUcall([](int32_t key, int32_t x, int32_t y, void* p) { self(p)->keyboardFunc(key, true, true); }, this);
  glutSpecialUpFuncUcall([](int32_t key, int32_t x, int32_t y, void* p) { self(p)->keyboardFunc(key, true, false); }, this);
  glutMouseFuncUcall([](int32_t button, int32_t state, int32_t x, int32_t y, void* p) { self(p)->mouseFunc(button, state, x, y); }, this);
  glutMotionFuncUcall(mouseMoveFunc, this);
  glutPassiveMotionFuncUcall(mouseMoveFunc, this);
  glutMouseWheelFuncUcall([](int32_t button, int32_t dir, int32_t x, int32_t y, void* p) { self(p)->mMouseWheelFunc(button, dir, x, y); }, this);
  applyInitialWindowState();
  SetVSync(vsyncEnabled());

  {
    std::lock_guard lock(mExitStateMutex);
    mDisplayRunning = true;
  }
  glutMainLoop();
  {
    std::lock_guard lock(mExitStateMutex);
    setDisplayControl(2);
    mDisplayRunning = false;
  }

  ExitDisplay();
  return 0;
}

void GPUDisplayFrontendGlut::StopDisplayA()
{
  glutLeaveMainLoop();
}

void GPUDisplayFrontendGlut::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!fromBotton) {
    y = mDisplayHeight - y;
  }
  glColor4f(r, g, b, a);
  glRasterPos2f(x, y);
  glutBitmapString(GLUT_BITMAP_HELVETICA_12, (const uint8_t*)s);
#endif
}

void GPUDisplayFrontendGlut::SwitchFullscreen(bool set)
{
  mFullScreen = set;
  if (set) {
    glutFullScreen();
  } else {
    glutReshapeWindow(mWidth, mHeight);
  }
}

void GPUDisplayFrontendGlut::ToggleMaximized(bool set) // GLUT has no API for maximizing, request it from the window manager via GLX / X11 directly (no-op if freeglut does not use GLX)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
  Display* display = glXGetCurrentDisplay();
  if (!display) {
    return;
  }
  XEvent xev;
  memset(&xev, 0, sizeof(xev));
  xev.type = ClientMessage;
  xev.xclient.window = glXGetCurrentDrawable(); // freeglut renders directly to the X window
  xev.xclient.message_type = XInternAtom(display, "_NET_WM_STATE", False);
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = set ? 1 : 0; // _NET_WM_STATE_ADD / _NET_WM_STATE_REMOVE
  xev.xclient.data.l[1] = XInternAtom(display, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
  xev.xclient.data.l[2] = XInternAtom(display, "_NET_WM_STATE_MAXIMIZED_VERT", False);
  XSendEvent(display, DefaultRootWindow(display), False, SubstructureNotifyMask | SubstructureRedirectMask, &xev);
  XFlush(display);
#endif
}

void GPUDisplayFrontendGlut::SetVSync(bool enable) // GLUT has no swap interval API, set it via GLX directly (no-op if freeglut does not use GLX)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
  if (canSetVSync()) {
    ((PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT"))(glXGetCurrentDisplay(), glXGetCurrentDrawable(), enable);
  }
#endif
}

bool GPUDisplayFrontendGlut::canSetVSync()
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
  return glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT") && glXGetCurrentDisplay();
#else
  return false;
#endif
}
