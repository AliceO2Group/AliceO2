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
#include <unistd.h>

#include <pthread.h>
using namespace GPUCA_NAMESPACE::gpu;
static GPUDisplayFrontendGlut* me = nullptr;

GPUDisplayFrontendGlut::GPUDisplayFrontendGlut()
{
  mFrontendType = TYPE_GLUT;
  mFrontendName = "GLUT";
}

void GPUDisplayFrontendGlut::displayFunc()
{
  me->DrawGLScene();
  glutSwapBuffers();
}

void GPUDisplayFrontendGlut::glutLoopFunc()
{
  me->HandleSendKey();
  displayFunc();
}

int32_t GPUDisplayFrontendGlut::GetKey(int32_t key)
{
  if (key == GLUT_KEY_UP) {
    return KEY_UP;
  }
  if (key == GLUT_KEY_DOWN) {
    return KEY_DOWN;
  }
  if (key == GLUT_KEY_LEFT) {
    return KEY_LEFT;
  }
  if (key == GLUT_KEY_RIGHT) {
    return KEY_RIGHT;
  }
  if (key == GLUT_KEY_PAGE_UP) {
    return KEY_PAGEUP;
  }
  if (key == GLUT_KEY_PAGE_DOWN) {
    return KEY_PAGEDOWN;
  }
  if (key == GLUT_KEY_HOME) {
    return KEY_HOME;
  }
  if (key == GLUT_KEY_END) {
    return KEY_END;
  }
  if (key == GLUT_KEY_INSERT) {
    return KEY_INSERT;
  }
  if (key == GLUT_KEY_F1) {
    return KEY_F1;
  }
  if (key == GLUT_KEY_F2) {
    return KEY_F2;
  }
  if (key == GLUT_KEY_F3) {
    return KEY_F3;
  }
  if (key == GLUT_KEY_F4) {
    return KEY_F4;
  }
  if (key == GLUT_KEY_F5) {
    return KEY_F5;
  }
  if (key == GLUT_KEY_F6) {
    return KEY_F6;
  }
  if (key == GLUT_KEY_F7) {
    return KEY_F7;
  }
  if (key == GLUT_KEY_F8) {
    return KEY_F8;
  }
  if (key == GLUT_KEY_F9) {
    return KEY_F9;
  }
  if (key == GLUT_KEY_F10) {
    return KEY_F10;
  }
  if (key == GLUT_KEY_F11) {
    return KEY_F11;
  }
  if (key == GLUT_KEY_F12) {
    return KEY_F12;
  }
  if (key == 112 || key == 113) {
    return KEY_SHIFT;
  }
  if (key == 114) {
    return KEY_CTRL;
  }
  if (key == 115) {
    return KEY_RCTRL;
  }
  if (key == 116) {
    return KEY_ALT;
  }

  return (0);
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

void GPUDisplayFrontendGlut::keyboardDownFunc(uint8_t key, int32_t x, int32_t y)
{
  int32_t handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, false);
  me->mKeysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
  me->mKeys[keyPress] = true;
  me->HandleKey(handleKey);
}

void GPUDisplayFrontendGlut::keyboardUpFunc(uint8_t key, int32_t x, int32_t y)
{
  int32_t handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, false);
  me->mKeys[keyPress] = false;
  me->mKeysShift[keyPress] = false;
}

void GPUDisplayFrontendGlut::specialDownFunc(int32_t key, int32_t x, int32_t y)
{
  int32_t handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, true);
  me->mKeysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
  me->mKeys[keyPress] = true;
  me->HandleKey(handleKey);
}

void GPUDisplayFrontendGlut::specialUpFunc(int32_t key, int32_t x, int32_t y)
{
  int32_t handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, true);
  me->mKeys[keyPress] = false;
  me->mKeysShift[keyPress] = false;
}

void GPUDisplayFrontendGlut::ResizeSceneWrapper(int32_t width, int32_t height)
{
  if (!me->mFullScreen) {
    me->mWidth = width;
    me->mHeight = height;
  }
  me->ResizeScene(width, height);
}

void GPUDisplayFrontendGlut::mouseFunc(int32_t button, int32_t state, int32_t x, int32_t y)
{
  if (button == 3) {
    me->mMouseWheel += 100;
  } else if (button == 4) {
    me->mMouseWheel -= 100;
  } else if (state == GLUT_DOWN) {
    if (button == GLUT_LEFT_BUTTON) {
      me->mMouseDn = true;
    } else if (button == GLUT_RIGHT_BUTTON) {
      me->mMouseDnR = true;
    }
    me->mMouseDnX = x;
    me->mMouseDnY = y;
  } else if (state == GLUT_UP) {
    if (button == GLUT_LEFT_BUTTON) {
      me->mMouseDn = false;
    } else if (button == GLUT_RIGHT_BUTTON) {
      me->mMouseDnR = false;
    }
  }
}

void GPUDisplayFrontendGlut::mouseMoveFunc(int32_t x, int32_t y)
{
  me->mMouseMvX = x;
  me->mMouseMvY = y;
}

void GPUDisplayFrontendGlut::mMouseWheelFunc(int32_t button, int32_t dir, int32_t x, int32_t y) { me->mMouseWheel += dir; }

int32_t GPUDisplayFrontendGlut::FrontendMain()
{
  if (backend()->backendType() != GPUDisplayBackend::TYPE_OPENGL) {
    fprintf(stderr, "Only OpenGL backend supported\n");
    return 1;
  }
  me = this;
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

  glutDisplayFunc(displayFunc);
  glutIdleFunc(glutLoopFunc);
  glutReshapeFunc(ResizeSceneWrapper);
  glutKeyboardFunc(keyboardDownFunc);
  glutKeyboardUpFunc(keyboardUpFunc);
  glutSpecialFunc(specialDownFunc);
  glutSpecialUpFunc(specialUpFunc);
  glutMouseFunc(mouseFunc);
  glutMotionFunc(mouseMoveFunc);
  glutPassiveMotionFunc(mouseMoveFunc);
  glutMouseWheelFunc(mMouseWheelFunc);
  ToggleMaximized(true);

  pthread_mutex_lock(&mSemLockExit);
  mGlutRunning = true;
  pthread_mutex_unlock(&mSemLockExit);
  glutMainLoop();
  mDisplayControl = 2;
  pthread_mutex_lock(&mSemLockExit);
  mGlutRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  ExitDisplay();
  return 0;
}

void GPUDisplayFrontendGlut::DisplayExit()
{
  pthread_mutex_lock(&mSemLockExit);
  if (mGlutRunning) {
    glutLeaveMainLoop();
  }
  pthread_mutex_unlock(&mSemLockExit);
  while (mGlutRunning) {
    usleep(10000);
  }
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

void GPUDisplayFrontendGlut::ToggleMaximized(bool set) {}
void GPUDisplayFrontendGlut::SetVSync(bool enable) {}

int32_t GPUDisplayFrontendGlut::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, FrontendThreadWrapper, this)) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
  return (0);
}
