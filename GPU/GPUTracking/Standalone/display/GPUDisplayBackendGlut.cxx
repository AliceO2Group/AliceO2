// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendGlut.cxx
/// \author David Rohr

// GL EXT must be the first header
#include "GPUDisplayExt.h"

// Now the other headers
#include "GPUDisplayBackendGlut.h"
#include "GPULogging.h"
#include <cstdio>
#include <cstring>
#include <GL/freeglut.h>
#include <unistd.h>

#include <pthread.h>
using namespace GPUCA_NAMESPACE::gpu;
static GPUDisplayBackendGlut* me = nullptr;

void GPUDisplayBackendGlut::displayFunc(void)
{
  me->DrawGLScene();
  glutSwapBuffers();
}

void GPUDisplayBackendGlut::glutLoopFunc(void)
{
  me->HandleSendKey();
  displayFunc();
}

int GPUDisplayBackendGlut::GetKey(int key)
{
  if (key == GLUT_KEY_UP) {
    return (KEY_UP);
  }
  if (key == GLUT_KEY_DOWN) {
    return (KEY_DOWN);
  }
  if (key == GLUT_KEY_LEFT) {
    return (KEY_LEFT);
  }
  if (key == GLUT_KEY_RIGHT) {
    return (KEY_RIGHT);
  }
  if (key == GLUT_KEY_PAGE_UP) {
    return (KEY_PAGEUP);
  }
  if (key == GLUT_KEY_PAGE_DOWN) {
    return (KEY_PAGEDOWN);
  }
  if (key == GLUT_KEY_HOME) {
    return (KEY_HOME);
  }
  if (key == GLUT_KEY_END) {
    return (KEY_END);
  }
  if (key == GLUT_KEY_INSERT) {
    return (KEY_INSERT);
  }
  if (key == GLUT_KEY_F1) {
    return (KEY_F1);
  }
  if (key == GLUT_KEY_F2) {
    return (KEY_F2);
  }
  if (key == GLUT_KEY_F3) {
    return (KEY_F3);
  }
  if (key == GLUT_KEY_F4) {
    return (KEY_F4);
  }
  if (key == GLUT_KEY_F5) {
    return (KEY_F5);
  }
  if (key == GLUT_KEY_F6) {
    return (KEY_F6);
  }
  if (key == GLUT_KEY_F7) {
    return (KEY_F7);
  }
  if (key == GLUT_KEY_F8) {
    return (KEY_F8);
  }
  if (key == GLUT_KEY_F9) {
    return (KEY_F9);
  }
  if (key == GLUT_KEY_F10) {
    return (KEY_F10);
  }
  if (key == GLUT_KEY_F11) {
    return (KEY_F11);
  }
  if (key == GLUT_KEY_F12) {
    return (KEY_F12);
  }
  return (0);
}

void GPUDisplayBackendGlut::GetKey(int key, int& keyOut, int& keyPressOut, bool special)
{
  int specialKey = special ? GetKey(key) : 0;
  // GPUInfo("Key: key %d (%c) (special %d) -> %d (%c) special %d (%c)", key, (char) key, (int) special, (int) key, key, specialKey, (char) specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = key;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayBackendGlut::keyboardDownFunc(unsigned char key, int x, int y)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, false);
  me->mKeysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
  me->mKeys[keyPress] = true;
}

void GPUDisplayBackendGlut::keyboardUpFunc(unsigned char key, int x, int y)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, false);
  if (me->mKeys[keyPress]) {
    me->HandleKeyRelease(handleKey);
  }
  me->mKeys[keyPress] = false;
  me->mKeysShift[keyPress] = false;
}

void GPUDisplayBackendGlut::specialDownFunc(int key, int x, int y)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, true);
  me->mKeysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
  me->mKeys[keyPress] = true;
}

void GPUDisplayBackendGlut::specialUpFunc(int key, int x, int y)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, handleKey, keyPress, true);
  if (me->mKeys[keyPress]) {
    me->HandleKeyRelease(handleKey);
  }
  me->mKeys[keyPress] = false;
  me->mKeysShift[keyPress] = false;
}

void GPUDisplayBackendGlut::ReSizeGLSceneWrapper(int width, int height)
{
  if (!me->mFullScreen) {
    me->mWidth = width;
    me->mHeight = height;
  }
  me->ReSizeGLScene(width, height);
}

void GPUDisplayBackendGlut::mouseFunc(int button, int state, int x, int y)
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

void GPUDisplayBackendGlut::mouseMoveFunc(int x, int y)
{
  me->mouseMvX = x;
  me->mouseMvY = y;
}

void GPUDisplayBackendGlut::mMouseWheelFunc(int button, int dir, int x, int y) { me->mMouseWheel += dir; }

int GPUDisplayBackendGlut::OpenGLMain()
{
  me = this;
  int nopts = 2;
  char opt1[] = "progname";
  char opt2[] = "-direct";
  char* opts[] = {opt1, opt2};
  glutInit(&nopts, opts);
  glutInitContextVersion(GL_MIN_VERSION_MAJOR, GL_MIN_VERSION_MINOR);
  glutInitContextProfile(GPUCA_DISPLAY_OPENGL_CORE_FLAGS ? GLUT_CORE_PROFILE : GLUT_COMPATIBILITY_PROFILE);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
  glutCreateWindow(GL_WINDOW_NAME);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  if (GPUDisplayExtInit()) {
    fprintf(stderr, "Error initializing GL extension wrapper\n");
    return (-1);
  }
  if (InitGL()) {
    fprintf(stderr, "Error in OpenGL initialization\n");
    return (1);
  }

  glutDisplayFunc(displayFunc);
  glutIdleFunc(glutLoopFunc);
  glutReshapeFunc(ReSizeGLSceneWrapper);
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

  return 0;
}

void GPUDisplayBackendGlut::DisplayExit()
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

void GPUDisplayBackendGlut::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifndef GPUCA_DISPLAY_OPENGL_CORE
  if (!fromBotton) {
    y = mDisplayHeight - y;
  }
  glColor4f(r, g, b, a);
  glRasterPos2f(x, y);
  glutBitmapString(GLUT_BITMAP_HELVETICA_12, (const unsigned char*)s);
#endif
}

void GPUDisplayBackendGlut::SwitchFullscreen(bool set)
{
  mFullScreen = set;
  if (set) {
    glutFullScreen();
  } else {
    glutReshapeWindow(mWidth, mHeight);
  }
}

void GPUDisplayBackendGlut::ToggleMaximized(bool set) {}
void GPUDisplayBackendGlut::SetVSync(bool enable) {}

int GPUDisplayBackendGlut::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, OpenGLWrapper, this)) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
  return (0);
}
