// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendGlut.cpp
/// \author David Rohr

//GLEW must be the first header
#include <GL/glew.h>

//Now the other headers
#include "GPUDisplayBackendGlut.h"
#include <cstdio>
#include <cstring>
#include <GL/freeglut.h>
#include <unistd.h>

#include <pthread.h>
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
	if (key == GLUT_KEY_UP) return(KEY_UP);
	if (key == GLUT_KEY_DOWN) return(KEY_DOWN);
	if (key == GLUT_KEY_LEFT) return(KEY_LEFT);
	if (key == GLUT_KEY_RIGHT) return(KEY_RIGHT);
	if (key == GLUT_KEY_PAGE_UP) return(KEY_PAGEUP);
	if (key == GLUT_KEY_PAGE_DOWN) return(KEY_PAGEDOWN);
	if (key == GLUT_KEY_HOME) return(KEY_HOME);
	if (key == GLUT_KEY_END) return(KEY_END);
	if (key == GLUT_KEY_INSERT) return(KEY_INSERT);
	if (key == GLUT_KEY_F1) return(KEY_F1);
	if (key == GLUT_KEY_F2) return(KEY_F2);
	if (key == GLUT_KEY_F3) return(KEY_F3);
	if (key == GLUT_KEY_F4) return(KEY_F4);
	if (key == GLUT_KEY_F5) return(KEY_F5);
	if (key == GLUT_KEY_F6) return(KEY_F6);
	if (key == GLUT_KEY_F7) return(KEY_F7);
	if (key == GLUT_KEY_F8) return(KEY_F8);
	if (key == GLUT_KEY_F9) return(KEY_F9);
	if (key == GLUT_KEY_F10) return(KEY_F10);
	if (key == GLUT_KEY_F11) return(KEY_F11);
	if (key == GLUT_KEY_F12) return(KEY_F12);
	return(0);
}

void GPUDisplayBackendGlut::GetKey(int key, int& keyOut, int& keyPressOut, bool special)
{
	int specialKey = special ? GetKey(key) : 0;
	//printf("Key: key %d (%c) (special %d) -> %d (%c) special %d (%c)\n", key, (char) key, (int) special, (int) key, key, specialKey, (char) specialKey);

	if (specialKey)
	{
		keyOut = keyPressOut = specialKey;
	}
	else
	{
		keyOut = keyPressOut = key;
		if (keyPressOut >= 'a' && keyPressOut <= 'z') keyPressOut += 'A' - 'a';
	}
}

void GPUDisplayBackendGlut::keyboardDownFunc(unsigned char key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, false);
	me->keysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	me->keys[keyPress] = true;
}

void GPUDisplayBackendGlut::keyboardUpFunc(unsigned char key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, false);
	if (me->keys[keyPress]) me->HandleKeyRelease(handleKey);
	me->keys[keyPress] = false;
	me->keysShift[keyPress] = false;
}

void GPUDisplayBackendGlut::specialDownFunc(int key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, true);
	me->keysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	me->keys[keyPress] = true;
}

void GPUDisplayBackendGlut::specialUpFunc(int key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, true);
	if (me->keys[keyPress]) me->HandleKeyRelease(handleKey);
	me->keys[keyPress] = false;
	me->keysShift[keyPress] = false;
}

void GPUDisplayBackendGlut::ReSizeGLSceneWrapper(int width, int height)
{
	if (!me->fullScreen)
	{
		me->width = width;
		me->height = height;
	}
	me->ReSizeGLScene(width, height);
}

void GPUDisplayBackendGlut::mouseFunc(int button, int state, int x, int y)
{
	if (button == 3)
	{
		me->mouseWheel += 100;
	}
	else if (button == 4)
	{
		me->mouseWheel -= 100;
	}
	else if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			me->mouseDn = true;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			me->mouseDnR = true;
		}
		me->mouseDnX = x;
		me->mouseDnY = y;
	}
	else if (state == GLUT_UP)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			me->mouseDn = false;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			me->mouseDnR = false;
		}
	}
}

void GPUDisplayBackendGlut::mouseMoveFunc(int x, int y)
{
	me->mouseMvX = x;
	me->mouseMvY = y;
}

void GPUDisplayBackendGlut::mouseWheelFunc(int button, int dir, int x, int y)
{
	me->mouseWheel += dir;
}

int GPUDisplayBackendGlut::OpenGLMain()
{
	me = this;
	int nopts = 2;
	char* opts[] = {"progname", "-direct"};
	glutInit(&nopts, opts);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(init_width, init_height);
	glutCreateWindow(GL_WINDOW_NAME);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	if (glewInit()) return(-1);
	if (InitGL()) return(1);

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
	glutMouseWheelFunc(mouseWheelFunc);
	ToggleMaximized(true);

	pthread_mutex_lock(&semLockExit);
	glutRunning = true;
	pthread_mutex_unlock(&semLockExit);
	glutMainLoop();
	displayControl = 2;
	pthread_mutex_lock(&semLockExit);
	glutRunning = false;
	pthread_mutex_unlock(&semLockExit);
	
	return 0;
}

void GPUDisplayBackendGlut::DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (glutRunning) glutLeaveMainLoop();
	pthread_mutex_unlock(&semLockExit);
	while (glutRunning) usleep(10000);
}

void GPUDisplayBackendGlut::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
	if (!fromBotton) y = display_height - y;
	glColor4f(r, g, b, a);
	glRasterPos2f(x, y);
	glutBitmapString( GLUT_BITMAP_HELVETICA_12, (const unsigned char*) s);
}

void GPUDisplayBackendGlut::SwitchFullscreen(bool set)
{
	fullScreen = set;
	if (set) glutFullScreen();
	else glutReshapeWindow(width, height);
}

void GPUDisplayBackendGlut::ToggleMaximized(bool set) {}
void GPUDisplayBackendGlut::SetVSync(bool enable) {}

int GPUDisplayBackendGlut::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\n");
		return(1);
	}
	return(0);
}
