#include "AliGPUCADisplayBackendGlut.h"

#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <unistd.h>

#include <pthread.h>
static AliGPUCADisplayBackendGlut* me = nullptr;

void AliGPUCADisplayBackendGlut::displayFunc(void)
{
	me->DrawGLScene();
	glutSwapBuffers();
}

void AliGPUCADisplayBackendGlut::glutLoopFunc(void)
{
	me->HandleSendKey();
	displayFunc();
}

int AliGPUCADisplayBackendGlut::GetKey(int key)
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

void AliGPUCADisplayBackendGlut::GetKey(int key, int& keyOut, int& keyPressOut, bool special)
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

void AliGPUCADisplayBackendGlut::keyboardDownFunc(unsigned char key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, false);
	me->keysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	me->keys[keyPress] = true;
}

void AliGPUCADisplayBackendGlut::keyboardUpFunc(unsigned char key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, false);
	me->HandleKeyRelease(handleKey);
	me->keys[keyPress] = false;
	me->keysShift[keyPress] = false;
}

void AliGPUCADisplayBackendGlut::specialDownFunc(int key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, true);
	me->keysShift[keyPress] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	me->keys[keyPress] = true;
}

void AliGPUCADisplayBackendGlut::specialUpFunc(int key, int x, int y)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, handleKey, keyPress, true);
	me->HandleKeyRelease(handleKey);
	me->keys[keyPress] = false;
	me->keysShift[keyPress] = false;
}

void AliGPUCADisplayBackendGlut::ReSizeGLSceneWrapper(int width, int height)
{
	if (!me->fullScreen)
	{
		me->width = width;
		me->height = height;
	}
	me->ReSizeGLScene(width, height);
}

void AliGPUCADisplayBackendGlut::mouseFunc(int button, int state, int x, int y)
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

void AliGPUCADisplayBackendGlut::mouseMoveFunc(int x, int y)
{
	me->mouseMvX = x;
	me->mouseMvY = y;
}

void AliGPUCADisplayBackendGlut::mouseWheelFunc(int button, int dir, int x, int y)
{
	me->mouseWheel += dir;
}

void* AliGPUCADisplayBackendGlut::OpenGLMain()
{
	me = this;
	int nopts = 2;
	char* opts[] = {"progname", "-direct"};
	glutInit(&nopts, opts);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(init_width, init_height);
	glutCreateWindow(GL_WINDOW_NAME);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	if (InitGL()) return((void*) -1);

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

void AliGPUCADisplayBackendGlut::DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (glutRunning) glutLeaveMainLoop();
	pthread_mutex_unlock(&semLockExit);
	while (glutRunning) usleep(10000);
}

void AliGPUCADisplayBackendGlut::OpenGLPrint(const char* s)
{
	glutBitmapString( GLUT_BITMAP_HELVETICA_12, (const unsigned char*) s);
}

void AliGPUCADisplayBackendGlut::SwitchFullscreen(bool set)
{
	fullScreen = set;
	if (set) glutFullScreen();
	else glutReshapeWindow(width, height);
}

void AliGPUCADisplayBackendGlut::ToggleMaximized(bool set) {}
void AliGPUCADisplayBackendGlut::SetVSync(bool enable) {}

int AliGPUCADisplayBackendGlut::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\n");
		return(1);
	}
	return(0);
}
