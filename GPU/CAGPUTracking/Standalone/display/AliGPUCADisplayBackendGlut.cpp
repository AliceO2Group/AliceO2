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

static int GetKey(int key)
{
	if (key == 45) return('-');
	if (key == 43) return('+');
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	return(key);
}

void AliGPUCADisplayBackendGlut::keyboardDownFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	me->keysShift[key] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	me->keys[key] = true;
}

void AliGPUCADisplayBackendGlut::keyboardUpFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	me->HandleKeyRelease(key, 0);
	me->keys[key] = false;
	me->keysShift[key] = false;
}

void AliGPUCADisplayBackendGlut::ReSizeGLSceneWrapper(int width, int height)
{
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
	glutMouseFunc(mouseFunc);
	glutMotionFunc(mouseMoveFunc);
	glutPassiveMotionFunc(mouseMoveFunc);
	glutMouseWheelFunc(mouseWheelFunc);

	pthread_mutex_lock(&semLockExit);
	glutRunning = true;
	pthread_mutex_unlock(&semLockExit);
	glutMainLoop();
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

void AliGPUCADisplayBackendGlut::OpenGLPrint(const char* s) {}
void AliGPUCADisplayBackendGlut::SwitchFullscreen() {}
void AliGPUCADisplayBackendGlut::ToggleMaximized(bool set) {}
void AliGPUCADisplayBackendGlut::SetVSync(bool enable) {}

void AliGPUCADisplayBackendGlut::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\nExiting...\n");
	}
}
