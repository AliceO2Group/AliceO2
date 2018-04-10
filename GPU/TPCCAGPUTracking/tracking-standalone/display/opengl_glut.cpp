#include "opengl_backend.h"

#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <unistd.h>

#include <pthread.h>
pthread_mutex_t semLockDisplay = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;

static volatile bool glutRunning = false;

static void displayFunc(void)
{
	DrawGLScene();
	glutSwapBuffers();
}

static void glutLoopFunc(void)
{
	HandleSendKey();
	displayFunc();
}

int GetKey(int key)
{
	if (key == 45) return('-');
	if (key == 43) return('+');
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	return(key);
}

static void keyboardDownFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	keysShift[key] = glutGetModifiers() & GLUT_ACTIVE_SHIFT;
	keys[key] = true;
}

static void keyboardUpFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	HandleKeyRelease(key);
	keys[key] = false;
	keysShift[key] = false;
}

static void mouseFunc(int button, int state, int x, int y)
{
	if (button == 3)
	{
		mouseWheel += 100;
	}
	else if (button == 4)
	{
		mouseWheel -= 100;
	}
	else if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			mouseDn = true;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			mouseDnR = true;
		}
		mouseDnX = x;
		mouseDnY = y;
	}
	else if (state == GLUT_UP)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			mouseDn = false;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			mouseDnR = false;
		}
	}
}

static void mouseMoveFunc(int x, int y)
{
	mouseMvX = x;
	mouseMvY = y;
}

static void mouseWheelFunc(int button, int dir, int x, int y)
{
	mouseWheel += dir;
}

void *OpenGLMain(void *ptr)
{
	int nopts = 2;
	char* opts[] = {"progname", "-direct"};
	glutInit(&nopts, opts);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(init_width, init_height);
	glutCreateWindow(GL_WINDOW_NAME);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	if (!InitGL()) return((void*) -1);
	glewInit();

	glutDisplayFunc(displayFunc);
	glutIdleFunc(glutLoopFunc);
	glutReshapeFunc(ReSizeGLScene);
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

void DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (glutRunning) glutLeaveMainLoop();
	pthread_mutex_unlock(&semLockExit);
	while (glutRunning) usleep(10000);
}

void OpenGLPrint(const char* s) {}
