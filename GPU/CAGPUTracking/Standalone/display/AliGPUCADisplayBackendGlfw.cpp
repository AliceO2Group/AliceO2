#include "AliGPUCADisplayBackendGlfw.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
static AliGPUCADisplayBackendGlfw* me = nullptr;

static int GetKey(int key)
{
	if (key == 45) return('-');
	if (key == 43) return('+');
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	return(key);
}

void AliGPUCADisplayBackendGlfw::keyboardDownFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	//me->keysShift[key] = GlfwGetModifiers() & Glfw_ACTIVE_SHIFT;
	me->keys[key] = true;
}

void AliGPUCADisplayBackendGlfw::keyboardUpFunc(unsigned char key, int x, int y)
{
	key = GetKey(key);
	me->HandleKeyRelease(key, 0);
	me->keys[key] = false;
	me->keysShift[key] = false;
}

void AliGPUCADisplayBackendGlfw::ReSizeGLSceneWrapper(int width, int height)
{
	me->ReSizeGLScene(width, height);
}

void AliGPUCADisplayBackendGlfw::mouseFunc(int button, int state, int x, int y)
{
	/*if (button == 3)
	{
		me->mouseWheel += 100;
	}
	else if (button == 4)
	{
		me->mouseWheel -= 100;
	}
	else if (state == Glfw_DOWN)
	{
		if (button == Glfw_LEFT_BUTTON)
		{
			me->mouseDn = true;
		}
		else if (button == Glfw_RIGHT_BUTTON)
		{
			me->mouseDnR = true;
		}
		me->mouseDnX = x;
		me->mouseDnY = y;
	}
	else if (state == Glfw_UP)
	{
		if (button == Glfw_LEFT_BUTTON)
		{
			me->mouseDn = false;
		}
		else if (button == Glfw_RIGHT_BUTTON)
		{
			me->mouseDnR = false;
		}
	}*/
}

void AliGPUCADisplayBackendGlfw::mouseMoveFunc(int x, int y)
{
	me->mouseMvX = x;
	me->mouseMvY = y;
}

void AliGPUCADisplayBackendGlfw::mouseWheelFunc(int button, int dir, int x, int y)
{
	me->mouseWheel += dir;
}

void* AliGPUCADisplayBackendGlfw::OpenGLMain()
{
	me = this;
	
	if (!glfwInit()) return((void*) -1);
	window = glfwCreateWindow(init_width, init_height, GL_WINDOW_NAME, NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return((void*) -1);
	}
	glfwMakeContextCurrent(window);

	pthread_mutex_lock(&semLockExit);
	GlfwRunning = true;
	pthread_mutex_unlock(&semLockExit);
	
	if (InitGL()) return((void*) -1);

	while (!glfwWindowShouldClose(window))
	{
		HandleSendKey();
		DrawGLScene();

		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	pthread_mutex_lock(&semLockExit);
	glfwTerminate();
	GlfwRunning = false;
	pthread_mutex_unlock(&semLockExit);
	
	return 0;
}

void AliGPUCADisplayBackendGlfw::DisplayExit()
{
	pthread_mutex_lock(&semLockExit);
	if (GlfwRunning) glfwSetWindowShouldClose(window, true);
	pthread_mutex_unlock(&semLockExit);
	while (GlfwRunning) usleep(10000);
}

void AliGPUCADisplayBackendGlfw::OpenGLPrint(const char* s) {}
void AliGPUCADisplayBackendGlfw::SwitchFullscreen() {}
void AliGPUCADisplayBackendGlfw::ToggleMaximized(bool set) {}
void AliGPUCADisplayBackendGlfw::SetVSync(bool enable) {}

void AliGPUCADisplayBackendGlfw::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\nExiting...\n");
	}
}
