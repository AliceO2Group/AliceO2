#include "AliGPUCADisplayBackendGlfw.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
static AliGPUCADisplayBackendGlfw* me = nullptr;

int AliGPUCADisplayBackendGlfw::GetKey(int key)
{
	if (key == GLFW_KEY_KP_SUBTRACT) return('-');
	if (key == GLFW_KEY_KP_ADD) return('+');
	if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) return(KEY_SHIFT);
	if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) return(KEY_ALT);
	if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) return(KEY_CTRL);
	if (key == GLFW_KEY_UP) return(KEY_UP);
	if (key == GLFW_KEY_DOWN) return(KEY_DOWN);
	if (key == GLFW_KEY_LEFT) return(KEY_LEFT);
	if (key == GLFW_KEY_RIGHT) return(KEY_RIGHT);
	if (key == GLFW_KEY_PAGE_UP) return(KEY_PAGEUP);
	if (key == GLFW_KEY_PAGE_DOWN) return(KEY_PAGEDOWN);
	if (key == GLFW_KEY_ESCAPE) return(KEY_ESCAPE);
	if (key == GLFW_KEY_SPACE) return(KEY_SPACE);
	if (key == GLFW_KEY_HOME) return(KEY_HOME);
	if (key == GLFW_KEY_END) return(KEY_END);
	if (key == GLFW_KEY_INSERT) return(KEY_INSERT);
	if (key == GLFW_KEY_ENTER) return(KEY_ENTER);
	if (key == GLFW_KEY_F1) return(KEY_F1);
	if (key == GLFW_KEY_F2) return(KEY_F2);
	if (key == GLFW_KEY_F3) return(KEY_F3);
	if (key == GLFW_KEY_F4) return(KEY_F4);
	if (key == GLFW_KEY_F5) return(KEY_F5);
	if (key == GLFW_KEY_F6) return(KEY_F6);
	if (key == GLFW_KEY_F7) return(KEY_F7);
	if (key == GLFW_KEY_F8) return(KEY_F8);
	if (key == GLFW_KEY_F9) return(KEY_F9);
	if (key == GLFW_KEY_F10) return(KEY_F10);
	if (key == GLFW_KEY_F11) return(KEY_F11);
	if (key == GLFW_KEY_F12) return(KEY_F12);
	return(0);
}

void AliGPUCADisplayBackendGlfw::GetKey(int key, int scancode, int mods, int& keyOut, int& keyPressOut)
{
	int specialKey = GetKey(key);
	const char* str = glfwGetKeyName(key, scancode);
	char localeKey = str ? str[0] : 0;
	if ((mods & GLFW_MOD_SHIFT) && localeKey >= 'a' && localeKey <= 'z') localeKey += 'A' - 'a';
	//printf("Key: key %d (%c) -> %d (%c) special %d (%c)\n", key, (char) key, (int) localeKey, localeKey, specialKey, (char) specialKey);

	if (specialKey)
	{
		keyOut = keyPressOut = specialKey;
	}
	else
	{
		keyOut = keyPressOut = localeKey;
		if (keyPressOut >= 'a' && keyPressOut <= 'z') keyPressOut += 'A' - 'a';
	}
}

void AliGPUCADisplayBackendGlfw::error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void AliGPUCADisplayBackendGlfw::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	int handleKey = 0, keyPress = 0;
	GetKey(key, scancode, mods, handleKey, keyPress);
	if (action == GLFW_PRESS)
	{
		me->keys[keyPress] = true;
		me->keysShift[keyPress] = mods & GLFW_MOD_SHIFT;
	}
	else if (action == GLFW_RELEASE)
	{
		printf("Handling %d %c\n", handleKey, (char) handleKey);
		me->HandleKeyRelease(handleKey);
		me->keys[keyPress] = false;
		me->keysShift[keyPress] = false;
	}
}

void AliGPUCADisplayBackendGlfw::mouseButton_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (button == 0)
		{
			me->mouseDn = true;
		}
		else if (button == 1)
		{
			me->mouseDnR = true;
		}
		me->mouseDnX = me->mouseMvX;
		me->mouseDnY = me->mouseMvY;
	}
	else if (action == GLFW_RELEASE)
	{
		if (button == 0)
		{
			me->mouseDn = false;
		}
		else if (button == 1)
		{
			me->mouseDnR = false;
		}
	}
}

void AliGPUCADisplayBackendGlfw::scroll_callback(GLFWwindow* window, double x, double y)
{
	me->mouseWheel += y * 100;
}

void AliGPUCADisplayBackendGlfw::cursorPos_callback(GLFWwindow* window, double x, double y)
{
	me->mouseMvX = x;
	me->mouseMvY = y;
}

void AliGPUCADisplayBackendGlfw::resize_callback(GLFWwindow* window, int width, int height)
{
	me->ReSizeGLScene(width, height);
}


void* AliGPUCADisplayBackendGlfw::OpenGLMain()
{
	me = this;
	
	if (!glfwInit()) return((void*) -1);
	glfwSetErrorCallback(error_callback);
	
	glfwWindowHint(GLFW_MAXIMIZED, 1);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(init_width, init_height, GL_WINDOW_NAME, NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return((void*) -1);
	}
	glfwMakeContextCurrent(window);
	
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouseButton_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetCursorPosCallback(window, cursorPos_callback);
	glfwSetWindowSizeCallback(window, resize_callback);

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

	displayControl = 2;
	pthread_mutex_lock(&semLockExit);
	glfwDestroyWindow(window);
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

void AliGPUCADisplayBackendGlfw::OpenGLPrint(const char* s)
{
	
}

void AliGPUCADisplayBackendGlfw::SwitchFullscreen(bool set)
{
	printf("Setting Full Screen %d\n", (int) set);
	if (set)
	{
		glfwGetWindowPos(window, &window_x, &window_y);
		glfwGetWindowSize(window, &window_width, &window_height);
		GLFWmonitor* primary = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(primary);
		glfwSetWindowMonitor(window, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
	}
	else
	{
		glfwSetWindowMonitor(window, NULL, window_x, window_y, window_width, window_height, GLFW_DONT_CARE);
	}
}

void AliGPUCADisplayBackendGlfw::ToggleMaximized(bool set)
{
	if (set) glfwMaximizeWindow(window);
	else glfwRestoreWindow(window);
}

void AliGPUCADisplayBackendGlfw::SetVSync(bool enable)
{
	glfwSwapInterval(enable);
}

void AliGPUCADisplayBackendGlfw::StartDisplay()
{
	static pthread_t hThread;
	if (pthread_create(&hThread, NULL, OpenGLWrapper, this))
	{
		printf("Coult not Create GL Thread...\nExiting...\n");
	}
}
