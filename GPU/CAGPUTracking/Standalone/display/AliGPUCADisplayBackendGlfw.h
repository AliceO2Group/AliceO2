#ifndef ALIGPUCADISPLAYBACKENDGlfw_H
#define ALIGPUCADISPLAYBACKENDGlfw_H

#include "AliGPUCADisplayBackend.h"
#include <pthread.h>

struct GLFWwindow;

class AliGPUCADisplayBackendGlfw : public AliGPUCADisplayBackend
{
public:
	AliGPUCADisplayBackendGlfw() = default;
	virtual ~AliGPUCADisplayBackendGlfw() = default;
	
	virtual int StartDisplay() override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen(bool set) override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s) override;
	
private:
	virtual void* OpenGLMain() override;
	
	static void GlfwLoopFunc(void);
	static void error_callback(int error, const char* description);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseButton_callback(GLFWwindow* window, int button, int action, int mods);
	static void scroll_callback(GLFWwindow* window, double x, double y);
	static void cursorPos_callback(GLFWwindow* window, double x, double y);
	static void resize_callback(GLFWwindow* window, int width, int height);
	static int GetKey(int key);
	static void GetKey(int keyin, int scancode, int mods, int& keyOut, int& keyPressOut);
	
	GLFWwindow* window;
	
	volatile bool GlfwRunning = false;
	pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
	int window_x = 0;
	int window_y = 0;
	int window_width = 0;
	int window_height = 0;
};

#endif
