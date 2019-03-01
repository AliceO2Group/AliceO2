#ifndef GPUDISPLAYBACKENDX11_H
#define GPUDISPLAYBACKENDX11_H

#include "GPUDisplayBackend.h"
#include <GL/glx.h>
#include <pthread.h>
#include <unistd.h>
#include <GL/glxext.h>

class GPUDisplayBackendX11 : public GPUDisplayBackend
{
public:
	GPUDisplayBackendX11() = default;
	virtual ~GPUDisplayBackendX11() = default;
	
	virtual int StartDisplay() override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen(bool set) override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

private:
	virtual int OpenGLMain();
	int GetKey(int key);
	void GetKey(XEvent& event, int& keyOut, int& keyPressOut);
	
	pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
	volatile bool displayRunning = false;

	GLuint font_base;

	Display *g_pDisplay = NULL;
	Window g_window;

	PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = NULL;
};

#endif
