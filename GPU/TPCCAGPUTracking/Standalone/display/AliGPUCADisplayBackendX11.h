#ifndef ALIGPUCADISPLAYBACKENDX11_H
#define ALIGPUCADISPLAYBACKENDX11_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AliGPUCADisplayBackend.h"
#include <GL/glx.h>
#include <pthread.h>
#include <unistd.h>
#include <GL/glxext.h>

class AliGPUCADisplayBackendX11 : public AliGPUCADisplayBackend
{
public:
	AliGPUCADisplayBackendX11() = default;
	
	virtual void StartDisplay() override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen() override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s) override;

private:
	virtual void* OpenGLMain();
	int GetKey(int key);
	
	pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
	volatile bool displayRunning = false;

	GLuint font_base;

	Display *g_pDisplay = NULL;
	Window g_window;

	PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = NULL;
};

#endif
