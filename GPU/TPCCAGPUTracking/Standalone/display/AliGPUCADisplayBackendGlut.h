#ifndef ALIGPUCADISPLAYBACKENDGLUT_H
#define ALIGPUCADISPLAYBACKENDGLUT_H

#include "AliGPUCADisplayBackend.h"
#include <pthread.h>

class AliGPUCADisplayBackendGlut : public AliGPUCADisplayBackend
{
public:
	AliGPUCADisplayBackendGlut() = default;
	
	virtual void StartDisplay() override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen() override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s) override;
	
private:
	virtual void* OpenGLMain() override;
	
	static void displayFunc(void);
	static void glutLoopFunc(void);
	static void keyboardUpFunc(unsigned char key, int x, int y);
	static void keyboardDownFunc(unsigned char key, int x, int y);
	static void mouseMoveFunc(int x, int y);
	static void mouseWheelFunc(int button, int dir, int x, int y);
	static void mouseFunc(int button, int state, int x, int y);
	static void ReSizeGLSceneWrapper(int width, int height);
	
	volatile bool glutRunning = false;
	pthread_mutex_t semLockExit = PTHREAD_MUTEX_INITIALIZER;
};

#endif
