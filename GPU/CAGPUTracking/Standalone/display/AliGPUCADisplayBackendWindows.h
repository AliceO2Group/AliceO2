#ifndef ALIGPUCADISPLAYBACKENDWINDOWS_H
#define ALIGPUCADISPLAYBACKENDWINDOWS_H

#include "AliGPUCADisplayBackend.h"

class AliGPUCADisplayBackendWindows : public AliGPUCADisplayBackend
{
public:
	AliGPUCADisplayBackendWindows() = default;
	virtual ~AliGPUCADisplayBackendWindows() = default;
	
	virtual void StartDisplay() override;
	virtual void* OpenGLMain(void*) override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen() override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s) override;

private:
	virtual DWORD WINAPI OpenGLMain()
};

#endif
