#ifndef ALIGPUDISPLAYBACKENDWINDOWS_H
#define ALIGPUDISPLAYBACKENDWINDOWS_H

#include "AliGPUDisplayBackend.h"

class AliGPUDisplayBackendWindows : public AliGPUDisplayBackend
{
public:
	AliGPUDisplayBackendWindows() = default;
	virtual ~AliGPUDisplayBackendWindows() = default;
	
	virtual int StartDisplay() override;
	virtual void DisplayExit() override;
	virtual void SwitchFullscreen(bool set) override;
	virtual void ToggleMaximized(bool set) override;
	virtual void SetVSync(bool enable) override;
	virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

private:
	virtual int OpenGLMain();
};

#endif
