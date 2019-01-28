#ifndef ALIGPUCADISPLAYBACKENDWINDOWS_H
#define ALIGPUCADISPLAYBACKENDWINDOWS_H

#include "AliGPUCADisplayBackend.h"

class AliGPUCADisplayBackendWindows : public AliGPUCADisplayBackend
{
public:
	AliGPUCADisplayBackendWindows() = default;
	virtual ~AliGPUCADisplayBackendWindows() = default;
	
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
