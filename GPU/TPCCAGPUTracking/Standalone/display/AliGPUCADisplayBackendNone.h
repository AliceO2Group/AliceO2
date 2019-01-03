#ifndef ALIGPUCADISPLAYBACKENDNONE_H
#define ALIGPUCADISPLAYBACKENDNONE_H

#include "AliGPUCADisplay.h"

class AliGPUCADisplayBackendNone : public AliGPUCADisplayBackend
{
	AliGPUCADisplayBackendNone() = default;
	~AliGPUCADisplayBackendNone() = default;
	
	virtual void StartDisplay() override {}
	virtual void DisplayExit() override {}
	virtual void SwitchFullscreen() override {}
	virtual void ToggleMaximized(bool set) override {}
	virtual void SetVSync(bool enable) override {}
	virtual void OpenGLPrint(const char* s) override {}
};
#endif
