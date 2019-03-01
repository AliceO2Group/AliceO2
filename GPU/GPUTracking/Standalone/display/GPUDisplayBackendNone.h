#ifndef GPUDISPLAYBACKENDNONE_H
#define GPUDISPLAYBACKENDNONE_H

#include "GPUDisplay.h"

class GPUDisplayBackendNone : public GPUDisplayBackend
{
	GPUDisplayBackendNone() = default;
	virtual ~GPUDisplayBackendNone() = default;
	
	virtual int StartDisplay() override {return 1;}
	virtual void DisplayExit() override {}
	virtual void SwitchFullscreen(bool set) override {}
	virtual void ToggleMaximized(bool set) override {}
	virtual void SetVSync(bool enable) override {}
	virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override {}
};
#endif
