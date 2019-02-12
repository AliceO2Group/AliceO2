#include "AliGPUDisplayBackend.h"
#include "AliGPUDisplay.h"

void* AliGPUDisplayBackend::OpenGLWrapper(void* ptr)
{
	AliGPUDisplayBackend* me = (AliGPUDisplayBackend*) ptr;
	int retVal = me->OpenGLMain();
	if (retVal == -1) me->InitGL(true);
	return((void*) (size_t) retVal);
}

void AliGPUDisplayBackend::HandleSendKey()
{
	if (sendKey)
	{
		mDisplay->HandleSendKey(sendKey);
		sendKey = 0;
	}
}

void AliGPUDisplayBackend::HandleKeyRelease(unsigned char key) {mDisplay->HandleKeyRelease(key);}
int AliGPUDisplayBackend::DrawGLScene(bool mixAnimation, float animateTime) {return mDisplay->DrawGLScene(mixAnimation, animateTime);}
void AliGPUDisplayBackend::ReSizeGLScene(int width, int height) {display_height = height; display_width = width; mDisplay->ReSizeGLScene(width, height);}
int AliGPUDisplayBackend::InitGL(bool initFailure) {return mDisplay->InitGL(initFailure);}
void AliGPUDisplayBackend::ExitGL() {return mDisplay->ExitGL();}
bool AliGPUDisplayBackend::EnableSendKey() {return true;}
