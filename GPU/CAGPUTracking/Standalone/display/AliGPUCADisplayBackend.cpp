#include "AliGPUCADisplayBackend.h"
#include "AliGPUCADisplay.h"

void* AliGPUCADisplayBackend::OpenGLWrapper(void* ptr)
{
	AliGPUCADisplayBackend* me = (AliGPUCADisplayBackend*) ptr;
	return me->OpenGLMain();
}

void AliGPUCADisplayBackend::HandleKeyRelease(unsigned char key) {mDisplay->HandleKeyRelease(key);}
int AliGPUCADisplayBackend::DrawGLScene(bool mixAnimation, float animateTime) {return mDisplay->DrawGLScene(mixAnimation, animateTime);}
void AliGPUCADisplayBackend::HandleSendKey() {mDisplay->HandleSendKey();}
void AliGPUCADisplayBackend::ReSizeGLScene(int width, int height) {mDisplay->ReSizeGLScene(width, height);}
int AliGPUCADisplayBackend::InitGL() {return mDisplay->InitGL();}
void AliGPUCADisplayBackend::ExitGL() {return mDisplay->ExitGL();}
