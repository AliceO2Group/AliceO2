#ifndef CAGPU_STANDALONE_HEADER
#define CAGPU_STANDALONE_HEADER

struct AliGPUCAParam;

//Event display
#ifdef WIN32
#include <windows.h>
extern DWORD WINAPI OpenGLMain(LPVOID tmp);
extern void KillGLWindow();
extern HANDLE semLockDisplay;
#else
extern pthread_mutex_t semLockDisplay;
#endif

#endif
