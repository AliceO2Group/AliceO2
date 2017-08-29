#ifdef R__WIN32
extern DWORD WINAPI OpenGLMain(LPVOID tmp);
extern void KillGLWindow();
extern HANDLE semLockDisplay;
#else
extern pthread_mutex_t semLockDisplay;
extern void* OpenGLMain( void* );
#endif
extern void ShowNextEvent();
extern volatile int buttonPressed;
extern volatile int displayEventNr;
extern volatile int sendKey;
void RunQA();
void DrawQAHistograms();
