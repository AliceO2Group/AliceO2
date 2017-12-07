//Event display
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

//QA
extern void InitQA();
extern void RunQA();
extern int DrawQAHistograms();
extern void SetMCTrackRange(int min, int max);

//QA - Event Generator
extern void InitEventGenerator();
extern int GenerateEvent(const AliHLTTPCCAParam& sliceParam, char* filename);
extern void FinishEventGenerator();
