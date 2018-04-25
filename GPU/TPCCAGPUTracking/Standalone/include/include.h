//Event display
#ifdef R__WIN32
#include <windows.h>
extern DWORD WINAPI OpenGLMain(LPVOID tmp);
extern void KillGLWindow();
extern HANDLE semLockDisplay;
#else
extern pthread_mutex_t semLockDisplay;
extern void* OpenGLMain( void* );
#endif

#ifdef BUILD_EVENT_DISPLAY
extern void ShowNextEvent();
extern void DisplayExit();
extern void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster);
extern volatile int exitButton;
extern volatile int displayEventNr;
extern volatile int sendKey;
#else
static void ShowNextEvent() {}
static void DisplayExit() {}
static void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster) {}
static int exitButton;
static volatile int displayEventNr;
static volatile int sendKey;
#endif

//QA
#ifdef BUILD_QA
extern void InitQA();
extern void RunQA(bool matchOnly = false);
extern int DrawQAHistograms();
extern void SetMCTrackRange(int min, int max);
extern bool SuppressTrack(int iTrack);
extern bool SuppressHit(int iHit);
extern int GetMCLabel(unsigned int trackId);
#else
static void InitQA() {}
static void RunQA(bool matchOnly = false) {}
static int DrawQAHistograms() {}
static void SetMCTrackRange(int min, int max) {}
static bool SuppressTrack(int iTrack) {}
static bool SuppressHit(int iHit) {}
static int GetMCLabel(unsigned int trackId) {}
#endif

//QA - Event Generator
extern void InitEventGenerator();
extern int GenerateEvent(const AliHLTTPCCAParam& sliceParam, char* filename);
extern void FinishEventGenerator();
