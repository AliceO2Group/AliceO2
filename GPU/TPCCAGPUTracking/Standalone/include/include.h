#ifndef hasethstrhstr
#define hasethstrhstr

struct AliGPUCAParam;

//Event display
#ifdef WIN32
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
static void SetCollisionFirstCluster(unsigned int /*collision*/, int /*slice*/, int /*cluster*/) {}
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
extern bool clusterRemovable(int cid, bool prot);
#else
static void InitQA() {}
static void RunQA(bool matchOnly = false) {}
static int DrawQAHistograms() {return 0;}
static void SetMCTrackRange(int min, int max) {}
static bool SuppressTrack(int iTrack) {return false;}
static bool SuppressHit(int iHit) {return false;}
static int GetMCLabel(unsigned int trackId) {return(-1);}
static bool clusterRemovable(int cid, bool prot) {return false;}
#endif

//QA - Event Generator
extern void InitEventGenerator();
extern int GenerateEvent(const AliGPUCAParam& sliceParam, char* filename);
extern void FinishEventGenerator();

#endif
