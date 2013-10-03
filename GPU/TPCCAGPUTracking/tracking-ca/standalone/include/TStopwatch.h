// @(#)root/base:$Name:  $:$Id: TStopwatch.h,v 1.4 2004/04/26 14:41:31 brun Exp $
// Author: Fons Rademakers   11/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef _Stopwatch
#define _Stopwatch

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStopwatch                                                           //
//                                                                      //
// Stopwatch class. This class returns the real and cpu time between    //
// the start and stop events.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


class TStopwatch {

private:
   enum EState { kUndefined, kStopped, kRunning };

   double     fStartRealTime;   //wall clock start time
   double     fStopRealTime;    //wall clock stop time
   double     fStartCpuTime;    //cpu start time
   double     fStopCpuTime;     //cpu stop time
   double     fTotalCpuTime;    //total cpu time
   double     fTotalRealTime;   //total real time
   EState       fState;           //stopwatch state
   int        fCounter;         //number of times the stopwatch was started

   inline double GetRealTime();
   inline double GetCPUTime();

#ifdef WIN32
   unsigned long long int fFrequency;	//Timer Frequency
#endif

public:
   inline TStopwatch();
   inline void        Start(int reset = 1);
   inline void        Stop();
   inline void        Continue();
   inline int         Counter() const { return fCounter; }
   inline double      RealTime();
   inline void        Reset() { ResetCpuTime(); ResetRealTime(); }
   inline void        ResetCpuTime(double time = 0) { Stop();  fTotalCpuTime = time; }
   inline void        ResetRealTime(double time = 0) { Stop(); fTotalRealTime = time; }
   inline double      CpuTime();
};


//______________________________________________________________________________


#if defined(R__UNIX)
#  include <sys/time.h>
#  include <sys/times.h>
#  include <unistd.h>
static double gTicks = 0;
#elif defined(R__VMS)
#  include <time.h>
#  include <unistd.h>
static double gTicks = 1000;
#elif defined(R__WIN32)
//#  include "TError.h"
const double gTicks = 1.0e-7;
//#  include "Windows4Root.h"
#include <windows.h>
#else
#include <time.h>
#endif


inline TStopwatch::TStopwatch() : fStartRealTime(0), fStopRealTime(0), fStartCpuTime(0), fStopCpuTime(0), fTotalCpuTime(0), fTotalRealTime(0), fState(kUndefined), fCounter(0)
{
   // Create a stopwatch and start it.

#ifdef R__UNIX
   if (!gTicks)
      gTicks = (double)sysconf(_SC_CLK_TCK);
#endif

#ifdef WIN32
	QueryPerformanceFrequency((LARGE_INTEGER*) &fFrequency);
#endif

   Start();
}

//______________________________________________________________________________
inline void TStopwatch::Start(int reset)
{
   // Start the stopwatch. If reset is kTRUE reset the stopwatch before
   // starting it (including the stopwatch counter).
   // Use kFALSE to continue timing after a Stop() without
   // resetting the stopwatch.

   if (reset) {
      fState         = kUndefined;
      fTotalCpuTime  = 0;
      fTotalRealTime = 0;
      fCounter       = 0;
   }
   if (fState != kRunning) {
      fStartRealTime = GetRealTime();
      fStartCpuTime  = GetCPUTime();
   }
   fState = kRunning;
   fCounter++;
}

//______________________________________________________________________________
inline void TStopwatch::Stop()
{
   // Stop the stopwatch.

   fStopRealTime = GetRealTime();
   fStopCpuTime  = GetCPUTime();

   if (fState == kRunning) {
      fTotalCpuTime  += fStopCpuTime  - fStartCpuTime;
      fTotalRealTime += fStopRealTime - fStartRealTime;
   }
   fState = kStopped;
}

//______________________________________________________________________________
inline void TStopwatch::Continue()
{
   // Resume a stopped stopwatch. The stopwatch continues counting from the last
   // Start() onwards (this is like the laptimer function).

  if (fState == kUndefined){
    //cout<< "stopwatch not started"<<endl;
    return;
  }
   if (fState == kStopped) {
      fTotalCpuTime  -= fStopCpuTime  - fStartCpuTime;
      fTotalRealTime -= fStopRealTime - fStartRealTime;
   }

   fState = kRunning;
}

//______________________________________________________________________________
inline double TStopwatch::RealTime()
{
   // Return the realtime passed between the start and stop events. If the
   // stopwatch was still running stop it first.

  if (fState == kUndefined){
    //cout<<"stopwatch not started"<<endl;
    return 0;
  }
   if (fState == kRunning)
      Stop();

   return fTotalRealTime;
}

//______________________________________________________________________________
inline double TStopwatch::CpuTime()
{
   // Return the cputime passed between the start and stop events. If the
   // stopwatch was still running stop it first.

   if (fState == kUndefined){
     //cout<<"stopwatch not started"<<endl;
     return 0;
   }
   if (fState == kRunning)
      Stop();

   return fTotalCpuTime;
}

//______________________________________________________________________________
inline double TStopwatch::GetRealTime()
{
#if defined(R__UNIX)
  struct timeval tp;
  gettimeofday(&tp, 0);
  return tp.tv_sec + (tp.tv_usec)*1.e-6;
#elif defined(WIN32)
  unsigned long long int a;
  QueryPerformanceCounter((LARGE_INTEGER*) &a);
  return((double) a / (double) fFrequency);
#else
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + (t.tv_nsec)*1.e-9;
  //return 0;
#endif
}

//______________________________________________________________________________
inline double TStopwatch::GetCPUTime()
{
   // Private static method returning system CPU time.

#if defined(R__UNIX)
   struct tms cpt;
   times(&cpt);
   return (double)(cpt.tms_utime+cpt.tms_stime) / gTicks;
#elif defined(R__VMS)
   return (double)clock() / gTicks;
#elif defined(WIN32)

   OSVERSIONINFO OsVersionInfo;

   //         Value                      Platform
   //----------------------------------------------------
   //  VER_PLATFORM_WIN32s          Win32s on Windows 3.1
   //  VER_PLATFORM_WIN32_WINDOWS   Win32 on Windows 95
   //  VER_PLATFORM_WIN32_NT        Windows NT
   //
   OsVersionInfo.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
   GetVersionEx(&OsVersionInfo);
   if (OsVersionInfo.dwPlatformId == VER_PLATFORM_WIN32_NT) {
      DWORD       ret;
      FILETIME    ftCreate,       // when the process was created
                  ftExit;         // when the process exited

      union {
         FILETIME ftFileTime;
         __int64  ftInt64;
      } ftKernel; // time the process has spent in kernel mode

      union {
         FILETIME ftFileTime;
         __int64  ftInt64;
      } ftUser;   // time the process has spent in user mode

      HANDLE hProcess = GetCurrentProcess();
      ret = GetProcessTimes (hProcess, &ftCreate, &ftExit,
                                       &ftKernel.ftFileTime,
                                       &ftUser.ftFileTime);
      if (ret != TRUE) {
	ret = GetLastError ();
	//cout<<" Error on GetProcessTimes 0x%lx"<<endl;
	return 0;
      }

      // Process times are returned in a 64-bit structure, as the number of
      // 100 nanosecond ticks since 1 January 1601.  User mode and kernel mode
      // times for this process are in separate 64-bit structures.
      // To convert to floating point seconds, we will:
      //
      // Convert sum of high 32-bit quantities to 64-bit int

      return (double) (ftKernel.ftInt64 + ftUser.ftInt64) * gTicks;
   } else
      return GetRealTime();
#else
	return(0);
#endif
}

#endif
#endif
