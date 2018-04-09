#include "timer.h"
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#elif defined(__MACH__) || defined(__APPLE__)
#include <mach/clock.h>
#include <mach/mach.h>
#else
#include <time.h>
#endif

HighResTimer::HighResTimer() : ElapsedTime(0), StartTime(0), running(0)
{
}

HighResTimer::~HighResTimer() {}

inline double HighResTimer::GetTime()
{
#ifdef _WIN32
	__int64 istart;
	QueryPerformanceCounter((LARGE_INTEGER*) &istart);
	return((double) istart);
#elif defined(__MACH__) || defined(__APPLE__)
	clock_serv_t cclock;
	mach_timespec_t mts;
	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	return((double) mts.tv_sec * 1.0e9 + (double) mts.tv_nsec);
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return((double) tv.tv_sec * 1.0e9 + (double) tv.tv_nsec);
#endif
}

inline double HighResTimer::GetFrequency()
{
#ifdef _WIN32
	__int64 ifreq;
	QueryPerformanceFrequency((LARGE_INTEGER*) &ifreq);
	return((double) ifreq);
#else
	return(1.0e9);
#endif
}

void HighResTimer::Start()
{
	StartTime = GetTime();
	running = 1;
}

void HighResTimer::ResetStart()
{
	ElapsedTime = 0;
	Start();
}

void HighResTimer::Stop()
{
	if (running == 0) return;
	running = 0;
	double EndTime = 0;
	EndTime = GetTime();
	ElapsedTime += EndTime - StartTime;
}

void HighResTimer::Reset()
{
	ElapsedTime = 0;
	StartTime = 0;
	running = 0;
}

double HighResTimer::GetElapsedTime()
{
	return ElapsedTime / Frequency;
}

double HighResTimer::GetCurrentElapsedTime(bool reset)
{
	if (running == 0)
	{
		double retVal = GetElapsedTime();
		Reset();
		return(retVal);
	}
	double CurrentTime = GetTime();
	double retVal = (CurrentTime - StartTime + ElapsedTime) / Frequency;
	if (reset)
	{
		ElapsedTime = 0;
		Start();
	}
	return(retVal);
}

double HighResTimer::Frequency = HighResTimer::GetFrequency();
