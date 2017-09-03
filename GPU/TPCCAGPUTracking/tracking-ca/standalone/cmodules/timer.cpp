#include "timer.h"
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <time.h>
#endif

HighResTimer::HighResTimer() : ElapsedTime(0), StartTime(0), running(0)
{
}

HighResTimer::~HighResTimer() {}

void HighResTimer::Start()
{
#ifdef _WIN32
	__int64 istart;
	QueryPerformanceCounter((LARGE_INTEGER*) &istart);
	StartTime = (double) istart;
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	StartTime = (double) tv.tv_sec * 1.0e9 + (double) tv.tv_nsec;
#endif
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
#ifdef _WIN32
	__int64 iend;
	QueryPerformanceCounter((LARGE_INTEGER*) &iend);
	EndTime = (double) iend;
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	EndTime = (double) tv.tv_sec * 1.0e9 + (double) tv.tv_nsec;
#endif
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
	double CurrentTime = 0;
#ifdef _WIN32
	__int64 iend;
	QueryPerformanceCounter((LARGE_INTEGER*) &iend);
	CurrentTime = (double) iend;
#else
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	CurrentTime = (double) tv.tv_sec * 1.0e9 + (double) tv.tv_nsec;
#endif
    double retVal = (CurrentTime - StartTime + ElapsedTime) / Frequency;
    if (reset)
    {
        ElapsedTime = 0;
        Start();
    }
	return(retVal);
}

double HighResTimer::GetFrequency()
{
#ifdef _WIN32
	__int64 ifreq;
	QueryPerformanceFrequency((LARGE_INTEGER*) &ifreq);
	return((double) ifreq);
#else
	return(1.0e9);
#endif
}

double HighResTimer::Frequency = HighResTimer::GetFrequency();
