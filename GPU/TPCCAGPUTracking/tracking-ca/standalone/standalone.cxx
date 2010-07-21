#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCADef.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#endif

int main(int argc, char** argv)
{
	int i;
	int RUNGPU = 1, SAVE = 0, DebugLevel = 0, NEvents = -1, StartEvent = 0, noprompt = 0, cudaDevice = -1, forceSlice = -1, sliceCount = -1, eventDisplay = 0, runs = 1, merger = 1, cleardebugout = 0;
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	char EventsDir[256] = "";

	if (hlt.GetGPUStatus() == 0)
	{
		printf("No GPU Available, restricting to CPU\n");
		RUNGPU = 0;
	}

  for( int i=0; i < argc; i++ ){
    if ( !strcmp( argv[i], "-CPU" ) ) 
    {
		printf("CPU enabled\n");
		hlt.ExitGPU();
		RUNGPU=0;        
    }
    
	if ( !strcmp( argv[i], "-GPU" ) ) 
    {
		if (hlt.GetGPUStatus())
		{
			printf("GPU enabled\n");
			RUNGPU=1;
		}
		else
		{
			printf("Cannot enable GPU\n");
		}
    }

	if ( !strcmp( argv[i], "-NOPROMPT" ) ) 
    {
	noprompt=1;        
    }
    
	if ( !strcmp( argv[i], "-SAVE" ) ) 
    {
	printf("Saving Tracks enabled\n");
	SAVE=1;        
    }
	
	if ( !strcmp( argv[i], "-DEBUG" ) && argc > i + 1)
	{
		DebugLevel = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-CLEARDEBUG" ) && argc > i + 1)
	{
		cleardebugout = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-SLICECOUNT" ) && argc > i + 1)
	{
		sliceCount = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-SLICE" ) && argc > i + 1)
	{
		forceSlice = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-CUDA" ) && argc > i + 1)
	{
		cudaDevice = atoi(argv[i + 1]);
	}
	
	if ( !strcmp( argv[i], "-N" ) && argc > i + 1)
	{
		NEvents = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-S" ) && argc > i + 1)
	{
		if (atoi(argv[i + 1]) > 0)
		StartEvent = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-MERGER" ) && argc > i + 1)
	{
		merger = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-RUNS" ) && argc > i + 1)
	{
		if (atoi(argv[i + 1]) > 0)
		runs = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-EVENTS" ) && argc > i + 1)
	{
		printf("Reading events from Directory events%s\n", argv[i + 1]);
		strcpy(EventsDir, argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-OMP" ) && argc > i + 1)
	{
		printf("Using %s OpenMP Threads\n", argv[i + 1]);
		omp_set_num_threads(atoi(argv[i + 1]));
	}

    if ( !strcmp( argv[i], "-DISPLAY" ) ) 
    {
		printf("Event Display enabled\n");
		hlt.ExitGPU();
		RUNGPU=1;
		eventDisplay = 1;
    }

#ifndef _WIN32
	if ( !strcmp( argv[i], "-AFFINITY") && argc > i + 1)
	{
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(atoi(argv[i + 1]), &mask);
		
		printf("Setting affinitiy to restrict on CPU %d\n", atoi(argv[i + 1]));
		if (0 != sched_setaffinity(0, sizeof(mask), &mask))
		{
			printf("Error setting CPU affinity\n");
			return(1);
		}
		
		printf("Setting FIFO scheduler\n");
		sched_param param;
		sched_getparam( 0, &param );
		param.sched_priority = 1;
		if ( 0 != sched_setscheduler( 0, SCHED_FIFO, &param ) ) {
			printf("Error setting scheduler\n");
			return(1);
		}
	}
#endif
  }	
	std::ofstream CPUOut, GPUOut;

	if (DebugLevel >= 4)
	{
		CPUOut.open("CPU.out");
		GPUOut.open("GPU.out");
		omp_set_num_threads(1);
	}

    hlt.SetGPUDebugLevel(DebugLevel, &CPUOut, &GPUOut);
	hlt.SetEventDisplay(eventDisplay);
	hlt.SetRunMerger(merger);
	if (RUNGPU)
		printf("Standalone Test Framework for CA Tracker - Using GPU\n");
	else
		printf("Standalone Test Framework for CA Tracker - Using CPU\n");

	if (RUNGPU && (cudaDevice != -1 || DebugLevel || (sliceCount != -1 && sliceCount != hlt.GetGPUMaxSliceCount())) && hlt.InitGPU(sliceCount, cudaDevice))
	{
		printf("Error Initialising GPU\n");
		printf("Press a key to exit!\n");
		getchar();
		return(1);
	}
	sliceCount = hlt.GetGPUMaxSliceCount();
	hlt.SetGPUTracker(RUNGPU);

	printf("Reading Settings\n");
	std::ifstream in("events/settings.dump");
	hlt.ReadSettings(in);
	in.close();

 for( int i=0; i < argc; i++ ){
    if ( !strcmp( argv[i], "-GPUOPT" ) && argc >= i + 1 ) 
    {
		int tmpOption = atoi(argv[i + 2]);
		printf("Setting GPU Option %s to %d\n", argv[i + 1], tmpOption);
		hlt.SetGPUTrackerOption(argv[i + 1], tmpOption);
    }
 }

	for (i = StartEvent;i < NEvents || NEvents == -1;i++)
	{
		char filename[256];
		sprintf(filename, "events%s/event.%d.dump", EventsDir, i);
		in.open(filename, std::ifstream::binary);
		if (in.fail())
		{
			if (NEvents == -1) break;
			printf("Error opening file %s\n", filename);
			getchar();
			return(1);
		}
		printf("Loading Event %d\n", i);
		hlt.StartDataReading(0);
		hlt.ReadEvent(in);
		hlt.FinishDataReading();
		in.close();
		printf("Processing Event %d\n", i);
		for (int j = 0;j < runs;j++)
		{
			if (runs > 1) printf("Run %d\n", j + 1);
			
			if (DebugLevel >= 4 && cleardebugout)
			{
				GPUOut.close();
				GPUOut.open("GPU.out");
				CPUOut.close();
				CPUOut.open("GPU.out");
			}
			if (hlt.ProcessEvent(forceSlice))
			{
				printf("Error occured\n");
				goto breakrun;
			}
		}
	}
breakrun:

	if (DebugLevel >= 4)
	{
		CPUOut.close();
		GPUOut.close();
	}

	hlt.ExitGPU();

	if (!noprompt)
	{
		printf("Press a key to exit!\n");
		getchar();
	}
	return(0);
}
