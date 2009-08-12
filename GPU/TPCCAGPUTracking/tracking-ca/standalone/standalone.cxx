#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCADef.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int main(int argc, char** argv)
{
	int i;
	int RUNGPU = 1, SAVE = 0, DebugLevel = 0, NEvents = 100, StartEvent = 0, noprompt = 0, cudaDevice = -1, forceSlice = -1;
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	char EventsDir[256] = "";

  for( int i=0; i < argc; i++ ){
    if ( !strcmp( argv[i], "-CPU" ) ) 
    {
	printf("CPU enabled\n");
	RUNGPU=0;        
    }
    
	if ( !strcmp( argv[i], "-GPU" ) ) 
    {
	printf("GPU enabled\n");
	RUNGPU=1;        
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
  }	
	std::ofstream CPUOut, GPUOut;

	if (DebugLevel >= 3)
	{
		CPUOut.open("CPU.out");
		GPUOut.open("GPU.out");
		omp_set_num_threads(1);
	}

    hlt.SetGPUDebugLevel(DebugLevel, &CPUOut, &GPUOut);
	if (RUNGPU)
		printf("Standalone Test Framework for CA Tracker - Using GPU\n");
	else
		printf("Standalone Test Framework for CA Tracker - Using CPU\n");

	if (RUNGPU && hlt.InitGPU(cudaDevice))
	{
		printf("Error Initialising GPU\n");
		printf("Press a key to exit!\n");
		getchar();
		return(1);
	}

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

	for (i = StartEvent;i < NEvents;i++)
	{
		char filename[256];
		sprintf(filename, "events%s/event.%d.dump", EventsDir, i);
		in.open(filename, std::ifstream::binary);
		if (in.fail())
		{
			printf("Error opening file\n");
			getchar();
			return(1);
		}
		printf("Loading Event %d\n", i);
		hlt.StartDataReading(0);
		hlt.ReadEvent(in);
		hlt.FinishDataReading();
		in.close();
		printf("Processing Event %d\n", i);
		hlt.ProcessEvent(forceSlice);
		/*if (hlt.ProcessEvent())
		{
			printf("Error occured\n");
			printf("Press a key to exit!\n");
			getchar();
			break;
		}*/
	}

	if (DebugLevel >= 3)
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
