#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCADef.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv)
{
	int i;
	int RUNGPU = 1, SAVE = 0, DebugLevel = 0, NEvents = 100, StartEvent = 0;
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

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
    
	if ( !strcmp( argv[i], "-SAVE" ) ) 
    {
	printf("Saving Tracks enabled\n");
	SAVE=1;        
    }
	
	if ( !strcmp( argv[i], "-DEBUG" ) && argc >= i)
	{
		DebugLevel = atoi(argv[i + 1]);
	}
	
	if ( !strcmp( argv[i], "-N" ) && argc >= i)
	{
		if (atoi(argv[i + 1]) < 100)
		NEvents = atoi(argv[i + 1]);
	}

	if ( !strcmp( argv[i], "-S" ) && argc >= i)
	{
		if (atoi(argv[i + 1]) > 0)
		StartEvent = atoi(argv[i + 1]);
	}
  }	

	if (RUNGPU)
		printf("Standalone Test Framework for CA Tracker - Using GPU\n");
	else
		printf("Standalone Test Framework for CA Tracker - Using CPU\n");

	if (RUNGPU && hlt.InitGPU())
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

	std::ofstream CPUOut, GPUOut;

	if (DebugLevel >= 3)
	{
		CPUOut.open("CPU.out");
		GPUOut.open("GPU.out");
	}
	hlt.SetGPUDebugLevel(DebugLevel, &CPUOut, &GPUOut);

	for (i = StartEvent;i < NEvents;i++)
	{
		char filename[256];
		sprintf(filename, "events/event.%d.dump", i);
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
		if (hlt.ProcessEvent())
		{
			printf("Error occured\n");
			printf("Press a key to exit!\n");
			getchar();
			break;
		}
	}

	if (DebugLevel >= 3)
	{
		CPUOut.close();
		GPUOut.close();
	}

	hlt.ExitGPU();

	printf("Press a key to exit!\n");
	getchar();
	return(0);
}