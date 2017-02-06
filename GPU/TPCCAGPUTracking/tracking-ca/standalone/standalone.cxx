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

#include "AliHLTTPCGMMergedTrack.h"

//#define BROKEN_EVENTS

int main(int argc, char** argv)
{
	int i;
	int RUNGPU = 1, DebugLevel = 0, NEvents = -1, StartEvent = 0, noprompt = 0, cudaDevice = -1, forceSlice = -1, sliceCount = -1, eventDisplay = 0, runs = 1, runs2 = 1, merger = 1, cleardebugout = 0, outputcontrolmem = 0, clusterstats = 0, continueOnError = 0, seed = -1;
	void* outputmemory = NULL;
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	char EventsDir[256] = "";

	if (hlt.GetGPUStatus() == 0)
	{
		printf("No GPU Available, restricting to CPU\n");
		RUNGPU = 0;
	}

	for( int i=0; i < argc; i++ )
	{
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

		if ( !strcmp( argv[i], "-CONTINUE" ) ) 
		{
			continueOnError=1;        
		}

		if ( !strcmp( argv[i], "-DEBUG" ) && argc > i + 1)
		{
			DebugLevel = atoi(argv[i + 1]);
		}

		if ( !strcmp( argv[i], "-SEED" ) && argc > i + 1)
		{
			seed = atoi(argv[i + 1]);
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

		if ( !strcmp( argv[i], "-CLUSTERSTATS" ) && argc > i + 1)
		{
			clusterstats = atoi(argv[i + 1]);
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

		if ( !strcmp( argv[i], "-RUNS2" ) && argc > i + 1)
		{
			if (atoi(argv[i + 1]) > 0)
				runs2 = atoi(argv[i + 1]);
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

		if ( !strcmp( argv[i], "-OUTPUTMEMORY" ) && argc > i + 1)
		{
			outputcontrolmem = atoi(argv[i + 1]);
			printf("Using %d bytes as output memory\n", outputcontrolmem);
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
		}

		if ( !strcmp( argv[i], "-FIFO"))
		{
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

	if (outputcontrolmem)
	{
		outputmemory = malloc(outputcontrolmem);
		if (outputmemory == 0)
		{
			printf("Memory allocation error\n");
			exit(1);
		}
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

	hlt.SetSettings();
	
	for( int i=0; i < argc; i++ ){
		if ( !strcmp( argv[i], "-GPUOPT" ) && argc >= i + 1 ) 
		{
			int tmpOption = atoi(argv[i + 2]);
			printf("Setting GPU Option %s to %d\n", argv[i + 1], tmpOption);
			hlt.SetGPUTrackerOption(argv[i + 1], tmpOption);
		}
	}

	int ClusterStat[HLTCA_ROW_COUNT + 1];
	if (clusterstats >= 2)
	{
		memset(ClusterStat, 0, (HLTCA_ROW_COUNT + 1) * sizeof(int));
	}

	if (seed != -1) srand(seed);

	for (int jj = 0;jj < runs2;jj++) {if (runs2 > 1) printf("RUN2: %d\n", jj);
	for (i = StartEvent;i < NEvents || NEvents == -1;i++)
	{
		std::ifstream in;
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
		
#ifdef BROKEN_EVENTS
		int break_slices = rand() % 36;
		for (int k = 0;k < 2;k++) if (rand() % 2) break_slices /= 2;
//		break_slices = 1;
		for (int j = 0;j < break_slices;j++)
		{
			int break_slice = rand() % 36;
			int break_clusters = rand() % 500000;
			for (int k = 0;k < 2;k++) if (rand() % 2) break_clusters /= 2;
			//printf("Adding %d random clusters to slice %d\n", break_clusters, break_slice);
			for (int k = 0;k < break_clusters;k++)
			{
				hlt.ReadCluster(rand() % 1000000000, break_slice, rand() % HLTCA_ROW_COUNT, (float) (rand() % 200) / 10, (float) (rand() % 200) / 10, (float) (rand() % 200) / 10, (float) (rand() % 200) / 10);
			}
		}
#endif
		
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

			if (outputcontrolmem)
			{
				hlt.SetOutputControl((char*) outputmemory, outputcontrolmem);
			}

			if (hlt.ProcessEvent(forceSlice) && !continueOnError)
			{
				printf("Error occured\n");
#ifdef BROKEN_EVENTS
				continue;
#endif
				goto breakrun;
			}

			if (merger)
			{
				const AliHLTTPCGMMerger& merger = hlt.Merger();
				if (clusterstats)
				{
					unsigned int minid = 2000000000, maxid = 0;
					for (int i = 0;i < merger.NOutputTrackClusters();i++)
					{
						if (merger.OutputClusterIds()[i] < minid) minid = merger.OutputClusterIds()[i];
						if (merger.OutputClusterIds()[i] > maxid) maxid = merger.OutputClusterIds()[i];
					}
					printf("\nCluster id range: %d %d\n", minid, maxid);
					char* idused = new char[maxid - minid + 1];
					memset(idused, 0, maxid - minid);
					for (int itrack = 0;itrack < merger.NOutputTracks();itrack++)
					{
						if (merger.OutputTracks()[itrack].OK())
						{
							for (int icluster = 0;icluster < merger.OutputTracks()[itrack].NClusters();icluster++)
							{
								idused[merger.OutputClusterIds()[merger.OutputTracks()[itrack].FirstClusterRef() + icluster] - minid] = 1;
							}
						}
					}
					int nClustersUsed = 0;
					for (unsigned int i = 0;i < maxid - minid;i++)
					{
						nClustersUsed += idused[i];
					}
					delete[] idused;
					int totalclusters = 0;
					for (int i = 0;i < hlt.NSlices();i++)
					{
						totalclusters += hlt.ClusterData(i).NumberOfClusters();
					}
					printf("Clusters used: %d of %d, %4.2f%%\n", nClustersUsed, totalclusters, 100. * (float) nClustersUsed / (float) totalclusters);
				}

				if (clusterstats >= 2)
				{
					for (int i = 0;i < merger.NOutputTracks();i++)
					{
						const AliHLTTPCGMMergedTrack* tracks = merger.OutputTracks();
						int nCluster = tracks[i].NClusters();
						if (nCluster < 0)
						{
							printf("Error in Merger: Track %d contains %d clusters\n", i, nCluster);
							return(1);
						}
						else
						{
							if (nCluster >= HLTCA_ROW_COUNT) nCluster = HLTCA_ROW_COUNT;
							ClusterStat[nCluster]++;
						}
					}
				}
			}
		}
	}}
breakrun:

	if (clusterstats >= 2)
	{
		for (int i = 0;i < (HLTCA_ROW_COUNT + 1);i++)
		{
			printf("CLUSTER STATS: Clusters: %3d, Tracks: %5d\n", i, ClusterStat[i]);
		}
	}

	if (DebugLevel >= 4)
	{
		CPUOut.close();
		GPUOut.close();
	}

	hlt.Merger().Clear();
	hlt.Merger().SetGPUTracker(NULL);

	hlt.ExitGPU();

	if (outputcontrolmem)
	{
		free(outputmemory);
	}
	else
	{

	}

	if (!noprompt)
	{
		printf("Press a key to exit!\n");
		getchar();
	}
	return(0);
}
