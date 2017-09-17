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
#include "interface/outputtrack.h"
#include "include.h"
#include <vector>

#include "cmodules/qconfig.h"

//#define BROKEN_EVENTS

int main(int argc, char** argv)
{
	void* outputmemory = NULL;
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	int iEventInTimeframe = 0;
	
	if (hlt.GetGPUStatus() == 0)
	{
		printf("No GPU Available, restricting to CPU\n");
		configStandalone.RUNGPU = 0;
	}
	
	if (qConfigParse(argc, (const char**) argv)) return(1);
	if (configStandalone.RUNGPU && hlt.GetGPUStatus() == 0) {printf("Cannot enable GPU\n"); configStandalone.RUNGPU = 0;}
	if (configStandalone.RUNGPU == 0 || configStandalone.eventDisplay) hlt.ExitGPU();
#ifndef _WIN32
	if (configStandalone.affinity != -1)
	{
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(configStandalone.affinity, &mask);

		printf("Setting affinitiy to restrict on CPU %d\n", configStandalone.affinity);
		if (0 != sched_setaffinity(0, sizeof(mask), &mask))
		{
			printf("Error setting CPU affinity\n");
			return(1);
		}
	}
	if (configStandalone.fifo)
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
#else
	if (configStandalone.affinity != -1) {printf("Affinity setting not supported on Windows\n"); return(1);}
	if (configStandalone.fifi) {printf("FIFO Scheduler setting not supported on Windows\n"); return(1);}
#endif

	if (configStandalone.OMPTHreads != -1) omp_set_num_threads(configStandalone.OMPTHreads);
	
	std::ofstream CPUOut, GPUOut;
	FILE* fpBinaryOutput = NULL;

	if (configStandalone.eventDisplay) configStandalone.noprompt = 1;
	if (configStandalone.DebugLevel >= 4)
	{
		CPUOut.open("CPU.out");
		GPUOut.open("GPU.out");
		omp_set_num_threads(1);
	}
	if (configStandalone.writebinary)
	{
		if ((fpBinaryOutput = fopen("output.bin", "w+b")) == NULL)
		{
			printf("Error opening output file\n");
			exit(1);
		}
	}

	if (configStandalone.outputcontrolmem)
	{
		outputmemory = malloc(configStandalone.outputcontrolmem);
		if (outputmemory == 0)
		{
			printf("Memory allocation error\n");
			exit(1);
		}
	}
	hlt.SetGPUDebugLevel(configStandalone.DebugLevel, &CPUOut, &GPUOut);
	hlt.SetEventDisplay(configStandalone.eventDisplay);
	hlt.SetRunQA(configStandalone.qa);
	hlt.SetRunMerger(configStandalone.merger);
	if (configStandalone.RUNGPU)
		printf("Standalone Test Framework for CA Tracker - Using GPU\n");
	else
		printf("Standalone Test Framework for CA Tracker - Using CPU\n");

	if (configStandalone.RUNGPU && (configStandalone.cudaDevice != -1 || configStandalone.DebugLevel || (configStandalone.sliceCount != -1 && configStandalone.sliceCount != hlt.GetGPUMaxSliceCount())) && hlt.InitGPU(configStandalone.sliceCount, configStandalone.cudaDevice))
	{
		printf("Error Initialising GPU\n");
		printf("Press a key to exit!\n");
		getchar();
		return(1);
	}
	configStandalone.sliceCount = hlt.GetGPUMaxSliceCount();
	hlt.SetGPUTracker(configStandalone.RUNGPU);

	hlt.SetSettings();
	if (configStandalone.lowpt) hlt.SetHighQPtForward(1./0.1);
	hlt.SetNWays(configStandalone.nways);
	if (configStandalone.cont) hlt.SetContinuousTracking(configStandalone.cont);
	if (configStandalone.dzdr != 0.) hlt.SetSearchWindowDZDR(configStandalone.dzdr);
	hlt.UpdateGPUSliceParam();
	hlt.SetGPUTrackerOption("GlobalTracking", 1);
	
	for(int i = 0;i < argc;i++)
	{
		if ( !strcmp( argv[i], "-GPUOPT" ) && argc >= i + 1 ) 
		{
			int tmpOption = atoi(argv[i + 2]);
			printf("Setting GPU Option %s to %d\n", argv[i + 1], tmpOption);
			hlt.SetGPUTrackerOption(argv[i + 1], tmpOption);
		}
	}

	if (configStandalone.seed != -1) srand(configStandalone.seed);

	for (int jj = 0;jj < configStandalone.runs2;jj++) {if (configStandalone.runs2 > 1) printf("RUN2: %d\n", jj);
	for (int i = configStandalone.StartEvent;i < configStandalone.NEvents || configStandalone.NEvents == -1;i++)
	{
		std::ifstream in;
		char filename[256];
		sprintf(filename, "events%s/event.%d.dump", configStandalone.EventsDir, i);
		in.open(filename, std::ifstream::binary);
		if (in.fail())
		{
			if (configStandalone.NEvents == -1) break;
			printf("Error opening file %s\n", filename);
			getchar();
			return(1);
		}
		printf("Loading Event %d\n", i);
		
		float shift;
		if (configStandalone.configTF.nMerge && (configStandalone.configTF.shiftFirstEvent || iEventInTimeframe))
		{
			if (configStandalone.configTF.randomizeDistance)
			{
				shift = (double) rand() / (double) RAND_MAX;
				if (configStandalone.configTF.shiftFirstEvent)
				{
					if (iEventInTimeframe == 0) shift = shift * configStandalone.configTF.averageDistance;
					else shift = (iEventInTimeframe + shift) * configStandalone.configTF.averageDistance;
				}
				else
				{
					if (iEventInTimeframe == 0) shift = 0;
					else shift = (iEventInTimeframe - 0.5 + shift) * configStandalone.configTF.averageDistance;
				}
			}
			else
			{
				if (configStandalone.configTF.shiftFirstEvent)
				{
					shift = configStandalone.configTF.averageDistance * (iEventInTimeframe + 0.5);
				}
				else
				{
					shift = configStandalone.configTF.averageDistance * (iEventInTimeframe);
				}
			}
		}
		else
		{
			shift = 0.;
		}

		if (configStandalone.configTF.nMerge == 0 || iEventInTimeframe == 0) hlt.StartDataReading(0);
		if (configStandalone.eventDisplay || configStandalone.qa) configStandalone.resetids = true;
		hlt.ReadEvent(in, configStandalone.resetids, configStandalone.configTF.nMerge > 0, shift);
		
		in.close();
		
		iEventInTimeframe++;
		if (configStandalone.configTF.nMerge)
		{
			if (iEventInTimeframe == configStandalone.configTF.nMerge || i == configStandalone.NEvents - 1)
			{
				iEventInTimeframe = 0;
			}
			else
			{
				continue;
			}
		}
		hlt.FinishDataReading();

		printf("Processing Event %d\n", i);
		for (int j = 0;j < configStandalone.runs;j++)
		{
			if (configStandalone.runs > 1) printf("Run %d\n", j + 1);
			
			if (configStandalone.DebugLevel >= 4 && configStandalone.cleardebugout)
			{
				GPUOut.close();
				GPUOut.open("GPU.out");
				CPUOut.close();
				CPUOut.open("GPU.out");
			}

			if (configStandalone.outputcontrolmem)
			{
				hlt.SetOutputControl((char*) outputmemory, configStandalone.outputcontrolmem);
			}

			int tmpRetVal = hlt.ProcessEvent(configStandalone.forceSlice);
			if (tmpRetVal == 2)
			{
				configStandalone.continueOnError = 0; //Forced exit from event display loop
				configStandalone.noprompt = 1;
			}
			if (tmpRetVal && !configStandalone.continueOnError)
			{
				if (tmpRetVal != 2) printf("Error occured\n");
				goto breakrun;
			}

			if (configStandalone.merger)
			{
				const AliHLTTPCGMMerger& merger = hlt.Merger();
				if (configStandalone.resetids && (configStandalone.writeoutput || configStandalone.writebinary))
				{
					printf("\nWARNING: Renumbering Cluster IDs, Cluster IDs in output do NOT match IDs from input\n\n");
				}
				if (configStandalone.writeoutput)
				{
					char filename[1024];
					sprintf(filename, "output.%d.txt", i);
					printf("Creating output file %s\n", filename);
					FILE* foutput = fopen(filename, "w+");
					if (foutput == NULL)
					{
						printf("Error creating file\n");
						exit(1);
					}
					fprintf(foutput, "Event %d\n", i);
					for (int k = 0;k < merger.NOutputTracks();k++)
					{
						const AliHLTTPCGMMergedTrack& track = merger.OutputTracks()[k];
						const AliHLTTPCGMTrackParam& param = track.GetParam();
						fprintf(foutput, "Track %d: %4s Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f q/Pt %f - Clusters ", k, track.OK() ? "OK" : "FAIL", track.GetAlpha(), param.GetX(), param.GetY(), param.GetZ(), param.GetSinPhi(), param.GetDzDs(), param.GetQPt());
						for (int l = 0;l < track.NClusters();l++)
						{
							fprintf(foutput, "%d ", merger.OutputClusterIds()[track.FirstClusterRef() + l]);
						}
						fprintf(foutput, "\n");
					}
					fclose(foutput);
				}
				
				if (configStandalone.writebinary)
				{
					int numTracks = merger.NOutputTracks();
					fwrite(&numTracks, sizeof(numTracks), 1, fpBinaryOutput);
					for (int k = 0;k < numTracks;k++)
					{
						OutputTrack tmpTrack;
						const AliHLTTPCGMMergedTrack& track = merger.OutputTracks()[k];
						const AliHLTTPCGMTrackParam& param = track.GetParam();
						
						tmpTrack.Alpha = track.GetAlpha();
						tmpTrack.X = param.GetX();
						tmpTrack.Y = param.GetY();
						tmpTrack.Z = param.GetZ();
						tmpTrack.SinPhi = param.GetSinPhi();
						tmpTrack.DzDs = param.GetDzDs();
						tmpTrack.QPt = param.GetQPt();
						tmpTrack.NClusters = track.NClusters();
						tmpTrack.FitOK = track.OK();
						fwrite(&tmpTrack, sizeof(tmpTrack), 1, fpBinaryOutput);
						const unsigned int* hitIds = merger.OutputClusterIds() + track.FirstClusterRef();
						fwrite(hitIds, sizeof(hitIds[0]), track.NClusters(), fpBinaryOutput);
					}
				}
				
			}
		}
	}}
breakrun:

#ifdef BUILD_QA
	if (configStandalone.qa) DrawQAHistograms();
#endif

	if (configStandalone.DebugLevel >= 4)
	{
		CPUOut.close();
		GPUOut.close();
	}
	if (configStandalone.writebinary) fclose(fpBinaryOutput);

	hlt.Merger().Clear();
	hlt.Merger().SetGPUTracker(NULL);

	hlt.ExitGPU();

	if (configStandalone.outputcontrolmem)
	{
		free(outputmemory);
	}

	if (!configStandalone.noprompt)
	{
		printf("Press a key to exit!\n");
		getchar();
	}
	return(0);
}
