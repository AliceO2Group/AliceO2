#include "cmodules/qconfig.h"
#include "AliGPUReconstruction.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCADef.h"
#include "AliGPUCAQA.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <tuple>
#include <algorithm>
#include <random>
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

#ifndef WIN32
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fenv.h>
#include <locale.h>
#include <sys/stat.h>
#endif

#include "AliHLTTPCGMMergedTrack.h"
#include "Interface/outputtrack.h"
#include "AliGPUCASettings.h"
#include <vector>
#include <xmmintrin.h>

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterHardware.h"
#endif

#ifdef BUILD_EVENT_DISPLAY
#ifdef WIN32
#include "AliGPUCADisplayBackendWindows.h"
#else
#include "AliGPUCADisplayBackendX11.h"
#endif
#include "AliGPUCADisplayBackendGlut.h"
#endif

//#define BROKEN_EVENTS

std::unique_ptr<AliGPUReconstruction> rec;
std::unique_ptr<char> outputmemory;
std::unique_ptr<AliGPUCADisplayBackend> eventDisplay;
int nEventsInDirectory = 0;

int main(int argc, char** argv)
{
	//int iEventInTimeframe = 0;

#ifdef FE_DFL_DISABLE_SSE_DENORMS_ENV //Flush and load denormals to zero in any case
	fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
#else
#ifndef _MM_FLUSH_ZERO_ON
#define _MM_FLUSH_ZERO_ON 0x8000
#endif
#ifndef _MM_DENORMALS_ZERO_ON
#define _MM_DENORMALS_ZERO_ON 0x0040
#endif
	_mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
#endif

	int qcRet = qConfigParse(argc, (const char**) argv);
	if (qcRet)
	{
		if (qcRet != qConfig::qcrHelp) printf("Error parsing command line parameters\n");
		return(1);
	}
	if (configStandalone.printSettings) qConfigPrint();

	if (configStandalone.eventDisplay) configStandalone.runGPU = 0;
#ifndef WIN32
	setlocale(LC_ALL, "");
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
	if (configStandalone.fpe)
	{
		feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
	}
#else
	if (configStandalone.affinity != -1) {printf("Affinity setting not supported on Windows\n"); return(1);}
	if (configStandalone.fifo) {printf("FIFO Scheduler setting not supported on Windows\n"); return(1);}
	if (configStandalone.fpe) {printf("FPE not supported on Windows\n"); return(1);}
#endif
#ifndef BUILD_QA
	if (configStandalone.qa || configStandalone.eventGenerator) {printf("QA not enabled in build\n"); return(1);}
#endif
#ifndef BUILD_EVENT_DISPLAY
	if (configStandalone.eventDisplay) {printf("EventDisplay not enabled in build\n"); return(1);}
#endif
	if (configStandalone.configTF.bunchSim && configStandalone.configTF.nMerge) {printf("Cannot run --MERGE and --SIMBUNCHES togeterh\n"); return(1);}
	if (configStandalone.configQA.inputHistogramsOnly && configStandalone.configQA.compareInputs.size() == 0) {printf("Can only produce QA pdf output when input files are specified!\n"); return(1);}
	if ((configStandalone.nways & 1) == 0) {printf("nWay setting musst be odd number!\n"); return(1);}


	if (configStandalone.eventDisplay) configStandalone.noprompt = 1;
	if (configStandalone.DebugLevel >= 4) configStandalone.OMPThreads = 1;
#ifdef GPUCA_HAVE_OPENMP
	if (configStandalone.OMPThreads != -1) omp_set_num_threads(configStandalone.OMPThreads);
	else configStandalone.OMPThreads = omp_get_max_threads();
	if (configStandalone.OMPThreads != omp_get_max_threads())
	{
		printf("Cannot set number of OMP threads!\n");
		return(1);
	}
#else
	configStandalone.OMPThreads = 1;
#endif
	if (configStandalone.outputcontrolmem) outputmemory.reset(new char[configStandalone.outputcontrolmem]);

#if !(defined(BUILD_CUDA) || defined(BUILD_OPENCL))
	if (configStandalone.runGPU)
	{
		printf("GPU disables at build time!\n");
		getchar();
		return(1);
	}
#endif

	rec.reset(AliGPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : AliGPUReconstruction::DEVICE_TYPE_NAMES[AliGPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce));
	if (rec == nullptr)
	{
		printf("Error initializing AliGPUReconstruction\n");
		return(1);
	}

	if (!configStandalone.eventGenerator)
	{
		char filename[256];
		snprintf(filename, 256, "events/%s/", configStandalone.EventsDir);
		rec->ReadSettings(filename);
		printf("Read event settings from dir %s (solenoidBz: %f, home-made events %d, constBz %d)\n", filename, rec->GetEventSettings().solenoidBz, (int) rec->GetEventSettings().homemadeEvents, (int) rec->GetEventSettings().constBz);
	}
	
	AliGPUCASettingsEvent ev = rec->GetEventSettings();
	AliGPUCASettingsRec recSet;
	AliGPUCASettingsDeviceProcessing devProc;
	
	if (configStandalone.eventGenerator) ev.homemadeEvents = true;
	if (configStandalone.solenoidBz != -1e6f) ev.solenoidBz = configStandalone.solenoidBz;
	if (configStandalone.constBz) ev.constBz = true;
	if (configStandalone.cont) ev.continuousMaxTimeBin = -1;
	if (rec->GetDeviceType() == AliGPUReconstruction::DeviceType::CPU) printf("Standalone Test Framework for CA Tracker - Using CPU\n");
	else printf("Standalone Test Framework for CA Tracker - Using GPU\n");

	recSet.SetMinTrackPt(MIN_TRACK_PT_DEFAULT);
	recSet.NWays = configStandalone.nways;
	recSet.NWaysOuter = configStandalone.nwaysouter;
	recSet.RejectMode = configStandalone.rejectMode;
	recSet.SearchWindowDZDR = configStandalone.dzdr;
	if (configStandalone.referenceX < 500.) recSet.TrackReferenceX = configStandalone.referenceX;
	
	if (configStandalone.OMPThreads != -1) devProc.nThreads = configStandalone.OMPThreads;
	devProc.deviceNum = configStandalone.cudaDevice;
	devProc.debugLevel = configStandalone.DebugLevel;
	devProc.runQA = configStandalone.qa;
	if (configStandalone.eventDisplay)
	{
#ifdef BUILD_EVENT_DISPLAY
#ifdef WIN32
		if (configStandalone.eventDisplay == 1) eventDisplay.reset(new AliGPUCADisplayBackendWindows);
#else
		if (configStandalone.eventDisplay == 1) eventDisplay.reset(new AliGPUCADisplayBackendX11);
#endif
		else if (configStandalone.eventDisplay == 2) eventDisplay.reset(new AliGPUCADisplayBackendGlut);
#endif
		devProc.eventDisplay = eventDisplay.get();
	}
	devProc.nDeviceHelperThreads = configStandalone.helperThreads;
	devProc.globalInitMutex = configStandalone.gpuInitMutex;
	devProc.gpuDeviceOnly = configStandalone.oclGPUonly;
	
	rec->SetSettings(&ev, &recSet, &devProc);
	if (rec->Init())
	{
		printf("Error initializing AliGPUReconstruction!\n");
		return 1;
	}

	//hlt.SetRunMerger(configStandalone.merger);

	if (configStandalone.seed == -1)
	{
		std::random_device rd;
		configStandalone.seed = (int) rd();
		printf("Using seed %d\n", configStandalone.seed);
	}

	srand(configStandalone.seed);
	std::uniform_real_distribution<double> disUniReal(0., 1.);
	std::uniform_int_distribution<unsigned long long int> disUniInt;
	std::mt19937_64 rndGen1(configStandalone.seed);
	std::mt19937_64 rndGen2(disUniInt(rndGen1));

	int trainDist = 0;
	float collisionProbability = 0.;
	const int orbitRate = 11245;
	//const int driftTime = 93000;
	//const int TPCZ = 250;
	const int timeOrbit = 1000000000 / orbitRate;
	const int maxBunchesFull = timeOrbit / configStandalone.configTF.bunchSpacing;
	const int maxBunches = (timeOrbit - configStandalone.configTF.abortGapTime) / configStandalone.configTF.bunchSpacing;
	if (configStandalone.configTF.bunchSim)
	{
		for (nEventsInDirectory = 0;true;nEventsInDirectory++)
		{
			std::ifstream in;
			char filename[256];
			snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, nEventsInDirectory);
			in.open(filename, std::ifstream::binary);
			if (in.fail()) break;
			in.close();
		}
		if (configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount > maxBunches)
		{
			printf("Invalid timeframe settings: too many colliding bunches requested!\n");
			return(1);
		}
		trainDist = maxBunches / configStandalone.configTF.bunchTrainCount;
		collisionProbability = (float) configStandalone.configTF.interactionRate * (float) (maxBunchesFull * configStandalone.configTF.bunchSpacing / 1e9f) / (float) (configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount);
		printf("Timeframe settings: %d trains of %d bunches, bunch spacing: %d, train spacing: %dx%d, filled bunches %d / %d (%d), collision probability %f, mixing %d events\n",
			configStandalone.configTF.bunchTrainCount, configStandalone.configTF.bunchCount, configStandalone.configTF.bunchSpacing, trainDist, configStandalone.configTF.bunchSpacing, configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount, maxBunches, maxBunchesFull, collisionProbability, nEventsInDirectory);
	}

	if (configStandalone.eventGenerator)
	{
		printf("Event Generator Disabled\n");
		/*char dirname[256];
		snprintf(dirname, 256, "events/%s/", configStandalone.EventsDir);
		mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		rec->DumpSettings(dirname);

		InitEventGenerator();

		for (int i = 0;i < (configStandalone.NEvents == -1 ? 10 : configStandalone.NEvents);i++)
		{
			printf("Generating event %d/%d\n", i, configStandalone.NEvents == -1 ? 10 : configStandalone.NEvents);
			snprintf(dirname, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, i);
			GenerateEvent(hlt.Param(), dirname); TODO!
		}
		FinishEventGenerator();*/
		return(1);
	}
	else
	{
		if (1 || configStandalone.eventDisplay || configStandalone.qa) configStandalone.resetids = true; //Force resetting of IDs in standalone mode for the time being, otherwise late cluster attachment in the merger cannot work with the forced cluster ids in the merger.
		for (int jj = 0;jj < configStandalone.runs2;jj++)
		{
			auto& config = configStandalone.configTF;
			if (configStandalone.configQA.inputHistogramsOnly) break;
			if (configStandalone.runs2 > 1) printf("RUN2: %d\n", jj);
			int nEventsProcessed = 0;
			long long int nTracksTotal = 0;
			long long int nClustersTotal = 0;
			int nTotalCollisions = 0;
			//long long int eventStride = configStandalone.seed;
			//int simBunchNoRepeatEvent = configStandalone.StartEvent;
			std::vector<char> eventUsed(nEventsInDirectory);
			if (config.noEventRepeat == 2) memset(eventUsed.data(), 0, nEventsInDirectory * sizeof(eventUsed[0]));

			for (int i = configStandalone.StartEvent;i < configStandalone.NEvents || configStandalone.NEvents == -1;i++)
			{
				if (config.nTotalInTFEvents && nTotalCollisions >= config.nTotalInTFEvents) break;
				if (i != configStandalone.StartEvent) printf("\n");
				HighResTimer timerLoad;
				timerLoad.Start();
				/*if (config.bunchSim) TODO!
				{
					hlt.StartDataReading(0);
					long long int nBunch = -driftTime / config.bunchSpacing;
					long long int lastBunch = config.timeFrameLen / config.bunchSpacing;
					long long int lastTFBunch = lastBunch - driftTime / config.bunchSpacing;
					int nCollisions = 0, nBorderCollisions = 0, nTrainCollissions = 0, nMultipleCollisions = 0, nTrainMultipleCollisions = 0;
					int nTrain = 0;
					int mcMin = -1, mcMax = -1;
					while (nBunch < lastBunch)
					{
						for (int iTrain = 0;iTrain < config.bunchTrainCount && nBunch < lastBunch;iTrain++)
						{
							int nCollisionsInTrain = 0;
							for (int iBunch = 0;iBunch < config.bunchCount && nBunch < lastBunch;iBunch++)
							{
								const bool inTF = nBunch >= 0 && nBunch < lastTFBunch && (config.nTotalInTFEvents == 0 || nCollisions < nTotalCollisions + config.nTotalInTFEvents);
								if (mcMin == -1 && inTF) mcMin = hlt.GetNMCInfo();
								if (mcMax == -1 && nBunch >= 0 && !inTF) mcMax = hlt.GetNMCInfo();
								int nInBunchPileUp = 0;
								double randVal = disUniReal(inTF ? rndGen2 : rndGen1);
								double p = exp(-collisionProbability);
								double p2 = p;
								while (randVal > p)
								{
									if (config.noBorder && (nBunch < 0 || nBunch >= lastTFBunch)) break;
									if (nCollisionsInTrain >= nEventsInDirectory)
									{
										printf("Error: insuffient events for mixing!\n");
										return(1);
									}
									if (nCollisionsInTrain == 0 && config.noEventRepeat == 0) memset(eventUsed.data(), 0, nEventsInDirectory * sizeof(eventUsed[0]));
									if (inTF) nCollisions++;
									else nBorderCollisions++;
									int useEvent;
									if (config.noEventRepeat == 1) useEvent = simBunchNoRepeatEvent;
									else while (eventUsed[useEvent = (inTF && config.eventStride ? (eventStride += config.eventStride) : disUniInt(inTF ? rndGen2 : rndGen1)) % nEventsInDirectory]);
									if (config.noEventRepeat) simBunchNoRepeatEvent++;
									eventUsed[useEvent] = 1;
									std::ifstream in;
									char filename[256];
									snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, useEvent);
									in.open(filename, std::ifstream::binary);
									if (in.fail()) {printf("Unexpected error\n");return(1);}
									double shift = (double) nBunch * (double) config.bunchSpacing * (double) TPCZ / (double) driftTime;
									int nClusters = hlt.ReadEvent(in, true, true, shift, 0, (double) config.timeFrameLen * TPCZ / driftTime, true, configStandalone.qa || configStandalone.eventDisplay);
									printf("Placing event %4d+%d (ID %4d) at z %7.3f (time %'dns) %s(collisions %4d, bunch %6lld, train %3d) (%'10d clusters, %'10d MC labels, %'10d track MC info)\n", nCollisions, nBorderCollisions, useEvent, shift, (int) (nBunch * config.bunchSpacing), inTF ? " inside" : "outside", nCollisions, nBunch, nTrain, nClusters, hlt.GetNMCLabels(), hlt.GetNMCInfo());
									in.close();
									nInBunchPileUp++;
									nCollisionsInTrain++;
									p2 *= collisionProbability / nInBunchPileUp;
									p += p2;
									if (config.noEventRepeat && simBunchNoRepeatEvent >= nEventsInDirectory) nBunch = lastBunch;
									for (int sl = 0;sl < 36;sl++) SetCollisionFirstCluster(nCollisions + nBorderCollisions - 1, sl, hlt.ClusterData(sl).NumberOfClusters());
									SetCollisionFirstCluster(nCollisions + nBorderCollisions - 1, 36, hlt.GetNMCInfo());
								}
								if (nInBunchPileUp > 1) nMultipleCollisions++;
								nBunch++;
							}
							nBunch += trainDist - config.bunchCount;
							if (nCollisionsInTrain) nTrainCollissions++;
							if (nCollisionsInTrain > 1) nTrainMultipleCollisions++;
							nTrain++;
						}
						nBunch += maxBunchesFull - trainDist * config.bunchTrainCount;
					}
					nTotalCollisions += nCollisions;
					printf("Timeframe statistics: collisions: %d+%d in %d trains (inside / outside), average rate %f (pile up: in bunch %d, in train %d)\n", nCollisions, nBorderCollisions, nTrainCollissions, (float) nCollisions / (float) (config.timeFrameLen - driftTime) * 1e9, nMultipleCollisions, nTrainMultipleCollisions);

					if (!config.noBorder) SetMCTrackRange(mcMin, mcMax);
				}
				else*/
				{
					char filename[256];
					snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, i);
					int r = rec->ReadData(filename);
					if (r == 0)
					{
						printf("Event loaded with new format\n");
						//hlt.ResetMC(); TODO!
						if (rec->mIOPtrs.clustersNative) rec->ConvertNativeToClusterData();
					}
					else if (r == -1)
					{
						/* TODO!
#ifdef GPUCA_TPC_GEOMETRY_O2
						printf("Not attempting old format for O2\n");
						break;
#endif
						printf("Attempting old format\n");
						std::ifstream in;
						in.open(filename, std::ifstream::binary);
						if (in.fail())
						{
							if (i && configStandalone.NEvents == -1) break;
							printf("Error opening file %s\n", filename);
							getchar();
							return(1);
						}
						printf("Loading Event %d\n", i);

						float shift;
						if (config.nMerge && (config.shiftFirstEvent || iEventInTimeframe))
						{
							if (config.randomizeDistance)
							{
								shift = disUniReal(rndGen2);
								if (config.shiftFirstEvent)
								{
									if (iEventInTimeframe == 0) shift = shift * config.averageDistance;
									else shift = (iEventInTimeframe + shift) * config.averageDistance;
								}
								else
								{
									if (iEventInTimeframe == 0) shift = 0;
									else shift = (iEventInTimeframe - 0.5 + shift) * config.averageDistance;
								}
							}
							else
							{
								if (config.shiftFirstEvent)
								{
									shift = config.averageDistance * (iEventInTimeframe + 0.5);
								}
								else
								{
									shift = config.averageDistance * (iEventInTimeframe);
								}
							}
						}
						else
						{
							shift = 0.;
						}

						hlt.ReadEvent(in, configStandalone.resetids, config.nMerge > 0, shift);
						in.close();

						for (int sl = 0;sl < 36;sl++) SetCollisionFirstCluster(iEventInTimeframe, sl, hlt.ClusterData(sl).NumberOfClusters());
						SetCollisionFirstCluster(iEventInTimeframe, 36, hlt.GetNMCInfo());

						if (config.nMerge)
						{
							iEventInTimeframe++;
							if (iEventInTimeframe == config.nMerge || i == configStandalone.NEvents - 1)
							{
								iEventInTimeframe = 0;
							}
							else
							{
								continue;
							}
						}*/
					}
					else
					{
						if (i && configStandalone.NEvents == -1) break;
						return(1);
					}
				}
				printf("Loading time: %'d us\n", (int) (1000000 * timerLoad.GetCurrentElapsedTime()));

				printf("Processing Event %d\n", i);
				for (int j = 0;j < configStandalone.runs;j++)
				{
					if (configStandalone.runs > 1) printf("Run %d\n", j + 1);

					if (configStandalone.outputcontrolmem) rec->SetOutputControl(outputmemory.get(), configStandalone.outputcontrolmem);

					rec->SetResetTimers(j <= configStandalone.runsInit);
					int tmpRetVal = rec->RunStandalone();
					if (configStandalone.configRec.runTRD)
					{
						rec->RunTRDTracking();
					}
					
					int nTracks = 0, nClusters = 0, nAttachedClusters = 0, nAttachedClustersFitted = 0;
					for (int k = 0;k < rec->GetTPCMerger().NOutputTracks();k++)
					{
						if (rec->GetTPCMerger().OutputTracks()[k].OK())
						{
							nTracks++;
							nAttachedClusters += rec->GetTPCMerger().OutputTracks()[k].NClusters();
							nAttachedClustersFitted += rec->GetTPCMerger().OutputTracks()[k].NClustersFitted();
						}
					}
					nClusters = rec->GetTPCMerger().NClusters();
					printf("Output Tracks: %d (%d/%d attached clusters)\n", nTracks, nAttachedClusters, nAttachedClustersFitted);
					if (j == 0)
					{
						nTracksTotal += nTracks;
						nClustersTotal += nClusters;
						nEventsProcessed++;
					}
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
				}
			}
			if (nEventsProcessed > 1)
			{
				printf("Total: %lld clusters, %lld tracks\n", nClustersTotal, nTracksTotal);
			}
		}
	}
breakrun:

	if (configStandalone.qa)
	{
#ifndef WIN32
		if (configStandalone.fpe) fedisableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
		if (rec->GetQA() == nullptr)
		{
			printf("QA Unavailable\n");
			return 1;
		}
		rec->GetQA()->DrawQAHistograms();
	}

	rec->Finalize();

	if (!configStandalone.noprompt)
	{
		printf("Press a key to exit!\n");
		getchar();
	}
	return(0);
}
