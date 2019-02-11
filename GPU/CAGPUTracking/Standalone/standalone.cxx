#include "cmodules/qconfig.h"
#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionTimeframe.h"
#include "AliGPUTPCDef.h"
#include "AliGPUCAQA.h"
#include "AliGPUCADisplayBackend.h"
#include "ClusterNativeAccessExt.h"
#include "genEvents.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <tuple>
#include <algorithm>
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <fenv.h>
#include <clocale>
#include <sys/stat.h>
#endif
#include "cmodules/timer.h"

#include "AliGPUTPCGMMergedTrack.h"
#include "Interface/outputtrack.h"
#include "AliGPUCASettings.h"
#include <vector>
#include <xmmintrin.h>

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterHardware.h"
#endif

#ifdef BUILD_EVENT_DISPLAY
#ifdef _WIN32
#include "AliGPUCADisplayBackendWindows.h"
#else
#include "AliGPUCADisplayBackendX11.h"
#include "AliGPUCADisplayBackendGlfw.h"
#endif
#include "AliGPUCADisplayBackendGlut.h"
#endif

//#define BROKEN_EVENTS

AliGPUReconstruction* rec;
std::unique_ptr<char[]> outputmemory;
std::unique_ptr<AliGPUCADisplayBackend> eventDisplay;
std::unique_ptr<AliGPUReconstructionTimeframe> tf;
int nEventsInDirectory = 0;

void SetCPUAndOSSettings()
{
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
}

int ReadConfiguration(int argc, char** argv)
{
	int qcRet = qConfigParse(argc, (const char**) argv);
	if (qcRet)
	{
		if (qcRet != qConfig::qcrHelp) printf("Error parsing command line parameters\n");
		return(1);
	}
	if (configStandalone.printSettings) qConfigPrint();
#ifndef _WIN32
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
	return(0);
}

int SetupReconstruction()
{
	if (!configStandalone.eventGenerator)
	{
		char filename[256];
		snprintf(filename, 256, "events/%s/", configStandalone.EventsDir);
		rec->ReadSettings(filename);
		printf("Read event settings from dir %s (solenoidBz: %f, home-made events %d, constBz %d, maxTimeBin %d)\n", filename,
			rec->GetEventSettings().solenoidBz, (int) rec->GetEventSettings().homemadeEvents, (int) rec->GetEventSettings().constBz, rec->GetEventSettings().continuousMaxTimeBin);
	}
	
	AliGPUCASettingsEvent ev = rec->GetEventSettings();
	AliGPUCASettingsRec recSet;
	AliGPUCASettingsDeviceProcessing devProc;
	
	if (configStandalone.eventGenerator) ev.homemadeEvents = true;
	if (configStandalone.solenoidBz != -1e6f) ev.solenoidBz = configStandalone.solenoidBz;
	if (configStandalone.constBz) ev.constBz = true;
	if (configStandalone.cont) ev.continuousMaxTimeBin = -1;
	if (ev.continuousMaxTimeBin == 0 && (configStandalone.configTF.nMerge || configStandalone.configTF.bunchSim))
	{
		printf("Continuous mode forced\n");
		ev.continuousMaxTimeBin = -1;
	}
	if (rec->GetDeviceType() == AliGPUReconstruction::DeviceType::CPU) printf("Standalone Test Framework for CA Tracker - Using CPU\n");
	else printf("Standalone Test Framework for CA Tracker - Using GPU\n");

	recSet.SetMinTrackPt(MIN_TRACK_PT_DEFAULT);
	recSet.NWays = configStandalone.nways;
	recSet.NWaysOuter = configStandalone.nwaysouter;
	recSet.RejectMode = configStandalone.rejectMode;
	recSet.SearchWindowDZDR = configStandalone.dzdr;
	recSet.GlobalTracking = configStandalone.configRec.globalTracking;
	recSet.DisableRefitAttachment = configStandalone.configRec.disableRefitAttachment;
	if (configStandalone.referenceX < 500.) recSet.TrackReferenceX = configStandalone.referenceX;
	
	if (configStandalone.OMPThreads != -1) devProc.nThreads = configStandalone.OMPThreads;
	devProc.deviceNum = configStandalone.cudaDevice;
	devProc.debugLevel = configStandalone.DebugLevel;
	devProc.runQA = configStandalone.qa;
	if (configStandalone.eventDisplay)
	{
#ifdef BUILD_EVENT_DISPLAY
#ifdef _WIN32
		if (configStandalone.eventDisplay == 1) eventDisplay.reset(new AliGPUCADisplayBackendWindows);
#else
		if (configStandalone.eventDisplay == 1) eventDisplay.reset(new AliGPUCADisplayBackendX11);
		if (configStandalone.eventDisplay == 3) eventDisplay.reset(new AliGPUCADisplayBackendGlfw);
#endif
		else if (configStandalone.eventDisplay == 2) eventDisplay.reset(new AliGPUCADisplayBackendGlut);
#endif
		devProc.eventDisplay = eventDisplay.get();
	}
	devProc.nDeviceHelperThreads = configStandalone.helperThreads;
	devProc.globalInitMutex = configStandalone.gpuInitMutex;
	devProc.gpuDeviceOnly = configStandalone.oclGPUonly;
	devProc.memoryAllocationStrategy = configStandalone.allocationStrategy;
	if (configStandalone.configRec.runTRD != -1) rec->GetRecoSteps().setBits(AliGPUReconstruction::RecoStep::TRDTracking, configStandalone.configRec.runTRD);
	if (!configStandalone.merger) rec->GetRecoSteps().setBits(AliGPUReconstruction::RecoStep::TPCMerging, false);
	
	if (configStandalone.configProc.nStreams >= 0) devProc.nStreams = configStandalone.configProc.nStreams;
	if (configStandalone.configProc.constructorPipeline >= 0) devProc.trackletConstructorInPipeline = configStandalone.configProc.constructorPipeline;
	if (configStandalone.configProc.selectorPipeline >= 0) devProc.trackletSelectorInPipeline = configStandalone.configProc.selectorPipeline;
	
	rec->SetSettings(&ev, &recSet, &devProc);
	if (rec->Init())
	{
		printf("Error initializing AliGPUReconstruction!\n");
		return 1;
	}
	return(0);
}

int ReadEvent(int n)
{
	char filename[256];
	snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, n);
	int r = rec->ReadData(filename);
	if (r) return r;
	if (rec->mIOPtrs.clustersNative) rec->ConvertNativeToClusterData();
	return 0;
}

int main(int argc, char** argv)
{
	std::unique_ptr<AliGPUReconstruction> recUnique;

	SetCPUAndOSSettings();

	if (ReadConfiguration(argc, argv)) return(1);

	recUnique.reset(AliGPUReconstruction::CreateInstance(configStandalone.runGPU ? configStandalone.gpuType : AliGPUReconstruction::DEVICE_TYPE_NAMES[AliGPUReconstruction::DeviceType::CPU], configStandalone.runGPUforce));
	rec = recUnique.get();
	if (rec == nullptr)
	{
		printf("Error initializing AliGPUReconstruction\n");
		return(1);
	}

	if (SetupReconstruction()) return(1);

	//hlt.SetRunMerger(configStandalone.merger); //TODO!

	if (configStandalone.seed == -1)
	{
		std::random_device rd;
		configStandalone.seed = (int) rd();
		printf("Using random seed %d\n", configStandalone.seed);
	}

	srand(configStandalone.seed);
	
	for (nEventsInDirectory = 0;true;nEventsInDirectory++)
	{
		std::ifstream in;
		char filename[256];
		snprintf(filename, 256, "events/%s/" GPUCA_EVDUMP_FILE ".%d.dump", configStandalone.EventsDir, nEventsInDirectory);
		in.open(filename, std::ifstream::binary);
		if (in.fail()) break;
		in.close();
	}
	
	if (configStandalone.configTF.bunchSim || configStandalone.configTF.nMerge)
	{
		tf.reset(new AliGPUReconstructionTimeframe(rec, ReadEvent, nEventsInDirectory));
	}

	if (configStandalone.eventGenerator)
	{
		genEvents::RunEventGenerator(rec);
		return(1);
	}
	else
	{
		int nEvents = configStandalone.NEvents;
		if (configStandalone.configTF.bunchSim)
		{
			nEvents = configStandalone.NEvents > 0 ? configStandalone.NEvents : 1;
		}
		else
		{
			if (nEvents == -1 || nEvents > nEventsInDirectory)
			{
				if (nEvents >= 0) printf("Only %d events available in directors %s (%d events requested)\n", nEventsInDirectory, configStandalone.EventsDir, nEvents);
				nEvents = nEventsInDirectory;
			}
			nEvents /= (configStandalone.configTF.nMerge > 1 ? configStandalone.configTF.nMerge : 1);
		}

		for (int j2 = 0;j2 < configStandalone.runs2;j2++)
		{
			if (configStandalone.configQA.inputHistogramsOnly) break;
			if (configStandalone.runs2 > 1) printf("RUN2: %d\n", j2);
			long long int nTracksTotal = 0;
			long long int nClustersTotal = 0;
			int nEventsProcessed = 0;

			for (int iEvent = configStandalone.StartEvent;iEvent < nEvents;iEvent++)
			{
				if (iEvent != configStandalone.StartEvent) printf("\n");
				HighResTimer timerLoad;
				timerLoad.Start();
				if (configStandalone.configTF.bunchSim)
				{
					if (tf->LoadCreateTimeFrame(iEvent)) break;
				}
				else if (configStandalone.configTF.nMerge)
				{
					if (tf->LoadMergedEvents(iEvent)) break;
				}
				else
				{
					if (ReadEvent(iEvent)) break;
				}
				printf("Loading time: %'d us\n", (int) (1000000 * timerLoad.GetCurrentElapsedTime()));

				printf("Processing Event %d\n", iEvent);
				for (int j1 = 0;j1 < configStandalone.runs;j1++)
				{
					if (configStandalone.runs > 1) printf("Run %d\n", j1 + 1);
					if (configStandalone.outputcontrolmem) rec->SetOutputControl(outputmemory.get(), configStandalone.outputcontrolmem);
					rec->SetResetTimers(j1 <= configStandalone.runsInit);
					
					int tmpRetVal = rec->RunStandalone();
					
					if (tmpRetVal == 0)
					{
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
						if (j1 == 0)
						{
							nTracksTotal += nTracks;
							nClustersTotal += nClusters;
							nEventsProcessed++;
						}
					}
					
					if (rec->GetRecoSteps() & AliGPUReconstruction::RecoStep::TRDTracking)
					{
						int nTracklets = 0;
						for (int k = 0;k < rec->GetTRDTracker()->NTracks();k++)
						{
							auto& trk = rec->GetTRDTracker()->Tracks()[k];
							nTracklets += trk.GetNtracklets();
						}
						printf("TRD Tracker reconstructed %d tracks (%d tracklets)\n", rec->GetTRDTracker()->NTracks(), nTracklets);
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
#ifndef _WIN32
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
