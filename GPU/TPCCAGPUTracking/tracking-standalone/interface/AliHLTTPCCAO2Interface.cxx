// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliHLTTPCCAO2Interface.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "standaloneSettings.h"
#include <iostream>
#include <fstream>

AliHLTTPCCAO2Interface::AliHLTTPCCAO2Interface() : fInitialized(false), fDumpEvents(false), fContinuous(false), fHLT(NULL)
{
}

AliHLTTPCCAO2Interface::~AliHLTTPCCAO2Interface()
{
	Deinitialize();
}

int AliHLTTPCCAO2Interface::Initialize(const char* options)
{
	if (fInitialized) return(1);
	fHLT = &AliHLTTPCCAStandaloneFramework::Instance(-1);
	if (fHLT == NULL) return(1);
	float solenoidBz = -5.00668;
	float refX = 1000.;

	if (options && *options)
	{
		printf("Received options %s\n", options);
		const char* optPtr = options;
		while (optPtr && *optPtr)
		{
			while (*optPtr == ' ') optPtr++;
			const char* nextPtr = strstr(optPtr, " ");
			const int optLen = nextPtr ? nextPtr - optPtr : strlen(optPtr);
			if (strncmp(optPtr, "cont", optLen) == 0)
			{
				fContinuous = true;
				printf("Continuous tracking mode enabled\n");
			}
			else if (strncmp(optPtr, "dump", optLen) == 0)
			{
				fDumpEvents = true;
				printf("Dumping of input events enabled\n");
			}
			else if (optLen > 3 && strncmp(optPtr, "bz=", 3) == 0)
			{
				sscanf(optPtr + 3, "%f", &solenoidBz);
				printf("Using solenoid field %f\n", solenoidBz);
			}
			else if (optLen > 5 && strncmp(optPtr, "refX=", 5) == 0)
			{
				sscanf(optPtr + 5, "%f", &refX);
				printf("Propagating to reference X %f\n", refX);
			}
			else
			{
				printf("Unknown option: %s\n", optPtr);
			}
			optPtr = nextPtr;
		}
	}

	fHLT->SetNWays(3);
	fHLT->SetSettings(solenoidBz, false, false);
	fHLT->SetGPUTrackerOption("HelperThreads", 0);
	fHLT->SetGPUTrackerOption("GlobalTracking", 1);
	fHLT->SetSearchWindowDZDR(2.5f);
	fHLT->SetContinuousTracking(fContinuous);
	fHLT->SetTrackReferenceX(refX);
	fHLT->UpdateGPUSliceParam();

	fInitialized = true;
	return(0);
}

void AliHLTTPCCAO2Interface::Deinitialize()
{
	if (fInitialized)
	{
		fHLT->Merger().Clear();
		fHLT->Merger().SetGPUTracker(NULL);
		fHLT->ExitGPU();
		fHLT = NULL;
	}
	fInitialized = false;
}

int AliHLTTPCCAO2Interface::RunTracking(const AliHLTTPCCAClusterData* inputClusters, const AliHLTTPCGMMergedTrack* &outputTracks, int &nOutputTracks, const AliHLTTPCGMMergedTrackHit* &outputTrackClusters)
{
	if (!fInitialized) return(1);
	static int nEvent = 0;
	fHLT->SetExternalClusterData((AliHLTTPCCAClusterData*) inputClusters);
	if (fDumpEvents)
	{
		std::ofstream out;
		char fname[1024];
		sprintf(fname, "event.%d.dump", nEvent);
		out.open(fname, std::ofstream::binary);
		fHLT->WriteEvent(out);
		out.close();
		if (nEvent == 0)
		{
			out.open("settings.dump", std::ofstream::binary);
			hltca_event_dump_settings settings;
			settings.setDefaults();
			settings.solenoidBz = fHLT->Param().BzkG();
			out.write((char*) &settings, sizeof(settings));
			out.close();
		}
	}
	fHLT->ProcessEvent();
	outputTracks = fHLT->Merger().OutputTracks();
	nOutputTracks = fHLT->Merger().NOutputTracks();
	outputTrackClusters = fHLT->Merger().Clusters();
	nEvent++;
	return(0);
}

int AliHLTTPCCAO2Interface::RunTracking(const AliHLTTPCCAClusterData* inputClusters, const AliHLTTPCGMMergedTrack* &outputTracks, int &nOutputTracks, const unsigned int* &outputTrackClusterIDs)
{
	const AliHLTTPCGMMergedTrackHit* outputTrackClusters;
	int retVal = RunTracking(inputClusters, outputTracks, nOutputTracks, outputTrackClusters);
	if (retVal) return(retVal);
	fOutputTrackClusterBuffer.resize(fHLT->Merger().NOutputTrackClusters());
	for (int i = 0;i < fHLT->Merger().NOutputTrackClusters();i++)
	{
		fOutputTrackClusterBuffer[i] = outputTrackClusters[i].fId;
	}
	outputTrackClusterIDs = fOutputTrackClusterBuffer.data();
	return(0);
}

void AliHLTTPCCAO2Interface::Cleanup()
{
	
}
