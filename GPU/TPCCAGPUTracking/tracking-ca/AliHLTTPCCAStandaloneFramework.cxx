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


#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCGeometry.h"
#include "TStopwatch.h"

#ifdef HLTCA_STANDALONE
#include <omp.h>
#include "include.h"
#ifdef R__WIN32
#include <conio.h>
#else
#include <pthread.h>
#include <unistd.h>
#include "../cmodules/linux_helpers.h"
#endif
#endif

AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::Instance(int allowGPU, const char* GPULibrary)
{
  // reference to static object
  static AliHLTTPCCAStandaloneFramework gAliHLTTPCCAStandaloneFramework(allowGPU, GPULibrary);
  return gAliHLTTPCCAStandaloneFramework;
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework(int allowGPU, const char* GPULibrary)
: fMerger(), fClusterData(fInternalClusterData), fOutputControl(),
  fTracker(allowGPU, GPULibrary ? GPULibrary : 
#ifdef HLTCA_STANDALONE
    getenv("HLTCA_GPUTRACKER_LIBRARY")
#else
    NULL
#endif
  ), fStatNEvents( 0 ), fDebugLevel(0), fEventDisplay(0), fRunQA(0), fRunMerger(1), fMCLabels(0), fMCInfo(0)
{
  //* constructor

  for ( int i = 0; i < 20; i++ ) {
    fLastTime[i] = 0;
    fStatTime[i] = 0;
  }
  for ( int i = 0;i < fgkNSlices;i++) fSliceOutput[i] = NULL;
  fTracker.SetOutputControl(&fOutputControl);
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& )
    : fMerger(), fClusterData(fInternalClusterData), fOutputControl(), fTracker(), fStatNEvents( 0 ), fDebugLevel(0), fEventDisplay(0), fRunQA(0), fRunMerger(1), fMCLabels(0), fMCInfo(0)
{
  //* dummy
  for ( int i = 0; i < 20; i++ ) {
    fLastTime[i] = 0;
    fStatTime[i] = 0;
  }
  for ( int i = 0;i < fgkNSlices;i++) fSliceOutput[i] = NULL;
}

const AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::operator=( const AliHLTTPCCAStandaloneFramework& ) const
{
  //* dummy
  return *this;
}

AliHLTTPCCAStandaloneFramework::~AliHLTTPCCAStandaloneFramework()
{
#ifndef HLTCA_STANDALONE
	for (int i = 0;i < fgkNSlices;i++) if (fSliceOutput[i]) free(fSliceOutput[i]);
#endif
  //* destructor
}


void AliHLTTPCCAStandaloneFramework::StartDataReading( int guessForNumberOfClusters )
{
  //prepare for reading of the event

  int sliceGuess = 2 * guessForNumberOfClusters / fgkNSlices;

  for ( int i = 0; i < fgkNSlices; i++ ) {
    fClusterData[i].StartReading( i, sliceGuess );
  }
  fMCLabels.clear();
  fMCInfo.clear();
}

void AliHLTTPCCAStandaloneFramework::FinishDataReading()
{
  // finish reading of the event

  /*static int event_number = 0;
  char filename[256];

  sprintf(filename, "events/event.%d.dump", event_number);
  printf("Dumping event into file %s\n", filename);
  std::ofstream outfile(filename, std::ofstream::binary);
  if (outfile.fail())
  {
    printf("Error opening event dump file\n");
    exit(1);
  }
  WriteEvent(outfile);
  if (outfile.fail())
  {
    printf("Error writing event dump file\n");
    exit(1);
  }
  outfile.close();

  event_number++;*/
  
  /*std::ifstream infile(filename, std::ifstream::binary);
  ReadEvent(infile);
  infile.close();*/

  for ( int i = 0; i < fgkNSlices; i++ ) {
    //fClusterData[i].FinishReading();			//No longer needed
  }
}


int AliHLTTPCCAStandaloneFramework::ProcessEvent(int forceSingleSlice)
{
  // perform the event reconstruction

  fStatNEvents++;

  TStopwatch timer0;
  TStopwatch timer1;

#ifdef HLTCA_STANDALONE
  static HighResTimer timerTracking, timerMerger;
  static int nCount = 0;
  timerTracking.Start();

  if (fEventDisplay)
  {
	fTracker.SetKeepData(1);
  }
#endif

  if (forceSingleSlice != -1)
  {
	if (fTracker.ProcessSlices(forceSingleSlice, 1, &fClusterData[forceSingleSlice], &fSliceOutput[forceSingleSlice])) return (1);
  }
  else
  {
	for (int iSlice = 0;iSlice < fgkNSlices;iSlice += fTracker.MaxSliceCount())
	{
		if (fTracker.ProcessSlices(iSlice, CAMath::Min(fTracker.MaxSliceCount(), fgkNSlices - iSlice), &fClusterData[iSlice], &fSliceOutput[iSlice])) return (1);
	}
  }

#ifdef HLTCA_STANDALONE
  timerTracking.Stop();
#endif

  if (fRunMerger)
  {
#ifdef HLTCA_STANDALONE
      timerMerger.Start();
#endif
	  fMerger.Clear();

	  for ( int i = 0; i < fgkNSlices; i++ ) {
		//printf("slice %d clusters %d tracks %d\n", i, fClusterData[i].NumberOfClusters(), fSliceOutput[i]->NTracks());
		fMerger.SetSliceData( i, fSliceOutput[i] );
	  }

#ifdef HLTCA_GPU_MERGER
	  if (fTracker.GetGPUTracker()->GPUMergerAvailable()) fMerger.SetGPUTracker(fTracker.GetGPUTracker());
#endif
	  fMerger.Reconstruct();
#ifdef HLTCA_STANDALONE
      timerMerger.Stop();
#endif
  }

#ifdef HLTCA_STANDALONE
#ifdef BUILD_QA
  if (fRunQA)
  {
    RunQA();
  }
#endif
#ifdef BUILD_EVENT_DISPLAY
  if (fEventDisplay)
  {
    static int displayActive = 0;
	if (!displayActive)
	{
#ifdef R__WIN32
		semLockDisplay = CreateSemaphore(0, 1, 1, 0);
		HANDLE hThread;
		if ((hThread = CreateThread(NULL, NULL, &OpenGLMain, NULL, NULL, NULL)) == NULL)
#else
		static pthread_t hThread;
		if (pthread_create(&hThread, NULL, OpenGLMain, NULL))
#endif
		{
			printf("Coult not Create GL Thread...\nExiting...\n");
		}
		displayActive = 1;
	}
	else
	{
#ifdef R__WIN32
		ReleaseSemaphore(semLockDisplay, 1, NULL);
#else
		pthread_mutex_unlock(&semLockDisplay);
#endif
		ShowNextEvent();
	}

	while (kbhit()) getch();
	printf("Press key for next event!\n");

	int iKey;
	do
	{
#ifdef R__WIN32
		Sleep(10);
#else
		usleep(10000);
#endif
		iKey = kbhit() ? getch() : 0;
		if (iKey == 'q') buttonPressed = 2;
        else if (iKey == 'n') break;
        else if (iKey)
        {
            while (sendKey != 0)
            {
                #ifdef R__WIN32
                		Sleep(1);
                #else
                		usleep(1000);
                #endif                
            }
            sendKey = iKey;
        }
	} while (buttonPressed == 0);
	if (buttonPressed == 2) return(2);
	buttonPressed = 0;
	printf("Loading next event\n");

#ifdef R__WIN32
	WaitForSingleObject(semLockDisplay, INFINITE);
#else
	pthread_mutex_lock(&semLockDisplay);
#endif

	displayEventNr++;
  }
#endif

  nCount++;
  printf("Tracking Time: %1.0f us\n", 1000000 * timerTracking.GetElapsedTime() / nCount);

  if (fDebugLevel >= 1)
  {
		const char* tmpNames[10] = {"Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Tracklet Constructor", "Tracklet Selector", "Global Tracking", "Write Output"};

		for (int i = 0;i < 10;i++)
		{
            double time = 0;
			for ( int iSlice = 0; iSlice < fgkNSlices;iSlice++)
			{
				if (forceSingleSlice != -1) iSlice = forceSingleSlice;
				time += fTracker.GetTimer(iSlice, i);
                if (!HLTCA_TIMING_SUM) fTracker.ResetTimer(iSlice, i);
				if (forceSingleSlice != -1) break;
			}
			if (forceSingleSlice == -1)
			{
				time /= fgkNSlices;
			}
			if (fTracker.GetGPUStatus() < 2) time /= omp_get_max_threads();

			printf("Execution Time: Task: %20s ", tmpNames[i]);
			printf("Time: %1.0f us", time * 1000000 / nCount);
			printf("\n");
		}
		printf("Execution Time: Task: %20s Time: %1.0f us\n", "Merger", timerMerger.GetElapsedTime() * 1000000. / nCount);
        if (!HLTCA_TIMING_SUM)
        {
            timerTracking.Reset();
            timerMerger.Reset();
            nCount = 0;
        }
  }
#endif

  for ( int i = 0; i < 3; i++ ) fStatTime[i] += fLastTime[i];

  return(0);
}


void AliHLTTPCCAStandaloneFramework::SetSettings( )
{
	float solenoidBz = -5.00668;
	
	for (int slice = 0;slice < fgkNSlices;slice++)
    {
      int iSec = slice;
      float inRmin = 83.65;
      float outRmax = 247.7;
      float plusZmin = 0.0529937;
      float plusZmax = 249.778;
      float minusZmin = -249.645;
      float minusZmax = -0.0799937;
      float dalpha = 0.349066;
      float alpha = 0.174533 + dalpha * iSec;

      bool zPlus = ( iSec < 18 );
      float zMin =  zPlus ? plusZmin : minusZmin;
      float zMax =  zPlus ? plusZmax : minusZmax;
      int nRows = HLTCA_ROW_COUNT;

      float padPitch = 0.4;
      float sigmaZ = 0.228808;

      float *rowX = new float [nRows];
      for ( int irow = 0; irow < nRows; irow++ ) {
        rowX[irow] = AliHLTTPCGeometry::Row2X( irow );
      }

      AliHLTTPCCAParam param;

      param.Initialize( iSec, nRows, rowX, alpha, dalpha,
        inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, solenoidBz );
      param.SetHitPickUpFactor( 2 );
      param.SetMinNTrackClusters( -1 );
      param.SetMinTrackPt( 0.015 );

      param.Update();
      fTracker.InitializeSliceParam( slice, param );
      delete[] rowX;
    }

	{
	  AliHLTTPCCAParam param;
	  // get gemetry
	  int iSec = 0;
	  float inRmin = 83.65;
	  float outRmax = 247.7;
	  float plusZmin = 0.0529937;
	  float plusZmax = 249.778;
	  float dalpha = 0.349066;
	  float alpha = 0.174533 + dalpha * iSec;
	  float zMin =  plusZmin;
	  float zMax =  plusZmax;
	  int nRows = HLTCA_ROW_COUNT;
	  float padPitch = 0.4;
	  float sigmaZ = 0.228808;
	  float *rowX = new float [nRows];
	  for ( int irow = 0; irow < nRows; irow++ ) {
		rowX[irow] = AliHLTTPCGeometry::Row2X( irow );
	  }

	  param.Initialize( iSec, nRows, rowX, alpha, dalpha,
						inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, solenoidBz );

	  param.SetClusterError2CorrectionZ( 1.1 );
	  param.Update();

	  delete[] rowX;
	  
	  fMerger.SetSliceParam(param);
	}

}

void AliHLTTPCCAStandaloneFramework::WriteEvent( std::ostream &out ) const
{
  // write event to the file
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    fClusterData[iSlice].WriteEvent( out );
  }
}

void AliHLTTPCCAStandaloneFramework::ReadEvent( std::istream &in, bool resetIds, bool addData, float shift )
{
  //* Read event from file
  int nClusters = 0, nCurrentClusters = 0;
  if (addData) for (int i = 0;i < fgkNSlices;i++) nCurrentClusters += fClusterData[i].NumberOfClusters();
  int nCurrentMCTracks = addData ? fMCInfo.size() : 0;
  
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    const int nSliceOldClusters = addData ? fClusterData[iSlice].NumberOfClusters() : 0;
    fClusterData[iSlice].ReadEvent( in, addData );
    if (resetIds)
    {
      for (int i = 0;i < fClusterData[iSlice].NumberOfClusters() - nSliceOldClusters;i++)
      {
        fClusterData[iSlice].Clusters()[nSliceOldClusters + i].fId = nCurrentClusters + nClusters + i;
      }
    }
    if (shift != 0.)
    {
      for (int i = 0;i < fClusterData[iSlice].NumberOfClusters() - nSliceOldClusters;i++)
      {
        AliHLTTPCCAClusterData::Data& tmp = fClusterData[iSlice].Clusters()[nSliceOldClusters + i];
        tmp.fZ += tmp.fZ > 0 ? shift : -shift;
      }
    }
    nClusters += fClusterData[iSlice].NumberOfClusters() - nSliceOldClusters;
  }
  fMCLabels.resize(nCurrentClusters + nClusters);
  in.read((char*) (fMCLabels.data() + nCurrentClusters), nClusters * sizeof(fMCLabels[0]));
  if (!in || in.gcount() != nClusters * (int) sizeof(fMCLabels[0]))
  {
    fMCLabels.clear();
    fMCInfo.clear();
  }
  else
  {
    if (addData)
    {
        for (int i = 0;i < nClusters;i++)
        {
            for (int j = 0;j < 3;j++)
            {
                AliHLTTPCClusterMCWeight& tmp = fMCLabels[nCurrentClusters + i].fClusterID[j];
                if (tmp.fMCID >= 0) tmp.fMCID += nCurrentMCTracks;
            }
        }
    }
    int nMCTracks = 0;
    in.read((char*) &nMCTracks, sizeof(nMCTracks));
    if (in.eof())
    {
      fMCInfo.clear();
    }
    else
    {
      fMCInfo.resize(nCurrentMCTracks + nMCTracks);
      in.read((char*) (fMCInfo.data() + nCurrentMCTracks), nMCTracks * sizeof(fMCInfo[0]));
      if (in.eof())
      {
          fMCInfo.clear();
      }
      else if (shift != 0.)
      {
          for (int i = 0;i < nMCTracks;i++)
          {
              AliHLTTPCCAMCInfo& tmp = fMCInfo[nCurrentMCTracks + i];
              tmp.fZ += tmp.fZ > 0 ? shift : -shift;
          }
      }
    }
  }
#ifdef HLTCA_STANDALONE
  printf("Read %d Clusters with %d MC labels and %d MC tracks\n", nClusters, (int) fMCLabels.size() - nCurrentClusters, (int) fMCInfo.size() - nCurrentMCTracks);
  if (addData) printf("Total %d Clusters with %d MC labels and %d MC tracks\n", nClusters + nCurrentClusters, (int) fMCLabels.size(), (int) fMCInfo.size());
#endif
  nClusters += nCurrentClusters;
}

void AliHLTTPCCAStandaloneFramework::WriteTracks( std::ostream &out ) const
{
  //* Write tracks to file

  for ( int i = 0; i < 20; i++ ) out << fLastTime[i] << std::endl;
  //fMerger.Output()->Write( out );
}

void AliHLTTPCCAStandaloneFramework::ReadTracks( std::istream &in )
{
  //* Read tracks  from file

  for ( int i = 0; i < 20; i++ ) {
    in >> fLastTime[i];
    fStatTime[i] += fLastTime[i];
  }
  //fMerger.Output()->Read( in );
}
