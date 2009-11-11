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
#include "AliHLTTPCCAMerger.h"
#include "AliHLTTPCCAMergerOutput.h"
#include "AliHLTTPCCADataCompressor.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAClusterData.h"
#include "TStopwatch.h"

#ifdef HLTCA_STANDALONE
#include <omp.h>
#include "include.h"
#ifdef R__WIN32
#include <conio.h>
#else
#include <pthread.h>
#endif
#endif

AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::Instance()
{
  // reference to static object
  static AliHLTTPCCAStandaloneFramework gAliHLTTPCCAStandaloneFramework;
  return gAliHLTTPCCAStandaloneFramework;
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework()
    : fMerger(), fOutputControl(), fTracker(), fStatNEvents( 0 ), fDebugLevel(0)
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
    : fMerger(), fOutputControl(), fTracker(), fStatNEvents( 0 ), fDebugLevel(0)
{
  //* dummy
}

const AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::operator=( const AliHLTTPCCAStandaloneFramework& ) const
{
  //* dummy
  return *this;
}

AliHLTTPCCAStandaloneFramework::~AliHLTTPCCAStandaloneFramework()
{
	for (int i = 0;i < fgkNSlices;i++) if (fSliceOutput[i]) free(fSliceOutput[i]);
  //* destructor
}


void AliHLTTPCCAStandaloneFramework::StartDataReading( int guessForNumberOfClusters )
{
  //prepare for reading of the event

  int sliceGuess = 2 * guessForNumberOfClusters / fgkNSlices;

  for ( int i = 0; i < fgkNSlices; i++ ) {
    fClusterData[i].StartReading( i, sliceGuess );
  }
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

  sprintf(filename, "events/settings.%d.dump", event_number);
  outfile.open(filename);
  WriteSettings(outfile);
  outfile.close();

  event_number++;*/
  
  /*std::ifstream infile(filename, std::ifstream::binary);
  ReadEvent(infile);
  infile.close();*/

  for ( int i = 0; i < fgkNSlices; i++ ) {
    fClusterData[i].FinishReading();
  }
}


//int
void AliHLTTPCCAStandaloneFramework::ProcessEvent(int forceSingleSlice)
{
  // perform the event reconstruction

  fStatNEvents++;

  TStopwatch timer0;
  TStopwatch timer1;

#ifdef HLTCA_STANDALONE
  unsigned long long int startTime, endTime, checkTime;
  unsigned long long int cpuTimers[16], gpuTimers[16], tmpFreq;
  StandaloneQueryFreq(&tmpFreq);
  StandaloneQueryTime(&startTime);

  fOutputControl.fObsoleteOutput = 0;
  fTracker.SetKeepData(1);
#endif

  if (forceSingleSlice != -1)
  {
	if (fTracker.ProcessSlices(forceSingleSlice, 1, &fClusterData[forceSingleSlice], &fSliceOutput[forceSingleSlice])) return;
  }
  else
  {
	for (int iSlice = 0;iSlice < fgkNSlices;iSlice += fTracker.MaxSliceCount())
	{
		if (fTracker.ProcessSlices(iSlice, fTracker.MaxSliceCount(), &fClusterData[iSlice], &fSliceOutput[iSlice])) return;
	}
  }

#ifdef HLTCA_STANDALONE
  StandaloneQueryTime(&endTime);
  StandaloneQueryTime(&checkTime);
#endif

  timer1.Stop();
  TStopwatch timer2;

  fMerger.Clear();
  fMerger.SetSliceParam( fTracker.Param(0) );

  for ( int i = 0; i < fgkNSlices; i++ ) {
	//printf("slice %d clusters %d tracks %d\n", i, fClusterData[i].NumberOfClusters(), fSliceOutput[i]->NTracks());
    fMerger.SetSliceData( i, fSliceOutput[i] );
  }

  fMerger.Reconstruct();

  timer2.Stop();
  timer0.Stop();

  fLastTime[0] = timer0.CpuTime();
  fLastTime[1] = timer1.CpuTime();
  fLastTime[2] = timer2.CpuTime();

#ifdef HLTCA_STANDALONE
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
	}

#ifdef R__WIN32
	while (kbhit()) getch();
#endif
	printf("Press key for next event!\n");

	int iKey;
	do
	{
#ifdef R__WIN32
		Sleep(10);
		iKey = kbhit() ? getch() : 0;
#else
		iKey = getchar();
#endif
		if (iKey == 'q') exit(0);
	} while (iKey != 'n' && buttonPressed == 0);
	buttonPressed = 0;
	printf("Loading next event\n");

#ifdef R__WIN32
	WaitForSingleObject(semLockDisplay, INFINITE);
#else
	pthread_mutex_lock(&semLockDisplay);
#endif

	displayEventNr++;

  printf("Tracking Time: %lld us\nTime uncertainty: %lld ns\n", (endTime - startTime) * 1000000 / tmpFreq, (checkTime - endTime) * 1000000000 / tmpFreq);

  if (fDebugLevel >= 1)
  {
		const char* tmpNames[16] = {"Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Reserved", "Tracklet Constructor", "Tracklet Selector", "Write Output", "Unused", "Unused", "Unused", "Unused", "Unused", "Unused"};

		for (int i = 0;i < 10;i++)
		{
			if (i == 6) continue;
			cpuTimers[i] = gpuTimers[i] = 0;
			for ( int iSlice = 0; iSlice < fgkNSlices;iSlice++)
			{
				if (forceSingleSlice != -1) iSlice = forceSingleSlice;
				cpuTimers[i] += *fTracker.PerfTimer(0, iSlice, i + 1) - *fTracker.PerfTimer(0, iSlice, i);
				if (forceSingleSlice != -1 || (fTracker.MaxSliceCount() && (iSlice % fTracker.MaxSliceCount() == 0 || i <= 5)))
					gpuTimers[i] += *fTracker.PerfTimer(1, iSlice, i + 1) - *fTracker.PerfTimer(1, iSlice, i);
				if (forceSingleSlice != -1) break;
			}
			if (forceSingleSlice == -1)
			{
				cpuTimers[i] /= fgkNSlices;
				gpuTimers[i] /= fgkNSlices;
			}
			cpuTimers[i] *= 1000000;
			gpuTimers[i] *= 1000000;
			cpuTimers[i] /= tmpFreq;
			gpuTimers[i] /= tmpFreq;
			cpuTimers[i] /= omp_get_max_threads();

			printf("Execution Time: Task: %20s ", tmpNames[i]);
				printf("CPU: %15lld\t\t", cpuTimers[i]);
				printf("GPU: %15lld\t\t", gpuTimers[i]);
			if (fDebugLevel >=6 && gpuTimers[i])
				printf("Speedup: %4lld%%", cpuTimers[i] * 100 / gpuTimers[i]);
			printf("\n");
		}
		printf("Execution Time: Task: %20s CPU: %15lld\n", "Merger", (long long int) (timer2.CpuTime() * 1000000));
  }
#endif

  for ( int i = 0; i < 3; i++ ) fStatTime[i] += fLastTime[i];

  //return(0);
}


void AliHLTTPCCAStandaloneFramework::WriteSettings( std::ostream &out ) const
{
  //* write settings to the file
  out << NSlices() << std::endl;
  for ( int iSlice = 0; iSlice < NSlices(); iSlice++ ) {
    fTracker.Param(iSlice).WriteSettings( out );
  }
}

void AliHLTTPCCAStandaloneFramework::ReadSettings( std::istream &in )
{
  //* Read settings from the file
  int nSlices = 0;
  in >> nSlices;
  for ( int iSlice = 0; iSlice < nSlices; iSlice++ ) {
    AliHLTTPCCAParam param;
    param.ReadSettings ( in );
	fTracker.InitializeSliceParam(iSlice, param);
  }
}

void AliHLTTPCCAStandaloneFramework::WriteEvent( std::ostream &out ) const
{
  // write event to the file
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    fClusterData[iSlice].WriteEvent( out );
  }
}

void AliHLTTPCCAStandaloneFramework::ReadEvent( std::istream &in )
{
  //* Read event from file
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    fClusterData[iSlice].ReadEvent( in );
  }
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
