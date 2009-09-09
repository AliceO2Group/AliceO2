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
#endif

AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::Instance()
{
  // reference to static object
  static AliHLTTPCCAStandaloneFramework gAliHLTTPCCAStandaloneFramework;
  return gAliHLTTPCCAStandaloneFramework;
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework()
    : fMerger(), fGPUTracker(), fStatNEvents( 0 ), fUseGPUTracker(false), fGPUDebugLevel(0), fGPUSliceCount(0)
{
  //* constructor

  for ( int i = 0; i < 20; i++ ) {
    fLastTime[i] = 0;
    fStatTime[i] = 0;
  }
  InitGPU(12, -1);
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& )
    : fMerger(), fGPUTracker(), fStatNEvents( 0 ), fUseGPUTracker(false), fGPUDebugLevel(0), fGPUSliceCount(0)
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
  //Do one initial run for the GPU to be initialized during benchmarks
#ifndef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
  if (fUseGPUTracker && fGPUDebugLevel < 2)
  {
	fSliceTrackers[0].ReadEvent( &( fClusterData[0] ) );
	fGPUTracker.Reconstruct(&fSliceTrackers[0], 1);
  }
#endif

  unsigned long long int sliceTimers[fgkNSlices][16], startTime, endTime, checkTime;
  unsigned long long int cpuTimers[16], gpuTimers[16], tmpFreq;
  unsigned long long int sliceDataStart, sliceDataEnd, sliceDataSumCPU = 0, sliceDataSumGPU = 0, writeoutstart, writeoutend;
  fSliceTrackers[0].StandaloneQueryFreq(&tmpFreq);
  fSliceTrackers[0].StandaloneQueryTime(&startTime);
#endif

  if (!fUseGPUTracker || fGPUDebugLevel >= 6)
  {
#ifdef HLTCA_STANDALONE
#pragma omp parallel for
#endif
	for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
	  if (forceSingleSlice == -1 || iSlice == forceSingleSlice)
	  {
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
			fSliceTrackers[0].StandaloneQueryTime(&sliceDataStart);
#endif
		fSliceTrackers[iSlice].ReadEvent( &( fClusterData[iSlice] ) );
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
		{
			fSliceTrackers[0].StandaloneQueryTime(&sliceDataEnd);
			sliceDataSumCPU += sliceDataEnd - sliceDataStart;
		}
#endif
		fSliceTrackers[iSlice].Reconstruct();
	  }
	}
	if (fGPUDebugLevel >= 2) printf("\n");
  }

  if (fUseGPUTracker)
  {
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
			fSliceTrackers[0].StandaloneQueryTime(&sliceDataStart);
#endif
#ifdef HLTCA_STANDALONE
#pragma omp parallel for
#endif
		for (int iSlice = 0;iSlice  < fgkNSlices; iSlice++)
		{
			fSliceTrackers[iSlice].ReadEvent( &( fClusterData[iSlice] ) );
		}
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
		{
			fSliceTrackers[0].StandaloneQueryTime(&sliceDataEnd);
			sliceDataSumGPU = sliceDataEnd - sliceDataStart;
		}
#endif	
	  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice += fGPUSliceCount ) {
		if (forceSingleSlice != -1) iSlice = forceSingleSlice;
		if (fGPUTracker.Reconstruct(&fSliceTrackers[iSlice], forceSingleSlice != -1 ? 1 : CAMath::Min(fGPUSliceCount, fgkNSlices - iSlice)))
		{
			printf("Error during GPU Reconstruction (Slice %d)!!!\n", iSlice);
			return;
			//return(1);
		}
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
		{
			for ( int i = 0;i < 16;i++)
				sliceTimers[iSlice][i] = *fGPUTracker.PerfTimer(i);
		}
#endif
		if (forceSingleSlice != -1) break;
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
		break;
#endif
	  }

#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
		{
			fSliceTrackers[0].StandaloneQueryTime(&writeoutstart);
		}
#pragma omp parallel for
#endif
		for (int iSlice = 0;iSlice  < fgkNSlices; iSlice++)
		{
			if (forceSingleSlice == -1 || forceSingleSlice == iSlice) fSliceTrackers[iSlice].WriteOutput();
		}
#ifdef HLTCA_STANDALONE
		if (fGPUDebugLevel >= 1)
		{
			fSliceTrackers[0].StandaloneQueryTime(&writeoutend);
		}
#endif

	  if (fGPUDebugLevel >= 2) printf("\n");
  }

#ifdef HLTCA_STANDALONE
  fSliceTrackers[0].StandaloneQueryTime(&endTime);
  fSliceTrackers[0].StandaloneQueryTime(&checkTime);
#endif

  timer1.Stop();
  TStopwatch timer2;

  fMerger.Clear();
  fMerger.SetSliceParam( fSliceTrackers[0].Param() );

  for ( int i = 0; i < fgkNSlices; i++ ) {
    fMerger.SetSliceData( i, fSliceTrackers[i].Output() );
  }

  fMerger.Reconstruct();

  timer2.Stop();
  timer0.Stop();

  fLastTime[0] = timer0.CpuTime();
  fLastTime[1] = timer1.CpuTime();
  fLastTime[2] = timer2.CpuTime();

#ifdef HLTCA_STANDALONE
  printf("Tracking Time: %lld us\nTime uncertainty: %lld ns\n", (endTime - startTime) * 1000000 / tmpFreq, (checkTime - endTime) * 1000000000 / tmpFreq);

  if (fGPUDebugLevel >= 1)
  {
		const char* tmpNames[16] = {"Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Tracklet Initializer", "Tracklet Constructor", "Tracklet Selector", "Read Out", "Write Output", "Unused", "Unused", "Unused", "Unused", "Unused"};

		for (int i = 0;i < 11;i++)
		{
			cpuTimers[i] = gpuTimers[i] = 0;
			for ( int iSlice = 0; iSlice < fgkNSlices;iSlice++)
			{
				if (forceSingleSlice != -1) iSlice = forceSingleSlice;
				cpuTimers[i] += *fSliceTrackers[iSlice].PerfTimer(i + 1) - *fSliceTrackers[iSlice].PerfTimer(i);
				if (forceSingleSlice != -1 || (fGPUSliceCount && iSlice % fGPUSliceCount == 0))
					gpuTimers[i] += sliceTimers[iSlice][i + 1] - sliceTimers[iSlice][i];
				if (forceSingleSlice != -1) break;
			}
			if (i == 10) gpuTimers[i] = writeoutend - writeoutstart;
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
			if (!fUseGPUTracker || fGPUDebugLevel >= 6)
				printf("CPU: %15lld\t\t", cpuTimers[i]);
			if (fUseGPUTracker)
				printf("GPU: %15lld\t\t", gpuTimers[i]);
			if (fGPUDebugLevel >=6 && fUseGPUTracker && gpuTimers[i])
				printf("Speedup: %4lld%%", cpuTimers[i] * 100 / gpuTimers[i]);
			printf("\n");
		}
		printf("Execution Time: Task: %20s CPU: %15lld\n", "Merger", (long long int) (timer2.CpuTime() * 1000000));
		if (forceSingleSlice == -1)
		{
			sliceDataSumCPU /= fgkNSlices;
			sliceDataSumGPU /= fgkNSlices;
		}
		sliceDataSumCPU /= omp_get_max_threads();
		printf("Execution Time: Task: %20s ", "Grid");
		if (!fUseGPUTracker || fGPUDebugLevel >= 6)
			printf("CPU: %15lld\t\t", sliceDataSumCPU * 1000000 / tmpFreq);
		if (fUseGPUTracker)
			printf("GPU: %15lld\t\t", sliceDataSumGPU * 1000000 / tmpFreq);
		printf("\n");
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
    fSliceTrackers[iSlice].Param().WriteSettings( out );
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
    fSliceTrackers[iSlice].Initialize( param );
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

int AliHLTTPCCAStandaloneFramework::InitGPU(int sliceCount, int forceDeviceID)
{
	int retVal;
	if (fUseGPUTracker && (retVal = ExitGPU())) return(retVal);
	retVal = fGPUTracker.InitGPU(sliceCount, forceDeviceID);
	fUseGPUTracker = retVal == 0;
	fGPUSliceCount = sliceCount;
	return(retVal);
}

int AliHLTTPCCAStandaloneFramework::ExitGPU()
{
	if (!fUseGPUTracker) return(0);
	fUseGPUTracker = 0;
	return(fGPUTracker.ExitGPU());
}

void AliHLTTPCCAStandaloneFramework::SetGPUDebugLevel(int Level, std::ostream *OutFile, std::ostream *GPUOutFile)
{
	fGPUTracker.SetDebugLevel(Level, GPUOutFile);
	fGPUDebugLevel = Level;
	for (int i = 0;i < fgkNSlices;i++)
	{
		fSliceTrackers[i].SetGPUDebugLevel(Level, OutFile);
	}
}
