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


AliHLTTPCCAStandaloneFramework &AliHLTTPCCAStandaloneFramework::Instance()
{
  // reference to static object
  static AliHLTTPCCAStandaloneFramework gAliHLTTPCCAStandaloneFramework;
  return gAliHLTTPCCAStandaloneFramework;
}
  
AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework()
  : fMerger(), fStatNEvents( 0 )
{
  //* constructor

  for ( int i = 0; i < 20; i++ ){
    fLastTime[i] = 0;
    fStatTime[i] = 0;
  }
}

AliHLTTPCCAStandaloneFramework::AliHLTTPCCAStandaloneFramework( const AliHLTTPCCAStandaloneFramework& )
  : fMerger(), fStatNEvents( 0 )
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

  int sliceGuess = 2*guessForNumberOfClusters/fgkNSlices;

  for ( int i = 0; i < fgkNSlices; i++ ){
    fClusterData[i].StartReading( i, sliceGuess );
  }
}

void AliHLTTPCCAStandaloneFramework::FinishDataReading()
{
  // finish reading of the event

  for ( int i = 0; i < fgkNSlices; i++ ){
    fClusterData[i].FinishReading();
  }
}


void AliHLTTPCCAStandaloneFramework::ProcessEvent()
{
  // perform the event reconstruction

  fStatNEvents++;

  TStopwatch timer0;
  TStopwatch timer1;

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {     
    fSliceTrackers[iSlice].ReadEvent( &( fClusterData[iSlice] ) );    
    fSliceTrackers[iSlice].Reconstruct();
  }

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

  for( int i=0; i<3; i++ ) fStatTime[i]+=fLastTime[i];
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

void AliHLTTPCCAStandaloneFramework::WriteEvent( std::ostream &/*out*/ ) const
{
  // write event to the file
  for( int iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    //fClusterData[i].WriteEvent( out );
  }
}

void AliHLTTPCCAStandaloneFramework::ReadEvent( std::istream &/*in*/ ) const
{
  //* Read event from file

  for( int iSlice=0; iSlice<fgkNSlices; iSlice++ ){
    //fClusterData[i].ReadEvent( in );
  }
}

void AliHLTTPCCAStandaloneFramework::WriteTracks( std::ostream &out ) const
{
  //* Write tracks to file

  for( int i=0; i<20; i++ ) out<<fLastTime[i]<<std::endl;
  //fMerger.Output()->Write( out );
}

void AliHLTTPCCAStandaloneFramework::ReadTracks( std::istream &in )
{
  //* Read tracks  from file

  for( int i=0; i<20; i++ ){
    in>>fLastTime[i];
    fStatTime[i]+=fLastTime[i];
  }
  //fMerger.Output()->Read( in );
}
