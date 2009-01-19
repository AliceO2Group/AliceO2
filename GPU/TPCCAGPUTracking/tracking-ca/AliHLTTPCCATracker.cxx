// @(#) $Id$
//***************************************************************************
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
//***************************************************************************

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAHit.h"

#include "TStopwatch.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCAProcess.h"
#include "AliHLTTPCCALinksWriter.h"
#include "AliHLTTPCCAUsedHitsInitialiser.h"

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCATrackParam1.h"

#if !defined(HLTCA_GPUCODE)
#if defined( HLTCA_STANDALONE )
#include <iostream.h>
#else
#include "Riostream.h"
#endif
#endif

//#define DRAW

#ifdef DRAW
  #include "AliHLTTPCCADisplay.h"
  #include "TApplication.h"
#endif //DRAW

ClassImp(AliHLTTPCCATracker)

#if !defined(HLTCA_GPUCODE)  

AliHLTTPCCATracker::AliHLTTPCCATracker()
  :
  fParam(),
  fNHitsTotal(0),
  fGridSizeTotal(0),
  fGrid1SizeTotal(0),
  fHits(0),
  fHits1(0),
  fGridContents(0),
  fGrid1Contents(0),
  fHitsID(0),
  fHitLinkUp(0),
  fHitLinkDown(0),
  fHitIsUsed(0),
  fStartHits(0),  
  fTracklets(0),
  fNTracks(0),
  fTracks(0),
  fTrackHits(0),
  fNOutTracks(0),
  fNOutTrackHits(0),
  fOutTracks(0),
  fOutTrackHits(0),
  fEventMemory(0),
  fEventMemSize(0),
  fTexHitsFullData(0),
  fTexHitsFullSize(0)
{
  // constructor
  //fRows = new AliHLTTPCCARow[fParam.NRows()];
  //Initialize( fParam );
}

AliHLTTPCCATracker::AliHLTTPCCATracker( const AliHLTTPCCATracker& )
  :
  fParam(),
  fNHitsTotal(0),
  fGridSizeTotal(0),
  fGrid1SizeTotal(0),
  fHits(0),
  fHits1(0),
  fGridContents(0),
  fGrid1Contents(0),
  fHitsID(0),
  fHitLinkUp(0),
  fHitLinkDown(0),
  fHitIsUsed(0),
  fStartHits(0),  
  fTracklets(0),
  fNTracks(0),
  fTracks(0),
  fTrackHits(0),
  fNOutTracks(0),
  fNOutTrackHits(0),
  fOutTracks(0),
  fOutTrackHits(0),
  fEventMemory(0),
  fEventMemSize(0),
  fTexHitsFullData(0),
  fTexHitsFullSize(0)
{
  // dummy
}

AliHLTTPCCATracker &AliHLTTPCCATracker::operator=( const AliHLTTPCCATracker& )
{
  // dummy
  fOutTrackHits=0;
  fOutTracks=0;
  fNOutTracks=0;
  fTrackHits = 0;
  fEventMemory = 0;
  return *this;
}

GPUd() AliHLTTPCCATracker::~AliHLTTPCCATracker()
{
  // destructor
  StartEvent();
}
#endif


GPUd() UChar_t AliHLTTPCCATracker::GetGridContent(  UInt_t i ) const
{
  //* get grid content
#if defined(HLTCA_GPUSTEP)
  return (UChar_t) tex1Dfetch(texGrid,i).x;  
#else
  return fGridContents[i];
#endif
}


GPUd() AliHLTTPCCAHit AliHLTTPCCATracker::GetHit(  UInt_t i ) const
{
  //* get hit
#if defined(HLTCA_USE_GPU)
  AliHLTTPCCAHit h;
  float2 f = tex1Dfetch(texHits,i);
  h.Y() = f.x;
  h.Z() = f.y;
  return h;
#else
  return fHits[i];
#endif
}




// ----------------------------------------------------------------------------------
GPUd() void AliHLTTPCCATracker::Initialize( AliHLTTPCCAParam &param )
{
  // initialisation
  StartEvent();
  fParam = param;
  fParam.Update(); 
  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    fRows[irow].X() = fParam.RowX(irow);        
    fRows[irow].MaxY() = CAMath::Tan( fParam.DAlpha()/2.)*fRows[irow].X();    
  }
  StartEvent();
}

GPUd() void AliHLTTPCCATracker::StartEvent()
{
  // start new event and fresh the memory  

  if( fEventMemory ) delete[] fEventMemory;
  if( fOutTracks ) delete[] fOutTracks;
  if( fOutTrackHits ) delete[] fOutTrackHits;
  fEventMemory = 0;
  fHits = 0;
  fHits1 = 0;
  fHitsID = 0;
  fTrackHits = 0;
  fTracks = 0;
  fOutTrackHits = 0;
  fOutTracks = 0;
  fNTracks = 0;
  fNOutTrackHits = 0;
  fNOutTracks = 0;
  fNHitsTotal = 0;
}

GPUhd() void  AliHLTTPCCATracker::SetPointers()
{
  // set all pointers to the event memory

  fEventMemSize = 0;
  UInt_t &size = fEventMemSize;
  fHits = (AliHLTTPCCAHit*) (fEventMemory+ size);
  size+= sizeof(AliHLTTPCCAHit)*fNHitsTotal;
  fHits1 = (ushort2*) (fEventMemory+ size);
  size+= sizeof(ushort2)*fNHitsTotal;
  fGridContents = (UChar_t *) (fEventMemory + size);
  size+= sizeof(Int_t)*fGridSizeTotal;
  fGrid1Contents = (UInt_t *) (fEventMemory + size);
  size+= sizeof(UInt_t)*fGrid1SizeTotal;
  fHitsID = (Int_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * fNHitsTotal;
  fHitLinkUp = (Short_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * fNHitsTotal;
  fHitLinkDown = (Short_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * fNHitsTotal;
  fHitIsUsed = (Int_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * fNHitsTotal;
  fStartHits = (Int_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * (fNHitsTotal+1);
  size = (size/16+1)*16;
  fTexHitsFullData = (uint4*)(fEventMemory+ size);
  size+= ((sizeof(UShort_t)*6*fNHitsTotal + sizeof(UShort_t)*2*fGrid1SizeTotal )/16+1)*16;

  fTracklets = (Int_t *) (fEventMemory + size);
  size+= sizeof(Int_t) * (1 + fNHitsTotal*(5+ sizeof(AliHLTTPCCATrackParam)/4 + 160 ));
  fNTracks = (Int_t *) (fEventMemory + size);
  size+= sizeof(Int_t);
  fTracks = (AliHLTTPCCATrack* )(fEventMemory + size);
  size+= sizeof(AliHLTTPCCATrack) * (fNHitsTotal+1);
  fTrackHits = ( Int_t *)(fEventMemory + size);
  size+= sizeof(Int_t) * (10*fNHitsTotal+1);

  fOutTrackHits = 0;
  fOutTracks = 0;
}

GPUd() void AliHLTTPCCATracker::ReadEvent( Int_t *RowFirstHit, Int_t *RowNHits, Float_t *Y, Float_t *Z, Int_t NHits )
{
  //* Read event

  fNHitsTotal = NHits;
  fGridSizeTotal = 0;
  fGrid1SizeTotal = 0;
  fTexHitsFullSize = 0;

  //cout<<"event mem = "<<fEventMemory<<endl;
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ ){
    //cout<<"row, nhits="<<iRow<<" "<<RowNHits[iRow]<<endl;
    //cout<<"row, firsthit="<<iRow<<" "<<RowFirstHit[iRow]<<endl;
    AliHLTTPCCARow &row = fRows[iRow];
    row.FirstHit() = RowFirstHit[iRow];
    row.NHits() = RowNHits[iRow];
    Float_t yMin=1.e3, yMax=-1.e3, zMin=1.e3, zMax=-1.e3;
    Int_t nGrid =  row.NHits();   
    for( Int_t i=0; i<row.NHits(); i++ ){       
      Int_t j = RowFirstHit[iRow]+i;
      if( yMax < Y[j] ) yMax = Y[j];
      if( yMin > Y[j] ) yMin = Y[j];
      if( zMax < Z[j] ) zMax = Z[j];
      if( zMin > Z[j] ) zMin = Z[j];
    }
    if( nGrid == 0 ){
      yMin = yMax = zMin = zMax = 0;
      nGrid = 1;
    }

    row.Grid().Create( yMin, yMax, zMin, zMax, nGrid );

    float sy = ( CAMath::Abs( row.Grid().StepYInv() ) >1.e-4 ) ?1./row.Grid().StepYInv() :1;
    float sz = ( CAMath::Abs( row.Grid().StepZInv() ) >1.e-4 ) ?1./row.Grid().StepZInv() :1;

    //cout<<"grid n = "<<row.Grid().N()<<" "<<sy<<" "<<sz<<" "<<yMin<<" "<<yMax<<" "<<zMin<<" "<<zMax<<endl;
    
    bool recreate=0;
    if( sy < 2. ) { recreate = 1; sy = 2; }
    if( sz < 2. ) { recreate = 1; sz = 2; }
    if( recreate ) row.Grid().Create( yMin, yMax, zMin, zMax, sy, sz );

    fGridSizeTotal+=row.Grid().N()+3+10;    
    //cout<<"grid n = "<<row.Grid().N()<<endl;
  }
  
  fGrid1SizeTotal = fGridSizeTotal+10;

  SetPointers();  

  fEventMemory = (char*) ( new uint4 [ fEventMemSize/sizeof(uint4) + 100]);
  SetPointers();

  fGridSizeTotal = 0;
  fGrid1SizeTotal = 0;

  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ ){
    AliHLTTPCCARow &row = fRows[iRow];
    AliHLTTPCCAGrid &grid = row.Grid();

    Int_t c[grid.N()+3+10];
    Int_t bins[row.NHits()];
    Int_t filled[ row.Grid().N() +3+10 ];

    for( UInt_t bin=0; bin<row.Grid().N()+3; bin++ ) filled[bin] = 0;  

    for( Int_t i=0; i<row.NHits(); i++ ){
      Int_t j = RowFirstHit[iRow]+i;
      Int_t bin = row.Grid().GetBin( Y[j], Z[j] );
      bins[i] = bin;
      filled[bin]++;
    }

    {
      Int_t n=0;
      for( UInt_t bin=0; bin<row.Grid().N()+3; bin++ ){
	c[bin] = n;
	n+=filled[bin];
      }
    }
    for( Int_t i=0; i<row.NHits(); i++ ){ 
      Int_t bin = bins[i];
      Int_t ind = c[bin] + filled[bin]-1;
      AliHLTTPCCAHit &h = fHits[RowFirstHit[iRow]+ind];
      fHitsID[RowFirstHit[iRow]+ind] = RowFirstHit[iRow]+i;
      h.Y() = Y[row.FirstHit()+i];
      h.Z() = Z[row.FirstHit()+i];
      filled[bin]--;
    }

    grid.Offset() = fGridSizeTotal;
    Int_t off= grid.N()+3+10;
    fGridSizeTotal+=off;
    Int_t n2 = grid.N()/2;
    grid.Content2() = c[n2];
    UChar_t *cnew = fGridContents + grid.Offset();

    for( Int_t i=0; i<n2; i++ ){
      Int_t v = c[i];
      if( v>=256 ){
	//cout<<" ERROR!!! "<<v<<endl;
	v = 255;
      }else if( v<0 ){
	//cout<<" ERROR!!! "<<v<<endl;
	v = 0;
      }	      
      cnew[i] = (UChar_t ) v;
    }
    for( UInt_t i=n2; i<grid.N()+3; i++ ){
      Int_t v = c[i] - grid.Content2();
      if( v>=256 ){
	//cout<<" ERROR 1 !!! "<<v<<endl;
	v = 255;
      }else if( v<0 ){
	//cout<<" ERROR 1 !!! "<<v<<endl;
	v = 0;
      }
      cnew[i] = (UChar_t) v;	  
    }

    
    UInt_t *cnew1 = fGrid1Contents + grid.Offset();

    for( UInt_t i=0; i<grid.N()+1; i++ ){
      UInt_t g0n = 0;
      UInt_t g1n = 0;
      UInt_t g1 = 0;
      UInt_t g0 = c[i];// max [gN]
      UInt_t g0e = c[i+2]; //max[gN+2]
      g0n = g0e - g0;
      if( i+grid.Ny()< grid.N()+1 ){// max [gN-gNy]
	g1 = c[i+grid.Ny()]; // max [gN]
	UInt_t g1e = c[i+grid.Ny()+2];//max [gN+2]
	g1n = g1e - g1;	 
      }

      if( g0n > 63 ) g0n = 63;
      if( g1n > 63 ) g1n = 63;
      cnew1[i] = (g1n<<26) + (g1<<16) + (g0n<<10) + g0;
    }
    {
      float y0 = row.Grid().YMin();
      float stepY = (row.Grid().YMax() - y0)*(1./65535.);
      float z0 = row.Grid().ZMin();
      float stepZ = (row.Grid().ZMax() - z0)*(1./65535.);
      float stepYi = 1./stepY;
      float stepZi = 1./stepZ;
      
      row.Hy0() = y0;
      row.Hz0() = z0;
      row.HstepY() = stepY;
      row.HstepZ() = stepZ;
      row.HstepYi() = stepYi;
      row.HstepZi() = stepZi;
      
      for( Int_t ih=0; ih<row.NHits(); ih++ ){
	Int_t ihTot = RowFirstHit[iRow]+ih;
	AliHLTTPCCAHit &hh = fHits[ihTot];
	ushort2  &h = fHits1[ihTot];
	float xx = ((hh.Y() - y0)*stepYi); //SG!!!
	float yy = ((hh.Z() - z0)*stepZi);
	if( xx<0 || yy<0 || xx>=65536 || yy>= 65536 ){
	  cout<<"!!!! hit packing error!!! "<<xx<<" "<<yy<<" "<<endl;
	}
	h.x = (UShort_t) xx;//((hh.Y() - y0)*stepYi);
	h.y = (UShort_t) yy;//((hh.Z() - z0)*stepZi);
      }
    }

    if(1){       
      row.FullOffset() = fTexHitsFullSize;
      ushort2 *p= (ushort2*)(fTexHitsFullData+row.FullOffset());      
      for( Int_t ih=0; ih<row.NHits(); ih++ ){
  	Int_t ihTot = RowFirstHit[iRow]+ih;
	p[ih] = fHits1[ihTot];
      }
      Int_t size = row.NHits()*sizeof(ushort2);

      row.FullGridOffset() = row.NHits()*2;      
      UShort_t *p1 = ((UShort_t *)p) + row.FullGridOffset();

      Int_t n = grid.N();
      for( Int_t i=0; i<n; i++ ){
	p1[i] = c[i];
      }     
      UShort_t a = c[n];
      Int_t nn = n+grid.Ny()+2;
      for( Int_t i=n; i<nn; i++ ) p1[i] = a;

      size+= (nn)*sizeof(UShort_t);
      row.FullLinkOffset() = row.NHits()*2 + nn;
      size+= row.NHits()*2*sizeof(Short_t);
      if( size%16 ) size = size/16+1;
      else size = size/16;
      row.FullSize()=size;
      //cout<<iRow<<", "<<row.fNHits<<"= "<<size*16<<"b: "<<row.fFullOffset<<" "<<row.fFullSize<<" "<<row.fFullGridOffset<<" "<<row.fFullLinkOffset<<endl;

      fTexHitsFullSize+=size;
    }
  }
  fGrid1SizeTotal = fGridSizeTotal+10;
}


GPUh() void AliHLTTPCCATracker::Reconstruct()
{
  //* reconstruction of event
  
#ifdef DRAW
  if( !gApplication ){
    TApplication *myapp = new TApplication("myapp",0,0);
  }
  //AliHLTTPCCADisplay::Instance().Init();
  
  AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
  AliHLTTPCCADisplay::Instance().SetSliceView();
  AliHLTTPCCADisplay::Instance().DrawSlice( this );  
  //for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
  //for (Int_t i = 0; i<fRows[iRow].NHits(); i++) 
  //AliHLTTPCCADisplay::Instance().DrawHit( iRow, i );
  //AliHLTTPCCADisplay::Instance().Ask();
#endif

  fTimers[0] = 0; // find neighbours
  fTimers[1] = 0; // construct tracklets
  fTimers[2] = 0; // fit tracklets
  fTimers[3] = 0; // prolongation of tracklets
  fTimers[4] = 0; // selection
  fTimers[5] = 0; // write output
  fTimers[6] = 0;
  fTimers[7] = 0;

  if( fNHitsTotal < 1 ) return;
  //if( fParam.ISlice()!=3 ) return;
  TStopwatch timer0;
  *fNTracks = 0;
#if !defined(HLTCA_GPUCODE)  

  AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder>( Param().NRows(), 1, *this );
  AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner>( Param().NRows()-2, 1, *this );
  AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder>( Param().NRows()-4, 1, *this );

  Int_t nStartHits = *fStartHits;      
  
  Int_t nThreads = 128;
  Int_t nBlocks = fNHitsTotal/nThreads + 1;
  if( nBlocks<12 ){
    nBlocks = 12; 
    nThreads = fNHitsTotal/12+1;
    if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
  }
      
  nThreads = fNHitsTotal;
  nBlocks = 1;

  AliHLTTPCCAProcess<AliHLTTPCCAUsedHitsInitialiser>(nBlocks, nThreads,*this);

  nThreads = 256;
  nBlocks = 30;

  nThreads = 1;
  nBlocks = 1;
  
  AliHLTTPCCAProcess<AliHLTTPCCALinksWriter>(nBlocks, nThreads,*this);

  Int_t nMemThreads = 128;
  nThreads = 256;//96;
  nBlocks = nStartHits/nThreads + 1;
  if( nBlocks<30 ){
    nBlocks = 30;
    nThreads = (nStartHits)/30+1;
    if( nThreads%32 ) nThreads = (nThreads/32+1)*32;
  }

  nThreads = nStartHits;
  nBlocks = 1;

  AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor>(nBlocks, nMemThreads+nThreads,*this);

  { 
    nThreads = 128;
    nBlocks = nStartHits/nThreads + 1;
    if( nBlocks<12 ){
      nBlocks = 12;  
      nThreads = nStartHits/12+1;
      nThreads = (nThreads/32+1)*32;
    }
    *fStartHits = 0;
    *fTrackHits = 0;

    nThreads = nStartHits;
    nBlocks = 1;


    AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector>(nBlocks, nThreads,*this);

    //cudaMemcpy(cpuTrackerCopy.fNTracks, gpuTrackerCopy.fNTracks, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(cpuTrackerCopy.fTrackHits, gpuTrackerCopy.fTrackHits, sizeof(int), cudaMemcpyDeviceToHost);
	  
    //Int_t size = sizeof(AliHLTTPCCATrack)*( *cpuTrackerCopy.fNTracks );
    //cudaMemcpy(cpuTrackerCopy.fTracks, gpuTrackerCopy.fTracks, size, cudaMemcpyDeviceToHost);
    //cout<<"Tracks size = "<<size<<endl;	
    
    //size = sizeof(Int_t)*( *cpuTrackerCopy.fTrackHits );
    //cudaMemcpy(cpuTrackerCopy.fTrackHits+1, gpuTrackerCopy.fTrackHits+1, size, cudaMemcpyDeviceToHost);
    //cout<<"Track hits size = "<<size<<endl;
    //cpuTrackerCopy.WriteOutput();
 
    Int_t nTracklets = *fStartHits;

    cout<<"Slice "<<Param().ISlice()<<": N start hits/tracklets/tracks = "<<nStartHits<<" "<<nTracklets<<" "<<*fNTracks<<endl;
   WriteOutput();      
  }

#endif

  timer0.Stop();
  fTimers[0] = timer0.CpuTime();

 }




GPUh() void AliHLTTPCCATracker::WriteOutput()
{
  // write output

  TStopwatch timer;
  fOutTrackHits = new Int_t[fNHitsTotal*10];
  fOutTracks = new AliHLTTPCCAOutTrack[*fNTracks];
  fNOutTrackHits = 0;
  fNOutTracks = 0;
  //cout<<"NTracks = "<<*fNTracks<<endl;
  //cout<<"NHits = "<<fNHitsTotal<<endl;
  for( Int_t iTr=0; iTr<*fNTracks; iTr++){
    //cout<<"iTr = "<<iTr<<endl;
    AliHLTTPCCATrack &iTrack = fTracks[iTr];
    if( !iTrack.Alive() ) continue;
    if( iTrack.NHits()<3 ) continue;      
    //cout<<10<<endl;
    AliHLTTPCCAOutTrack &out = fOutTracks[fNOutTracks];
    out.FirstHitRef() = fNOutTrackHits;
    out.NHits() = 0;
    out.OrigTrackID() = iTr;
    {
      out.StartPoint() = iTrack.Param();
      out.EndPoint() = iTrack.Param();
    }
    //cout<<11<<endl;

    Int_t iID = iTrack.FirstHitID();
    Int_t fNOutTrackHitsOld = fNOutTrackHits;
    //cout<<12<<" "<<iID<<" "<<iTrack.NHits()<<endl;
    for( Int_t ith=0; ith<iTrack.NHits(); ith++ ){
      //cout<<ith<<":"<<endl;
      Int_t ic = (fTrackHits[iID+ith]);
      //cout<<ic<<endl;
      AliHLTTPCCARow &row = ID2Row(ic);
      Int_t ih = ID2IHit(ic);
      //cout<<"write row,hit="<<ID2IRow(ic)<<" "<<ih<<endl;
      fOutTrackHits[fNOutTrackHits] = fHitsID[row.FirstHit()+ih];
      fNOutTrackHits++;
      //cout<<"ok"<<endl;
      if( fNOutTrackHits>fNHitsTotal*10 ){
	cout<<"fNOutTrackHits>fNHitsTotal"<<endl;
	exit(0);//SG!!!
	return;
      }
      out.NHits()++;      
    }
    //cout<<13<<endl;
    //cout<<fNOutTracks<<": itr = "<<iTr<<", n outhits = "<<out.NHits()<<endl;
    if( out.NHits() >= 2 ){
      fNOutTracks++;
    }else {
      fNOutTrackHits = fNOutTrackHitsOld;
    }
  }
  timer.Stop();
  fTimers[5]+=timer.CpuTime();
}

GPUh() void AliHLTTPCCATracker::FitTrackFull( AliHLTTPCCATrack &/**/, Float_t * /**/ ) const 
{  
  // fit track with material
#ifdef XXX    
  //* Fit the track   
  FitTrack( iTrack, tt0 );
  if( iTrack.NHits()<=3 ) return;
    
  AliHLTTPCCATrackParam &t = iTrack.Param();
  AliHLTTPCCATrackParam t0 = t;

  t.Chi2() = 0;
  t.NDF() = -5;	
  Bool_t first = 1;

  Int_t iID = iTrack.FirstHitID();
  for( Int_t ih=0; ih<iTrack.NHits(); ih++, iID++ ){
    Int_t *ic = &(fTrackHits[iID]);
    Int_t iRow = ID2IRow(*ic);
    AliHLTTPCCARow &row = fRows[iRow];      
    if( !t0.TransportToX( row.X() ) ) continue;      
    Float_t dy, dz;
    AliHLTTPCCAHit &h = ID2Hit(*ic);

    // check for wrong hits	
    if(0){
      dy = t0.GetY() - h.Y();
      dz = t0.GetZ() - h.Z();
      
      //if( dy*dy > 3.5*3.5*(/*t0.GetErr2Y() + */h.ErrY()*h.ErrY() ) ) continue;//SG!!!
      //if( dz*dz > 3.5*3.5*(/*t0.GetErr2Z() + */h.ErrZ()*h.ErrZ() ) ) continue;      
    }

    if( !t.TransportToX( row.X() ) ) continue;  

    //* Update the track
	    
    if( first ){
      t.Cov()[ 0] = .5*.5;
      t.Cov()[ 1] = 0;
      t.Cov()[ 2] = .5*.5;
      t.Cov()[ 3] = 0;
      t.Cov()[ 4] = 0;
      t.Cov()[ 5] = .2*.2;
      t.Cov()[ 6] = 0;
      t.Cov()[ 7] = 0;
      t.Cov()[ 8] = 0;
      t.Cov()[ 9] = .2*.2;
      t.Cov()[10] = 0;
      t.Cov()[11] = 0;
      t.Cov()[12] = 0;
      t.Cov()[13] = 0;
      t.Cov()[14] = .2*.2;
      t.Chi2() = 0;
      t.NDF() = -5;
    }
    Float_t err2Y, err2Z;
    GetErrors2( iRow, t, err2Y, err2Z );

    if( !t.Filter2( h.Y(), h.Z(), err2Y, err2Z ) ) continue;

    first = 0;      
  }  
  /*
  Float_t cosPhi = iTrack.Param().GetCosPhi();
  p0.Param().TransportToX(ID2Row( iTrack.PointID()[0] ).X());
  p2.Param().TransportToX(ID2Row( iTrack.PointID()[1] ).X());  
  if( p0.Param().GetCosPhi()*cosPhi<0 ){ // change direction
  Float_t *par = p0.Param().Par();
  Float_t *cov = p0.Param().Cov();
  par[2] = -par[2]; // sin phi
  par[3] = -par[3]; // DzDs
    par[4] = -par[4]; // kappa
    cov[3] = -cov[3];
    cov[4] = -cov[4];
    cov[6] = -cov[6];
    cov[7] = -cov[7];
    cov[10] = -cov[10];
    cov[11] = -cov[11];
    p0.Param().CosPhi() = -p0.Param().GetCosPhi();
  }
  */
#endif
}

GPUh() void AliHLTTPCCATracker::FitTrack( AliHLTTPCCATrack &/*track*/, Float_t */*t0[]*/ ) const 
{      
  //* Fit the track   
#ifdef XXX
  AliHLTTPCCAEndPoint &p2 = ID2Point(track.PointID()[1]);
  AliHLTTPCCAHit &c0 = ID2Hit(fTrackHits[p0.TrackHitID()].HitID());	
  AliHLTTPCCAHit &c1 = ID2Hit(fTrackHits[track.HitID()[1]].HitID());	
  AliHLTTPCCAHit &c2 = ID2Hit(fTrackHits[p2.TrackHitID()].HitID());	
  AliHLTTPCCARow &row0 = ID2Row(fTrackHits[p0.TrackHitID()].HitID());
  AliHLTTPCCARow &row1 = ID2Row(fTrackHits[track.HitID()[1]].HitID());
  AliHLTTPCCARow &row2 = ID2Row(fTrackHits[p2.TrackHitID()].HitID());
  Float_t sp0[5] = {row0.X(), c0.Y(), c0.Z(), c0.ErrY(), c0.ErrZ() };
  Float_t sp1[5] = {row1.X(), c1.Y(), c1.Z(), c1.ErrY(), c1.ErrZ() };
  Float_t sp2[5] = {row2.X(), c2.Y(), c2.Z(), c2.ErrY(), c2.ErrZ() };
  //cout<<"Fit track, points ="<<sp0[0]<<" "<<sp0[1]<<" / "<<sp1[0]<<" "<<sp1[1]<<" / "<<sp2[0]<<" "<<sp2[1]<<endl;
  if( track.NHits()>=3 ){    
    p0.Param().ConstructXYZ3(sp0,sp1,sp2,p0.Param().CosPhi(), t0);
    p2.Param().ConstructXYZ3(sp2,sp1,sp0,p2.Param().CosPhi(), t0);
    //p2.Param() = p0.Param();
    //p2.Param().TransportToX(row2.X());
    //p2.Param().Par()[1] = -p2.Param().Par()[1];
    //p2.Param().Par()[4] = -p2.Param().Par()[4];
  } else {
    p0.Param().X() = row0.X();
    p0.Param().Y() = c0.Y();
    p0.Param().Z() = c0.Z();
    p0.Param().Err2Y() = c0.ErrY()*c0.ErrY();
    p0.Param().Err2Z() = c0.ErrZ()*c0.ErrZ();
    p2.Param().X() = row2.X();
    p2.Param().Y() = c2.Y();
    p2.Param().Z() = c2.Z();
    p2.Param().Err2Y() = c2.ErrY()*c2.ErrY();
    p2.Param().Err2Z() = c2.ErrZ()*c2.ErrZ();
  }
#endif
}



GPUd() void AliHLTTPCCATracker::GetErrors2( Int_t iRow, const AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  Float_t z = CAMath::Abs((250.-0.275)-CAMath::Abs(t.GetZ()));
  Int_t    type = (iRow<63) ? 0: (iRow>126) ? 1:2;
  Float_t cosPhiInv = CAMath::Abs(t.GetCosPhi())>1.e-2 ?1./t.GetCosPhi() :0;
  Float_t angleY = t.GetSinPhi()*cosPhiInv ;
  Float_t angleZ = t.GetDzDs()*cosPhiInv ;

  Err2Y = fParam.GetClusterError2(0,type, z,angleY);  
  Err2Z = fParam.GetClusterError2(1,type, z,angleZ);
}

GPUd() void AliHLTTPCCATracker::GetErrors2( Int_t iRow, const AliHLTTPCCATrackParam1 &t, Float_t &Err2Y, Float_t &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  Float_t z = CAMath::Abs((250.-0.275)-CAMath::Abs(t.GetZ()));
  Int_t    type = (iRow<63) ? 0: (iRow>126) ? 1:2;
  Float_t cosPhiInv = CAMath::Abs(t.GetCosPhi())>1.e-2 ?1./t.GetCosPhi() :0;
  Float_t angleY = t.GetSinPhi()*cosPhiInv ;
  Float_t angleZ = t.GetDzDs()*cosPhiInv ;

  Err2Y = fParam.GetClusterError2(0,type, z,angleY);  
  Err2Z = fParam.GetClusterError2(1,type, z,angleZ);
}
