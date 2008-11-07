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

#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCACell.h"
#include "AliHLTTPCCAEndPoint.h"
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"

#include "TMath.h"
#include "Riostream.h"
//#include <algo.h>
#include "TStopwatch.h"

//#define DRAW

#ifdef DRAW
  #include "AliHLTTPCCADisplay.h"
  #include "TApplication.h"
#endif //DRAW

ClassImp(AliHLTTPCCATracker)


AliHLTTPCCATracker::AliHLTTPCCATracker()
  :fParam(),fRows(0),fOutTrackHits(0),fNOutTrackHits(0),fOutTracks(0),fNOutTracks(0),fNHitsTotal(0),fTracks(0),fNTracks(0),fCellHitPointers(0),fCells(0),fEndPoints(0)
{
  // constructor
  //fRows = new AliHLTTPCCARow[fParam.NRows()];
  //Initialize( fParam );
}

AliHLTTPCCATracker::AliHLTTPCCATracker( const AliHLTTPCCATracker& )
  :fParam(),fRows(0),fOutTrackHits(0),fNOutTrackHits(0),fOutTracks(0),fNOutTracks(0),fNHitsTotal(0),fTracks(0),fNTracks(0),fCellHitPointers(0),fCells(0),fEndPoints(0)
{
  // dummy
}

AliHLTTPCCATracker &AliHLTTPCCATracker::operator=( const AliHLTTPCCATracker& )
{
  // dummy
  fRows=0;
  fOutTrackHits=0;
  fOutTracks=0;
  fNOutTracks=0;
  return *this;
}

AliHLTTPCCATracker::~AliHLTTPCCATracker()
{
  // destructor
  StartEvent();
  delete[] fRows;
}

// ----------------------------------------------------------------------------------
void AliHLTTPCCATracker::Initialize( AliHLTTPCCAParam &param )
{
  // initialisation
  StartEvent();
  delete[] fRows;
  fRows = 0;
  fParam = param;  
  fParam.Update();
  fRows = new AliHLTTPCCARow[fParam.NRows()];
  Float_t xStep = 1;
  Float_t deltaY = TMath::Tan(fParam.CellConnectionAngleXY());
  Float_t deltaZ = TMath::Tan(fParam.CellConnectionAngleXZ());
  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    fRows[irow].X() = fParam.RowX(irow);        
    if( irow < fParam.NRows()-1 ) xStep = fParam.RowX(irow+1) - fParam.RowX(irow);
    fRows[irow].DeltaY() = xStep*deltaY;
    fRows[irow].DeltaZ() = xStep*deltaZ;
    fRows[irow].MaxY() = TMath::Tan( fParam.DAlpha()/2.)*fRows[irow].X();
  }
  StartEvent();
}

void AliHLTTPCCATracker::StartEvent()
{
  // start new event and fresh the memory  

  if( fTracks ) delete[] fTracks;
  if( fOutTrackHits ) delete[] fOutTrackHits;
  if( fOutTracks ) delete[] fOutTracks;
  if( fCellHitPointers ) delete[] fCellHitPointers;
  if( fCells ) delete[] fCells;
  if( fEndPoints ) delete[] fEndPoints;
  fTracks = 0;
  fOutTrackHits = 0;
  fOutTracks = 0;
  fCellHitPointers = 0;
  fCells = 0;
  fEndPoints = 0;
  fNTracks = 0;
  fNOutTrackHits = 0;
  fNOutTracks = 0;
  fNHitsTotal = 0;
  if( fRows ) {
    for( Int_t irow=0; irow<fParam.NRows(); irow++ )  fRows[irow].Clear();
  }
}


void AliHLTTPCCATracker::ReadHitRow( Int_t iRow, AliHLTTPCCAHit *Row, Int_t NHits )
{
  // read row of hits
  AliHLTTPCCARow &row = fRows[iRow];
  row.Hits() = new AliHLTTPCCAHit[NHits];
  for( Int_t i=0; i<NHits; i++ ){ 
    row.Hits()[i]=Row[i];
    row.Hits()[i].ErrY()*= fParam.YErrorCorrection();
    row.Hits()[i].ErrZ()*= fParam.ZErrorCorrection();
  }
  row.NHits() = NHits;
  fNHitsTotal += NHits;
}

void AliHLTTPCCATracker::Reconstruct()
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

  fTimers[0] = 0;
  fTimers[1] = 0;
  fTimers[2] = 0;
  fTimers[3] = 0;
  fTimers[4] = 0;
  fTimers[5] = 0;
  fTimers[6] = 0;
  fTimers[7] = 0;
  if( fNHitsTotal < 1 ) return;

  //cout<<"Find Cells..."<<endl;
  FindCells();
  //cout<<"Merge Cells..."<<endl;
  MergeCells();
  //cout<<"Find Tracks..."<<endl;
  FindTracks();
  //cout<<"Find Tracks OK"<<endl;
 }


void AliHLTTPCCATracker::FindCells()
{
  //* cell finder - neighbouring hits are grouped to cells

  TStopwatch timer;

  fCellHitPointers = new Int_t [fNHitsTotal];
  fCells = new AliHLTTPCCACell[fNHitsTotal]; 
  fEndPoints = new AliHLTTPCCAEndPoint[fNHitsTotal]; 
 
  struct THitCont{
    Float_t Ymin, Ymax, Zmin, Zmax;
    Int_t binYmin, binYmax, binZmin, binZmax;
    Bool_t used;
    AliHLTTPCCAHit *h;
    THitCont *next;
  };
  THitCont *hitCont = new THitCont[fNHitsTotal];    

  Int_t lastCellHitPointer = 0;
  Int_t lastCell = 0;

  for( Int_t irow=0; irow<fParam.NRows(); irow++ ){
    AliHLTTPCCARow &row=fRows[irow];
    Int_t nHits = row.NHits();
    //cout<<"row = "<<irow<<", x="<<row.X()<<endl;
    if( nHits<1 ) continue;
    //cout<<nHits*sizeof(AliHLTTPCCAHit)/1024.<<endl;

    Float_t deltaY = row.DeltaY();
    Float_t deltaZ = row.DeltaZ();

    Float_t yMin = 1.e20, zMin = 1.e20, yMax = -1.e20, zMax = -1.e20;
    for (Int_t ih = 0; ih<nHits; ih++){    
      AliHLTTPCCAHit &h  = row.Hits()[ih];    
      if( yMin> h.Y() ) yMin = h.Y();
      if( yMax< h.Y() ) yMax = h.Y();
      if( zMin> h.Z() ) zMin = h.Z();
      if( zMax< h.Z() ) zMax = h.Z();
    }
    AliHLTTPCCAGrid grid;
    grid.Create( yMin, yMax, zMin, zMax, nHits );
    //cout<<"row "<<irow<<", delta = "<<delta<<" :\n"<<endl;
    for (Int_t ih = 0; ih<nHits; ih++){    
      AliHLTTPCCAHit &h  = row.Hits()[ih];
      THitCont &cont = hitCont[ih];
      THitCont *&bin = * ((THitCont **) grid.GetNoCheck(h.Y(),h.Z()));
      cont.h = &h;
      cont.used = 0;
      Float_t y = h.Y();
      //cout<<"ih = "<<ih<<", y= "<<y<<endl;
      Float_t dY = 3.5*h.ErrY() + deltaY;
      cont.Ymin = y-dY;
      cont.Ymax = y+dY;
      Float_t z = h.Z();
      Float_t dZ = 3.5*h.ErrZ() + deltaZ;
      cont.Zmin = z-dZ;
      cont.Zmax = z+dZ;
      cont.binYmin = (Int_t ) ( (cont.Ymin-dY-grid.YMin())*grid.StepYInv() );
      cont.binYmax = (Int_t ) ( (cont.Ymax+dY-grid.YMin())*grid.StepYInv() );
      cont.binZmin = (Int_t ) ( (cont.Zmin-dZ-grid.ZMin())*grid.StepZInv() );
      cont.binZmax = (Int_t ) ( (cont.Zmax+dZ-grid.ZMin())*grid.StepZInv() );
      if( cont.binYmin<0 ) cont.binYmin = 0;
      if( cont.binYmin>=grid.Ny() ) cont.binYmin = grid.Ny()-1;
      if( cont.binYmax<0 ) cont.binYmax = 0;
      if( cont.binYmax>=grid.Ny() ) cont.binYmax = grid.Ny()-1;
      if( cont.binZmin<0 ) cont.binZmin = 0;
      if( cont.binZmin>=grid.Nz() ) cont.binZmin = grid.Nz()-1;
      if( cont.binZmax<0 ) cont.binZmax = 0;
      if( cont.binZmax>=grid.Nz() ) cont.binZmax = grid.Nz()-1;
      cont.next = bin;
      bin = &cont;
    }        

    row.CellHitPointers() = fCellHitPointers + lastCellHitPointer;
    row.Cells() = fCells + lastCell;
    Int_t nPointers = 0;
    Int_t nCells = 0;

    //Int_t statMaxBins = 0;
    //Int_t statMaxHits = 0;
    for (Int_t ih = 0; ih<nHits; ih++){    
      THitCont &cont = hitCont[ih];
      if( cont.used ) continue;
      // cell start
      AliHLTTPCCACell &cell = row.Cells()[nCells++];
      cell.FirstHitRef() = nPointers;
      cell.NHits() = 1;
      cell.Link() = -1;
      cell.Status() = 0;
      cell.TrackID() = -1;
      row.CellHitPointers()[nPointers++] = ih;
      cont.used = 1;      

#ifdef DRAW
      //AliHLTTPCCADisplay::Instance().DrawHit( irow, ih, kRed );
#endif
  // cell finder - neighbouring hits are grouped to cells

      
      Float_t ymin = cont.Ymin;
      Float_t ymax = cont.Ymax;
      Float_t zmin = cont.Zmin;
      Float_t zmax = cont.Zmax;
      Int_t binYmin = cont.binYmin;
      Int_t binYmax = cont.binYmax;
      Int_t binZmin = cont.binZmin;
      Int_t binZmax = cont.binZmax;

      Bool_t repeat = 1;
      while( repeat ){
	repeat = 0;
	THitCont ** startY = (THitCont **) grid.Grid() + binZmin*grid.Ny();
	//Float_t Ymax1 = Ymax;
	//Float_t Ymin1 = Ymin;
	//Float_t Zmax1 = Zmax;
	//Float_t Zmin1 = Zmin;
	Int_t binYmax1 = binYmax;
	Int_t binYmin1 = binYmin;
	Int_t binZmax1 = binZmax;
	Int_t binZmin1 = binZmin;
#ifdef DRAW
	//cell.Y() = .5*(Ymin+Ymax);
	//cell.Z() = .5*(Zmin+Zmax);
	//cell.ErrY() = .5*( Ymax - Ymin )/3.5;
	//cell.ErrZ() = .5*( Zmax - Zmin )/3.5;
	//cell.YMin() = Ymin;
	//cell.YMax() = Ymax;
	//AliHLTTPCCADisplay::Instance().DrawCell( irow, nCells-1, 1,kRed );
	//AliHLTTPCCADisplay::Instance().Ask();
#endif
	for( Int_t iGridZ=binZmin1; iGridZ<=binZmax1; iGridZ++, startY += grid.Ny() ){
	  for( Int_t iGridY=binYmin1; iGridY<=binYmax1; iGridY++ ){
	    for( THitCont *bin = *(startY + iGridY); bin; bin=bin->next ){
	      Int_t jh = bin->h-row.Hits();
	      THitCont &cont1 = hitCont[jh];
	      if( cont1.used ) continue;	      
	      //cout<<"["<<Ymin<<","<<Ymax<<"]: ["<<cont1.Ymin<<","<<cont1.Ymax<<"]"<<endl;
	      if( cont1.Ymax < ymin ) continue; 
	      if( cont1.Ymin > ymax ) continue;
	      if( cont1.Zmax < zmin ) break;// in the grid cell hit Y is decreasing
	      if( cont1.Zmin > zmax ) continue;
		
	      if( cont1.Ymin < ymin ){ ymin = cont1.Ymin; repeat = 1; }
	      if( cont1.Ymax > ymax ){ ymax = cont1.Ymax; repeat = 1; }
	      if( cont1.Zmin < zmin ){ zmin = cont1.Zmin; repeat = 1; }
	      if( cont1.Zmax > zmax ){ zmax = cont1.Zmax; repeat = 1; }
	      if( cont1.binYmin < binYmin ){ binYmin = cont1.binYmin; repeat = 1; }
	      if( cont1.binYmax > binYmax ){ binYmax = cont1.binYmax; repeat = 1; }
	      if( cont1.binZmin < binZmin ){ binZmin = cont1.binZmin; repeat = 1; }
	      if( cont1.binZmax > binZmax ){ binZmax = cont1.binZmax; repeat = 1; }
		
	      row.CellHitPointers()[nPointers++] = jh;
	      cell.NHits()++;
	      cont1.used = 1;	    
#ifdef DRAW
	      //AliHLTTPCCADisplay::Instance().DrawHit( irow, jh, kRed );
	      //AliHLTTPCCADisplay::Instance().Ask();  
#endif
	    }
	  }
	}
      }
      
      cell.Y() = .5*(ymin+ymax);
      cell.Z() = .5*(zmin+zmax);
      cell.ErrY() = .5*( ymax - ymin - 2*deltaY)/3.5;
      cell.ErrZ() = .5*( zmax - zmin -2*deltaZ)/3.5;
      cell.ZMin() = zmin;
      cell.ZMax() = zmax;
#ifdef DRAW
      //AliHLTTPCCADisplay::Instance().DrawCell( irow, nCells-1 );
      //AliHLTTPCCADisplay::Instance().Ask();  
#endif
    }
    //cout<<statMaxBins<<"/"<<grid.N()<<" "<<statMaxHits<<"/"<<nHits<<endl;

    
    row.NCells() = nCells;
    lastCellHitPointer += nPointers;
    lastCell += nCells;
  }
  delete[] hitCont;
  timer.Stop();
  fTimers[0] = timer.CpuTime();
}


void AliHLTTPCCATracker::MergeCells()
{
  // First step: 
  //  for each Cell find one neighbour in the next row (irow+1)
  //  when there are no neighbours, look to the rows (irow+2),(irow+3)
  //
  // Initial state: 
  //  cell.Link  =-1
  //  cell.Link1  =-1
  //  cell.Track      =-1
  //
  // Intermediate state: same as final
  //
  // Final state:
  //  cell.Link = Neighbour ID, if there is a neighbour 
  //             = -1, if no neighbours found
  //             = -2, if there was more than one neighbour
  //  cell.Link1 = ID of the cell which has this Cell as a forward neighbour
  //             = -1 there are no backward neighbours
  //             = -2 there are more than one neighbour
  //  cell.Track = -1
  //

  TStopwatch timer;

  Int_t nStartCells = 0;
  for( Int_t iRow1=0; iRow1<fParam.NRows(); iRow1++ ){
    AliHLTTPCCARow &row1 = fRows[iRow1];
      
    Float_t deltaY = row1.DeltaY();
    Float_t deltaZ = row1.DeltaZ();
    Float_t xStep = 1;
    if( iRow1 < fParam.NRows()-1 ) xStep = fParam.RowX(iRow1+1) - fParam.RowX(iRow1);
    Float_t tx = xStep/row1.X();
 
    Int_t lastRow2 = iRow1+3;
    if( lastRow2>=fParam.NRows() ) lastRow2 = fParam.NRows()-1;

    for (Int_t i1 = 0; i1<row1.NCells(); i1++){
      AliHLTTPCCACell &c1  = row1.Cells()[i1];
      //cout<<"row, cell= "<<iRow1<<" "<<i1<<" "<<c1.Y()<<" "<<c1.ErrY()<<" "<<c1.Z()<<" "<<c1.ErrZ()<<endl;
      //Float_t sy1 = c1.ErrY()*c1.ErrY();      
      Float_t yy = c1.Y() +tx*c1.Y();
      Float_t zz = c1.Z() +tx*c1.Z();
      
      Float_t yMin = yy - 3.5*c1.ErrY() - deltaY;
      Float_t yMax = yy + 3.5*c1.ErrY() + deltaY;
      Float_t zMin = zz - 3.5*c1.ErrZ() - deltaZ;
      Float_t zMax = zz + 3.5*c1.ErrZ() + deltaZ;
      //Float_t sz1 = c1.ErrZ()*c1.ErrZ();
      if( c1.Status()<=0 ) nStartCells++;

      // looking for neighbour for the Cell c1
      Bool_t found = 0;
      for( Int_t iRow2=iRow1+1; iRow2<=lastRow2&&(!found); iRow2++ ){
	AliHLTTPCCARow &row2 = fRows[iRow2];
	AliHLTTPCCACell *cc2 = lower_bound(row2.Cells(),row2.Cells()+row2.NCells(),zMin,AliHLTTPCCARow::CompareCellZMax);
	for (Int_t i2 = (cc2 - row2.Cells()); i2<row2.NCells(); i2++){
	  //cout<<"   candidat = "<<iRow2<<" "<<i2<<endl;
	  
	  AliHLTTPCCACell &c2  = row2.Cells()[i2];
	  Float_t y2Min = c2.Y() - 3.5*c2.ErrY();
	  Float_t y2Max = c2.Y() + 3.5*c2.ErrY();
	  Float_t z2Min = c2.Z() - 3.5*c2.ErrZ();
	  Float_t z2Max = c2.Z() + 3.5*c2.ErrZ();

	  if( y2Min > yMax ) continue;
	  if( y2Max < yMin ) continue;
	  if( z2Min > zMax ) break;
	  if( z2Max < zMin ) continue;

	  // c1 & c2 are neighbours

	  found = 1;
	  
	  if( c1.Link() ==-1 && c2.Status()==0 ){ 
	    // one-to-one connection - OK
	    c1.Link() = IRowICell2ID(iRow2,i2);
	    c2.Status() = 1;
	  }else{
	    // multi-connection - break all links
	    if( c1.Link()>=0 ) ID2Cell(c1.Link()).Status() = -1;	    
	    c1.Link()  = -2;
	    c2.Status() = -1;
	  }
	}
      }//row2
    }
  }//row1

  timer.Stop();
  fTimers[1] = timer.CpuTime();
  
  // Second step: create tracks 
  //  for each sequence of neighbouring Cells create Track object
  //
  // Final state:
  //  cell.Track     = TrackNumber for first and last track cell
  //                 = -1 for other cells
  //  cell.Link     = Neighbour ID, if there is a neighbour 
  //                 = -1, if no neighbour (last cell on the track )
  //  cell.Link1     = backward neighbour ID, if there is a neighbour 
  //                 = -1 for first and last track cells
  
  TStopwatch timer2;
  
  fTracks = new AliHLTTPCCATrack[nStartCells];
  fNTracks = 0;

  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ ){
    AliHLTTPCCARow &row = fRows[iRow];
    for( Int_t iCell = 0; iCell<row.NCells(); iCell++){ 
      AliHLTTPCCACell &c  = row.Cells()[iCell];
      if( c.Status()>0 ) continue; // not a starting cell

      Int_t firstID = IRowICell2ID( iRow, iCell );
      Int_t midID = firstID;
      Int_t lastID = firstID;

      AliHLTTPCCATrack &track = fTracks[fNTracks];
      track.Alive() = 1;
      track.NCells() = 1;
      AliHLTTPCCACell *last = &c;
      while( last->Link() >=0 ){
	Int_t nextID = last->Link();
	AliHLTTPCCACell *next = & ID2Cell(nextID);
	if(next->Status()!=1 ){
	  last->Link() = -1;
	  break;
	}	
	track.NCells()++;
	last = next;
	lastID = nextID;
      }
      Int_t nCells05 = (track.NCells()-1)/2;
      for( Int_t i=0; i<nCells05; i++ ) midID = ID2Cell(midID).Link();
      //cout<<fNTracks<<", NCells="<<track.NCells()<<" "<<nCells05<<"id="<<firstID<<" "<<midID<<" "<<lastID<<endl;
      c.TrackID() = fNTracks;
      last->TrackID() = fNTracks;
      track.FirstCellID() = firstID;
      track.CellID()[0] = firstID;
      track.CellID()[1] = midID;
      track.CellID()[2] = lastID;
      track.PointID()[0] = -1;
      track.PointID()[1] = -1;
      //cout<<"Track N "<<fNTracks<<", NCells="<<track.NCells()<<endl;

      fNTracks++;
    }
  }
  if( fNTracks != nStartCells ){
    //cout<<"fNTracks="<<fNTracks<<", NStrartCells="<<nStartCells<<endl;
    //exit(0);
    return;
  }

  // create endpoints 

  Int_t nEndPointsTotal = 0;
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ ){
    AliHLTTPCCARow &row = fRows[iRow];
    row.EndPoints()= fEndPoints + nEndPointsTotal;
    row.NEndPoints()=0;
    for( Int_t iCell = 0; iCell<row.NCells(); iCell++){ 
      AliHLTTPCCACell &c  = row.Cells()[iCell];
      if( c.TrackID()< 0 ) continue; // not an endpoint
      AliHLTTPCCAEndPoint &p = row.EndPoints()[row.NEndPoints()];
      p.CellID() = IRowICell2ID(iRow,iCell);
      p.TrackID() = c.TrackID();
      p.Link() = -1;
      AliHLTTPCCATrack &track = fTracks[c.TrackID()];
      if( c.Link()>=0 ){
	track.PointID()[0] = IRowICell2ID(iRow,row.NEndPoints());
      }else{
	track.PointID()[1] = IRowICell2ID(iRow,row.NEndPoints());
	if( track.PointID()[0]<0 )track.PointID()[0] = track.PointID()[1];
      }
      row.NEndPoints()++;
    }
    nEndPointsTotal += row.NEndPoints();
  }
  timer2.Stop();
  fTimers[2] = timer2.CpuTime();
}

void AliHLTTPCCATracker::FindTracks()
{
  // the Cellular Automaton track finder  
  TStopwatch timer3;
  //cout<<"combine & fit tracks"<<endl;
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    AliHLTTPCCATrack &iTrack = fTracks[itr];
    //if( iTrack.NCells()<3 ) continue;
    //cout<<" fit track "<<itr<<", NCells="<<iTrack.NCells()<<endl;    
    ID2Point(iTrack.PointID()[0]).Param().CosPhi() = -1;
    ID2Point(iTrack.PointID()[1]).Param().CosPhi() = 1;
    FitTrack( iTrack );
    //if( iTrack.Param().Chi2() > fParam.TrackChi2Cut()*iTrack.Param().NDF() ){
      //iTrack.Alive() = 0;
    //}
  }
  timer3.Stop();
  fTimers[3] = timer3.CpuTime();

#ifdef DRAW
  if( !gApplication ){
    TApplication *myapp = new TApplication("myapp",0,0);
  }    
  //AliHLTTPCCADisplay::Instance().Init();
  
  AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
  AliHLTTPCCADisplay::Instance().DrawSlice( this );
  cout<<"draw hits..."<<endl;
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NHits(); i++) 
      AliHLTTPCCADisplay::Instance().DrawHit( iRow, i );
  AliHLTTPCCADisplay::Instance().Ask();
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().DrawSlice( this );
  cout<<"draw cells..."<<endl;
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      AliHLTTPCCADisplay::Instance().DrawCell( iRow, i );
  AliHLTTPCCADisplay::Instance().Ask();  

  Int_t nConnectedCells = 0;

  cout<<"draw merged cells..."<<endl;

  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      {
	AliHLTTPCCACell &c = fRows[iRow].Cells()[i];
	Int_t id = c.Link();
	if( id<0 ) continue;	
 	AliHLTTPCCADisplay::Instance().ConnectCells( iRow,c,ID2IRow(id),ID2Cell(id) );
	nConnectedCells++;
      }
  if( nConnectedCells>0 ){
    AliHLTTPCCADisplay::Instance().Ask();  
  }

  
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().DrawSlice( this );
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      AliHLTTPCCADisplay::Instance().DrawCell( iRow, i );  
  
  cout<<"draw initial tracks"<<endl;
  
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    AliHLTTPCCATrack &iTrack = fTracks[itr];
    if( iTrack.NCells()<3 ) continue;
    if( !iTrack.Alive() ) continue;
    AliHLTTPCCADisplay::Instance().DrawTrack( iTrack );
  }
  //if( fNTracks>0 ) 
  AliHLTTPCCADisplay::Instance().Ask();
#endif 

  TStopwatch timer4;

  Bool_t doMerging=1;//SG!!!
  
  Float_t factor2 = fParam.TrackConnectionFactor()*fParam.TrackConnectionFactor();
  Int_t *refEndPoints = new Int_t[fNHitsTotal];
  Int_t nRefEndPoints = 0;
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ ){
    AliHLTTPCCARow &irow = fRows[iRow];
    for (Int_t iPoint = 0; iPoint<irow.NEndPoints(); iPoint++){ 
      refEndPoints[nRefEndPoints++] = IRowICell2ID(iRow,iPoint);
    }
  }
  bool first = 1;
  while( doMerging ){

    //cout<<"do merging"<<endl;
    
    doMerging = 0;

    // find nejghbouring tracks
    TStopwatch timer5;
    //cout<<"nRefEndPoints = "<<nRefEndPoints<<endl;

    for( Int_t iRef=0; iRef<nRefEndPoints; iRef++ ){
      Int_t iRow = ID2IRow(refEndPoints[iRef]);
      AliHLTTPCCARow &irow = fRows[iRow];
      Int_t iPoint = ID2ICell(refEndPoints[iRef]);      
      AliHLTTPCCAEndPoint &ip  = irow.EndPoints()[iPoint];
      if( ip.TrackID()<0 ) continue; // not active endpoint

      AliHLTTPCCATrack &iTrack = fTracks[ip.TrackID()];
      if( iTrack.NCells()<3 ) continue;
      
      Int_t jRowMin = iRow - fParam.MaxTrackMatchDRow();
      Int_t jRowMax = iRow + fParam.MaxTrackMatchDRow();
      if( jRowMin<0 ) jRowMin = 0;
      if( jRowMax>=fParam.NRows() ) jRowMax = fParam.NRows()-1;

      if( ip.Param().CosPhi()>=0 ){ //TMath::Abs([2])<TMath::Pi()/2. ){
	jRowMin = iRow;
      }else{
	jRowMax = iRow;
      }

      Int_t bestNeighbourN = -1;
      Float_t bestDist2 = 1.e20;
      Int_t bestLink = -1;
      if( ip.Link()>=0 ){ 
	bestNeighbourN = fTracks[ID2Point(ip.Link()).TrackID()].NCells();
      }
      
      Float_t y0 = ip.Param().GetY();
      Float_t z0 = ip.Param().GetZ();

 #ifdef DRAW      
      //AliHLTTPCCADisplay::Instance().DrawTrackletPoint(ip.Param(),kBlue);
      //AliHLTTPCCADisplay::Instance().Ask();
#endif
     
      for( Int_t jRow = jRowMin; jRow<=jRowMax; jRow++){
	AliHLTTPCCARow &jrow = fRows[jRow];
	Float_t dx2 = (irow.X()-jrow.X())*(irow.X()-jrow.X());

	// extrapolate the track to row jRow

	//for( int ii=0; ii<10; ii++){
	//AliHLTTPCCATrackParam iPar = ip.Param();
	//iPar.TransportToX( jrow.X());
	//}
	AliHLTTPCCATrackParam iPar = ip.Param();
	Bool_t ok = iPar.TransportToX( jrow.X());
	if( !ok ) continue;
#ifdef DRAW
	//AliHLTTPCCADisplay::Instance().DrawTrackletPoint(iPar, kBlack);
	//AliHLTTPCCADisplay::Instance().Ask();
#endif
	Float_t zMin = iPar.GetZ() - 3.5*TMath::Sqrt(iPar.GetErr2Z()) - .2*3.5;
	AliHLTTPCCAEndPoint *jjp = lower_bound(jrow.EndPoints(),jrow.EndPoints()+jrow.NEndPoints(),zMin,AliHLTTPCCARow::CompareEndPointZ);
	
	for (Int_t jPoint = jjp-jrow.EndPoints(); jPoint<jrow.NEndPoints(); jPoint++){ 

	  AliHLTTPCCAEndPoint &jp  = jrow.EndPoints()[jPoint];
	  
	  if( jp.TrackID()<0 ) continue; // endpoint not active
	  if( jp.TrackID()==ip.TrackID() ) continue; // same track
	  
	  AliHLTTPCCATrack &jTrack = fTracks[jp.TrackID()];
	  
	  if( bestNeighbourN > jTrack.NCells() ){
	    continue; // there is already better neighbour found
	  }
	  if( jp.Link()>=0 &&
	      fTracks[ID2Point(jp.Link()).TrackID()].NCells()>=iTrack.NCells() ){
	    continue; // jTrack is already linked to a better track or to iTrack
	  }

	  Float_t dy2, dz2;
	  AliHLTTPCCATrackParam &jPar = jp.Param();

	  // check direction
	  {
	    if( jPar.GetCosPhi()*iPar.GetCosPhi()>=0 ) continue;
	  }	  
	  // check for neighbouring
	  {
	    float d = jPar.GetY() - iPar.GetY();
	    float s2 = jPar.GetErr2Y() + iPar.GetErr2Y();
	    if( d*d>factor2*s2 ){
	      continue;
	    }
	    //cout<<"\ndy="<<TMath::Sqrt(d*d)<<", err="<<TMath::Sqrt(factor2*s2)<<endl;
	    d = jPar.GetZ() - iPar.GetZ();
	    s2 = jPar.GetErr2Z() + iPar.GetErr2Z();
	    if( d*d>factor2*s2 ){
	      if( d>0 ) break;
	      continue;
	    }
	    //cout<<"dz="<<TMath::Sqrt(d*d)<<", err="<<TMath::Sqrt(factor2*s2)<<endl;
	    if( iTrack.NCells()>=3 && jTrack.NCells()>=3 ){
	      d = jPar.GetSinPhi() + iPar.GetSinPhi(); //! phi sign is different
	      s2 = jPar.GetErr2SinPhi() + iPar.GetErr2SinPhi();
	      if( d*d>factor2*s2 ) continue;
	      //cout<<"dphi="<<TMath::Sqrt(d*d)<<", err="<<TMath::Sqrt(factor2*s2)<<endl;
	      d = jPar.GetKappa() + iPar.GetKappa(); // ! kappa sign iz different
	      s2 = jPar.GetErr2Kappa() + iPar.GetErr2Kappa();
	      if( d*d>factor2*s2 ) continue;
	      //cout<<"dk="<<TMath::Sqrt(d*d)<<", err="<<TMath::Sqrt(factor2*s2)<<endl;
	      d = jPar.GetDzDs() + iPar.GetDzDs(); // ! DzDs sign iz different
	      s2 = jPar.GetErr2DzDs() + iPar.GetErr2DzDs();
	      if( d*d>factor2*s2 ) continue;
	      //cout<<"dlam="<<TMath::Sqrt(d*d)<<", err="<<TMath::Sqrt(factor2*s2)<<endl;
	    }
	  }
	  
	  Float_t dy = jPar.GetY() - y0;
	  dy2 = dy*dy;
	  Float_t dz = jPar.GetZ() - z0;
	  dz2 = dz*dz;
	  Float_t dist2 = dx2+dy2+dz2;

	  if( ( bestNeighbourN == jTrack.NCells() ) && dist2>bestDist2 ) continue;
	  
	  // tracks can be matched
	  
	  bestLink = IRowICell2ID( jRow, jPoint );
	  bestNeighbourN = jTrack.NCells();
	  bestDist2 = dist2;
	}
      }
		
      if( bestLink < 0 ) continue; // no neighbours found

      AliHLTTPCCAEndPoint &jp = ID2Point(bestLink);
      
      if( ip.Link()>=0 ){ // break existing link of iTrack
	ID2Point(ip.Link()).Link() = -1;
      }
      if( jp.Link()>=0 ){ // break existing link of jTrack
	ID2Point(jp.Link()).Link() = -1;
      }
      ip.Link() = bestLink;
      jp.Link()= IRowICell2ID( iRow, iPoint );
      
      //cout<<"create link ("<<jp.Link()<<","<<ip.TrackID()<<")->("<<ip.Link()<<","<<jp.TrackID()<<")"<<endl;
      
    }

    timer5.Stop();
    if(first) fTimers[5] += timer5.CpuTime();

    //cout<<"merge neighbours"<<endl;
    // merge neighbours

    TStopwatch timer6;

    Int_t nRefEndPointsNew = 0;
    for( Int_t iRef=0; iRef<nRefEndPoints; iRef++ ){

      Int_t iRow = ID2IRow(refEndPoints[iRef]);
      Int_t iPoint = ID2ICell(refEndPoints[iRef]);
      AliHLTTPCCARow &irow = fRows[iRow];
      AliHLTTPCCAEndPoint &ip  = irow.EndPoints()[iPoint];
      if( ip.TrackID()<0 ) continue; // not active endpoint 
      if( ip.Link()<0 ) continue; // no neighbours found

      Int_t ipID = IRowICell2ID(iRow,iPoint);
      Int_t jpID = ip.Link();
      AliHLTTPCCAEndPoint &jp  = ID2Point(jpID);
      
      if( jp.Link()!=ipID ){
	//cout<<"broken link: jp.Link()!=iID"<<endl;
	//exit(0);
	return;
      }
      if( jp.TrackID()<0 ){ 
	//cout<<"broken link: jp.TrackID()<=0"<<endl;
	//exit(0);
	return;
      }
      if( jp.TrackID()==ip.TrackID() ){
	//cout<<"broken link: jp.TrackID()==ip.TrackID()"<<endl;
	//exit(0);	  
	return;
      }
     
      //cout<<"Merge neighbours ("<<ipID<<","<<ip.TrackID()<<")->("<<jpID<<","<<jp.TrackID()<<")"<<endl;

      AliHLTTPCCATrack &iTrack = fTracks[ip.TrackID()];
      AliHLTTPCCATrack &jTrack = fTracks[jp.TrackID()];

      // rotate cell link direction for jTrack if necessary
      
      Int_t icID = ip.CellID();
      Int_t jcID = jp.CellID();
      AliHLTTPCCACell &ic  = ID2Cell(icID);
      AliHLTTPCCACell &jc  = ID2Cell(jcID);
      
      if( ( ic.Link()<0 && jc.Link()<0 ) || ( ic.Link()>=0 && jc.Link()>=0 ) ){

	Int_t currID =  jTrack.CellID()[0];
	jTrack.CellID()[0] = jTrack.CellID()[2];
	jTrack.CellID()[2] = currID;

	Int_t pID =  jTrack.PointID()[0];
	jTrack.PointID()[0] = jTrack.PointID()[1];
	jTrack.PointID()[1] = pID;

	currID = jTrack.FirstCellID();
	Int_t lastID = -1;
	while( currID>=0 ){
	  AliHLTTPCCACell &c = ID2Cell( currID );
	  Int_t nextID = c.Link();
	  c.Link() = lastID;
	  lastID = currID;
	  currID = nextID;
	}
	jTrack.FirstCellID() = lastID;	
      }
      //cout<<"track i "<<ip.TrackID()<<", points="<<ipID<<", "<<iTrack.PointID()[0]<<", "<<iTrack.PointID()[1]<<endl;
      //cout<<"track j "<<jp.TrackID()<<", points="<<jpID<<", "<<jTrack.PointID()[0]<<", "<<jTrack.PointID()[1]<<endl;
      Int_t itr = ip.TrackID();
      ip.TrackID() = -1;
      jp.TrackID() = -1;
      ip.Link()  = -1;
      jp.Link()  = -1;
      jTrack.Alive() = 0;
      
      //cout<<"iTrack ID: "<<iTrack.CellID()[0]<<" "<<iTrack.CellID()[1]<<" "<<iTrack.CellID()[2]<<endl;
      //cout<<"jTrack ID: "<<jTrack.CellID()[0]<<" "<<jTrack.CellID()[1]<<" "<<jTrack.CellID()[2]<<endl;
      if( ic.Link()<0 ){ //match iTrack->jTrack
	ic.Link() = jcID;
	iTrack.PointID()[1] = jTrack.PointID()[1];
	ID2Point(iTrack.PointID()[1]).TrackID() = itr;
	if( jTrack.NCells()<3 ){
	  refEndPoints[nRefEndPointsNew++] = iTrack.PointID()[1];
	  doMerging = 1;
	  ID2Point(iTrack.PointID()[1]).Param() = ip.Param();// just to set phi direction
	}
	if( iTrack.NCells()<3 ){
	  refEndPoints[nRefEndPointsNew++] = iTrack.PointID()[0];
	  doMerging = 1;
	  ID2Point(iTrack.PointID()[0]).Param() = jp.Param();// just to set phi direction
	}

	if( TMath::Abs(ID2Cell(jTrack.CellID()[2]).Z()-ID2Cell(iTrack.CellID()[0]).Z())>
	    TMath::Abs(ID2Cell(iTrack.CellID()[2]).Z()-ID2Cell(iTrack.CellID()[0]).Z())  ){
	  iTrack.CellID()[2] = jTrack.CellID()[2];
	}
      }else{ //match jTrack->iTrack
	jc.Link() = icID;
	iTrack.FirstCellID()=jTrack.FirstCellID();
	iTrack.PointID()[0] = jTrack.PointID()[0];
	ID2Point(iTrack.PointID()[0]).TrackID() = itr;
	if( jTrack.NCells()<3 ){
	  refEndPoints[nRefEndPointsNew++] = iTrack.PointID()[0];
	  doMerging = 1;
	  ID2Point(iTrack.PointID()[0]).Param() = ip.Param(); // just to set phi direction
	}
	if( iTrack.NCells()<3 ){
	  refEndPoints[nRefEndPointsNew++] = iTrack.PointID()[1];
	  doMerging = 1;
	  ID2Point(iTrack.PointID()[1]).Param() = jp.Param();// just to set phi direction
	}
	if( TMath::Abs(ID2Cell(jTrack.CellID()[0]).Z()-ID2Cell(iTrack.CellID()[2]).Z())>
	    TMath::Abs(ID2Cell(iTrack.CellID()[0]).Z()-ID2Cell(iTrack.CellID()[2]).Z())  ){
	  iTrack.CellID()[0] = jTrack.CellID()[0];
	}
      }

      //cout<<"merged ID: "<<iTrack.CellID()[0]<<" "<<iTrack.CellID()[1]<<" "<<iTrack.CellID()[2]<<endl;

      if( jTrack.NCells()>iTrack.NCells() ){
	iTrack.CellID()[1] = jTrack.CellID()[1];
      }
      
      AliHLTTPCCAEndPoint &p0 = ID2Point(iTrack.PointID()[0]);
      AliHLTTPCCAEndPoint &p1 = ID2Point(iTrack.PointID()[1]);
      
      if( p0.Link() == iTrack.PointID()[1] ){
	p0.Link() = -1;
	p1.Link() = -1;
      }
      //cout<<" NCells itr/jtr= "<<iTrack.NCells()<<" "<<jTrack.NCells()<<endl;
      //cout<<" fit merged track "<<itr<<", NCells="<<iTrack.NCells()<<endl;
      Float_t *t0 = ( jTrack.NCells()>iTrack.NCells() ) ?jp.Param().Par() :ip.Param().Par();            
      iTrack.NCells()+=jTrack.NCells();      
      FitTrack(iTrack,t0);

#ifdef DRAW
      cout<<"merged points..."<<ipID<<"/"<<jpID<<endl;
      //AliHLTTPCCADisplay::Instance().ConnectCells( iRow,ic,ID2IRow(jcID),jc,kRed );
      AliHLTTPCCADisplay::Instance().ConnectEndPoints( ipID,jpID,1.,2,kRed );
      AliHLTTPCCADisplay::Instance().DrawEndPoint( ipID,1.,2,kRed );
      AliHLTTPCCADisplay::Instance().DrawEndPoint( jpID,1.,2,kRed );
      AliHLTTPCCADisplay::Instance().Ask();
      cout<<"merged track"<<endl;
      AliHLTTPCCADisplay::Instance().DrawTrack(iTrack);
      AliHLTTPCCADisplay::Instance().Ask();
#endif
      /*
      static int ntr=0;
      if( ntr++==1 ){
	doMerging = 0;
	break;
      }
      */
      //doMerging = 1;    
    }

    timer6.Stop();  
    if(first)fTimers[6] += timer6.CpuTime();

    nRefEndPoints = nRefEndPointsNew;

    //cout<<"merging ok"<<endl;
    //first = 0;
  }// do merging
 
  delete[] refEndPoints;
  timer4.Stop();  
  fTimers[4] = timer4.CpuTime();

#ifdef DRAW
  //if( fNTracks>0 ) AliHLTTPCCADisplay::Instance().Ask();
#endif 




#ifdef DRAWXX
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().DrawSlice( this );
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      AliHLTTPCCADisplay::Instance().DrawCell( iRow, i );  
  
  cout<<"draw final tracks"<<endl;
  
  for( Int_t itr=0; itr<fNTracks; itr++ ){
    AliHLTTPCCATrack &iTrack = fTracks[itr];
    if( iTrack.NCells()<3 ) continue;
    if( !iTrack.Alive() ) continue;
    AliHLTTPCCADisplay::Instance().DrawTrack( iTrack );
    cout<<"final track "<<itr<<", ncells="<<iTrack.NCells()<<endl;
    AliHLTTPCCADisplay::Instance().Ask();
  }
  AliHLTTPCCADisplay::Instance().Ask();
#endif

  // write output
  TStopwatch timer7;

  //cout<<"write output"<<endl;
#ifdef DRAW
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().DrawSlice( this );
  for( Int_t iRow=0; iRow<fParam.NRows(); iRow++ )
    for (Int_t i = 0; i<fRows[iRow].NCells(); i++) 
      AliHLTTPCCADisplay::Instance().DrawCell( iRow, i );  
  
  cout<<"draw out tracks"<<endl;
#endif

  fOutTrackHits = new Int_t[fNHitsTotal];
  fOutTracks = new AliHLTTPCCAOutTrack[fNTracks];
  fNOutTrackHits = 0;
  fNOutTracks = 0;

  for( Int_t iTr=0; iTr<fNTracks; iTr++){
    AliHLTTPCCATrack &iTrack = fTracks[iTr];
    if( !iTrack.Alive() ) continue;
    if( iTrack.NCells()<3 ) continue;      
    AliHLTTPCCAOutTrack &out = fOutTracks[fNOutTracks];
    out.FirstHitRef() = fNOutTrackHits;
    out.NHits() = 0;
    out.OrigTrackID() = iTr;
    {
      AliHLTTPCCAEndPoint &p0 = ID2Point(iTrack.PointID()[0]);	
      AliHLTTPCCAEndPoint &p2 = ID2Point(iTrack.PointID()[1]);	
      out.StartPoint() = p0.Param();
      out.EndPoint() = p2.Param();
    }
    AliHLTTPCCATrackParam &t = out.StartPoint();//SG!!!
    AliHLTTPCCATrackParam t0 = t;

    t.Chi2() = 0;
    t.NDF() = -5;	
    first = 1; // removed  bool in front  -> JT

    Int_t iID = iTrack.FirstCellID();
    Int_t fNOutTrackHitsOld = fNOutTrackHits;
    for( AliHLTTPCCACell *ic = &ID2Cell(iID); ic->Link()>=0; iID = ic->Link(), ic = &ID2Cell(iID) ){
      //cout<<"itr="<<iTr<<", cell ="<<ID2IRow(iID)<<" "<<ID2ICell(iID)<<endl;
      AliHLTTPCCARow &row = ID2Row(iID);
      if( !t0.TransportToX( row.X() ) ) continue;
      Int_t jHit = -1;
      Float_t dy, dz, d = 1.e10;
      for( Int_t iHit=0; iHit<ic->NHits(); iHit++ ){
	AliHLTTPCCAHit &h = row.GetCellHit(*ic,iHit);

	// check for wrong hits	
	{
	  Float_t ddy = t0.GetY() - h.Y();
	  Float_t ddz = t0.GetZ() - h.Z();
	  Float_t dd = ddy*ddy+ddz*ddz;
	  if( dd<d ){
	    d = dd;
	    dy = ddy;
	    dz = ddz;
	    jHit = iHit;
	  }
	}
      }
      if( jHit<0 ) continue;
      AliHLTTPCCAHit &h = row.GetCellHit(*ic,jHit);
      //if( dy*dy > 3.5*3.5*(/*t0.GetErr2Y() + */h.ErrY()*h.ErrY() ) ) continue;//SG!!!
      //if( dz*dz > 3.5*3.5*(/*t0.GetErr2Z() + */h.ErrZ()*h.ErrZ() ) ) continue;
      //if( !t0.Filter2( h.Y(), h.Z(), h.ErrY()*h.ErrY(), h.ErrZ()*h.ErrZ() ) ) continue;
 
      
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

      if( t.Filter2( h.Y(), h.Z(), h.ErrY()*h.ErrY(), h.ErrZ()*h.ErrZ() ) ) first = 0;
      else continue;

      fOutTrackHits[fNOutTrackHits] = h.ID();
      fNOutTrackHits++;
      if( fNOutTrackHits>fNHitsTotal ){
	cout<<"fNOutTrackHits>fNHitsTotal"<<endl;
	exit(0);//SG!!!
	return;
      }
      out.NHits()++;
    }
    //cout<<fNOutTracks<<": itr = "<<iTr<<", n outhits = "<<out.NHits()<<endl;
    if( out.NHits() > 3 ){
      fNOutTracks++;
#ifdef DRAW
      AliHLTTPCCADisplay::Instance().DrawTrack( iTrack );
      cout<<"out track "<<(fNOutTracks-1)<<", orig = "<<iTr<<", nhits="<<out.NHits()<<endl;
      AliHLTTPCCADisplay::Instance().Ask();    
#endif
      
    }else {
      fNOutTrackHits = fNOutTrackHitsOld;
    }
  }
  //cout<<"end writing"<<endl;
  timer7.Stop();  
  fTimers[7] = timer7.CpuTime();

#ifdef DRAW
  AliHLTTPCCADisplay::Instance().Ask();
  //AliHLTTPCCADisplay::Instance().DrawMCTracks(fParam.fISec);
  //AliHLTTPCCADisplay::Instance().Update();
  //AliHLTTPCCADisplay::Instance().Ask();
#endif 
}


void AliHLTTPCCATracker::FitTrack( AliHLTTPCCATrack &track, Float_t t0[] ) const 
{      
  //* Fit the track 

  AliHLTTPCCAEndPoint &p0 = ID2Point(track.PointID()[0]);	
  AliHLTTPCCAEndPoint &p2 = ID2Point(track.PointID()[1]);	
  AliHLTTPCCACell &c0 = ID2Cell(p0.CellID());	
  AliHLTTPCCACell &c1 = ID2Cell(track.CellID()[1]);	
  AliHLTTPCCACell &c2 = ID2Cell(p2.CellID());	
  AliHLTTPCCARow &row0 = ID2Row(p0.CellID());
  AliHLTTPCCARow &row1 = ID2Row(track.CellID()[1]);
  AliHLTTPCCARow &row2 = ID2Row(p2.CellID());


  Float_t sp0[5] = {row0.X(), c0.Y(), c0.Z(), c0.ErrY(), c0.ErrZ() };
  Float_t sp1[5] = {row1.X(), c1.Y(), c1.Z(), c1.ErrY(), c1.ErrZ() };
  Float_t sp2[5] = {row2.X(), c2.Y(), c2.Z(), c2.ErrY(), c2.ErrZ() };
  if( track.NCells()>=3 ){
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
}
