// $Id$
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

#include "AliHLTTPCCADisplay.h"

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAEndPoint.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"

//#include "TString.h"
#include "Riostream.h"
#include "TMath.h"
#include "TStyle.h"
#include "TCanvas.h"

ClassImp(AliHLTTPCCADisplay)

AliHLTTPCCADisplay &AliHLTTPCCADisplay::Instance()
{
  // reference to static object
  static AliHLTTPCCADisplay gAliHLTTPCCADisplay;
  return gAliHLTTPCCADisplay; 
}

AliHLTTPCCADisplay::AliHLTTPCCADisplay() : TObject(), fYX(0), fZX(0), fAsk(1), fSliceView(1), fSlice(0), 
					   fCos(1), fSin(0), fZMin(-250), fZMax(250),fSliceCos(1), fSliceSin(0),
					   fRInnerMin(83.65), fRInnerMax(133.3), fROuterMin(133.5), fROuterMax(247.7),
					   fTPCZMin(-250.), fTPCZMax(250), fArc(), fLine(), fPLine(), fMarker(), fBox(), fCrown(), fLatex()
{
  // constructor
} 


AliHLTTPCCADisplay::AliHLTTPCCADisplay( const AliHLTTPCCADisplay& ) 
  : TObject(), fYX(0), fZX(0), fAsk(1), fSliceView(1), fSlice(0), 
    fCos(1), fSin(0), fZMin(-250), fZMax(250),fSliceCos(1), fSliceSin(0),
    fRInnerMin(83.65), fRInnerMax(133.3), fROuterMin(133.5), fROuterMax(247.7),
    fTPCZMin(-250.), fTPCZMax(250), fArc(), fLine(), fPLine(), fMarker(), fBox(), fCrown(), fLatex()
{
  // dummy
}

AliHLTTPCCADisplay& AliHLTTPCCADisplay::operator=( const AliHLTTPCCADisplay& )
{
  // dummy
  return *this;
}
 
AliHLTTPCCADisplay::~AliHLTTPCCADisplay()
{
  // destructor
  delete fYX; 
  delete fZX;
}

void AliHLTTPCCADisplay::Init()
{
  // initialization
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasBorderSize(1);
  gStyle->SetCanvasColor(0);
  fYX = new TCanvas ("YX", "YX window", -1, 0, 600, 600);
  fZX = new TCanvas ("ZX", "ZX window", -610, 0, 590, 600);  
  fMarker = TMarker(0.0, 0.0, 20);//6);
}

void AliHLTTPCCADisplay::Update()
{
  // update windows
  if( !fAsk ) return;
  fYX->Update();
  fZX->Update();
}

void AliHLTTPCCADisplay::ClearView()
{
  // clear windows
  fYX->Clear();
  fZX->Clear();
}

void AliHLTTPCCADisplay::Ask()
{
  // whait for the pressed key, when "r" pressed, don't ask anymore
  char symbol;
  if (fAsk){
    Update();
    std::cout<<"ask> "<<std::endl;
    do{
      std::cin.get(symbol);
      if (symbol == 'r')
	fAsk = false;
    } while (symbol != '\n');
  }
}


void AliHLTTPCCADisplay::SetSliceView()
{
  // switch to slice view
  fSliceView = 1;
}

void AliHLTTPCCADisplay::SetTPCView()
{
  // switch to full TPC view
  fSliceView = 0;
  fCos = 1;
  fSin = 0;
  fZMin = fTPCZMin;
  fZMax = fTPCZMax;
}

void AliHLTTPCCADisplay::SetCurrentSlice( AliHLTTPCCATracker *slice )
{
  // set reference to the current CA tracker, and read the current slice geometry
  fSlice = slice;
  SetSliceTransform( slice );
  if( fSliceView ){
    fCos = slice->Param().SinAlpha();
    fSin = slice->Param().CosAlpha();
    fZMin = slice->Param().ZMin();
    fZMax = slice->Param().ZMax();
    ClearView();
    Double_t r0 = .5*(slice->Param().RMax()+slice->Param().RMin());
    Double_t dr = .5*(slice->Param().RMax()-slice->Param().RMin());
    Double_t cx = 0;
    Double_t cy = r0;    
    Double_t cz = .5*(slice->Param().ZMax()+slice->Param().ZMin());
    Double_t dz = .5*(slice->Param().ZMax()-slice->Param().ZMin())*1.2;
    fYX->Range(cx-dr, cy-dr*1.05, cx+dr, cy+dr);
    fZX->Range(cz-dz, cy-dr*1.05, cz+dz, cy+dr);

    //fYX->Range(cx-dr/6, cy-dr, cx+dr/8, cy-dr + dr/8);
    //fZX->Range(cz+dz/2+dz/6, cy-dr , cz+dz-dz/8, cy-dr + dr/8);

    //fYX->Range(cx-dr/3, cy-dr/3, cx+dr, cy+dr);
    //fZX->Range(cz-dz/3, cy-dr/3, cz+dz, cy+dr);//+dr);
    //fYX->Range(cx-dr/3, cy-dr/2*1.3, cx+dr/3, cy-dr/2*1.1);//+dr);
    //fZX->Range(cz-dz*0.65, cy-dr/2*1.3, cz+dz*0-dz*0.55, cy-dr/2*1.1);//+dr);
   }
}

void AliHLTTPCCADisplay::Set2Slices( AliHLTTPCCATracker *slice )
{
  //* Set view for two neighbouring slices

  fSlice = slice;
  fSliceView = 0;
  fCos = TMath::Cos(TMath::Pi()/2 - (slice->Param().Alpha()+10./180.*TMath::Pi()));
  fSin = TMath::Sin(TMath::Pi()/2 - (slice->Param().Alpha()+10./180.*TMath::Pi()));
  fZMin = slice->Param().ZMin();
  fZMax = slice->Param().ZMax();
  ClearView();
  Double_t r0 = .5*(slice->Param().RMax()+slice->Param().RMin());
  Double_t dr = .5*(slice->Param().RMax()-slice->Param().RMin());
  Double_t cx = 0;
  Double_t cy = r0;    
  fYX->Range(cx-1.3*dr, cy-1.1*dr, cx+1.3*dr, cy+1.1*dr);
  fYX->cd();
  Int_t islice = slice->Param().ISlice();
  Int_t jslice = slice->Param().ISlice()+1;
  if( islice==17 ) jslice = 0;
  else if( islice==35 ) jslice = 18;
  fLatex.DrawLatex(cx-1.3*dr+1.3*dr*.05,cy-dr+dr*.05, Form("YX, Slices %2i/%2i",islice,jslice));
  Double_t cz = .5*(slice->Param().ZMax()+slice->Param().ZMin());
  Double_t dz = .5*(slice->Param().ZMax()-slice->Param().ZMin())*1.2;
  fZX->Range(cz-dz, cy-1.1*dr, cz+dz, cy+1.1*dr);//+dr);
  fZX->cd();  
  fLatex.DrawLatex(cz-dz+dz*.05,cy-dr+dr*.05, Form("ZX, Slices %2i/%2i",islice,jslice));
}


Int_t AliHLTTPCCADisplay::GetColor( Double_t z ) const 
{
  // Get color with respect to Z coordinate
  const Color_t kMyColor[11] = { kGreen, kBlue, kYellow, kMagenta, kCyan, 
				 kOrange, kSpring, kTeal, kAzure, kViolet, kPink };

  Double_t zz = (z-fZMin)/(fZMax-fZMin);    
  Int_t iz = (int) (zz*11);
  if( iz<0 ) iz = 0;
  if( iz>10 ) iz = 10;
  return kMyColor[iz];
}

void AliHLTTPCCADisplay::Global2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const
{
  // convert coordinates global->view
  *xv = x*fCos + y*fSin;
  *yv = y*fCos - x*fSin;
}

void AliHLTTPCCADisplay::SetSliceTransform( Double_t alpha )
{
  fSliceCos = TMath::Cos( alpha );
  fSliceSin = TMath::Sin( alpha );
} 

void AliHLTTPCCADisplay::SetSliceTransform( AliHLTTPCCATracker *slice )
{
  SetSliceTransform(slice->Param().Alpha());
}

void AliHLTTPCCADisplay::Slice2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const
{
  // convert coordinates slice->view
  Double_t xg = x*fSliceCos - y*fSliceSin;
  Double_t yg = y*fSliceCos + x*fSliceSin;
  *xv = xg*fCos - yg*fSin;
  *yv = yg*fCos + xg*fSin;
}


void AliHLTTPCCADisplay::DrawTPC()
{
  // schematically draw TPC detector
  fYX->Range(-fROuterMax, -fROuterMax, fROuterMax, fROuterMax);
  fYX->Clear();
  {
    fArc.SetLineColor(kBlack);
    fArc.SetFillStyle(0);
    fYX->cd();    
    for( Int_t iSlice=0; iSlice<18; iSlice++){
      fCrown.SetLineColor(kBlack);
      fCrown.SetFillStyle(0);
      fCrown.DrawCrown(0,0,fRInnerMin, fRInnerMax, 360./18.*iSlice, 360./18.*(iSlice+1) );
      fCrown.DrawCrown(0,0,fROuterMin, fROuterMax, 360./18.*iSlice, 360./18.*(iSlice+1) );
    }
  }
  fZX->cd();
  fZX->Range( fTPCZMin, -fROuterMax, fTPCZMax, fROuterMax );
  fZX->Clear();
}

void AliHLTTPCCADisplay::DrawSlice( AliHLTTPCCATracker *slice )
{     
  // draw current the TPC slice
  fYX->cd();
  Double_t r0 = .5*(slice->Param().RMax()+slice->Param().RMin());
  Double_t dr = .5*(slice->Param().RMax()-slice->Param().RMin());
  Double_t cx = r0*slice->Param().CosAlpha();
  Double_t cy = r0*slice->Param().SinAlpha();
  Double_t raddeg = 180./3.1415;
  Double_t a0 = raddeg*.5*(slice->Param().AngleMax() + slice->Param().AngleMin());
  Double_t da = raddeg*.5*(slice->Param().AngleMax() - slice->Param().AngleMin());
  if( fSliceView ){
    cx = 0; cy = r0;
    a0 = 90.;
    fLatex.DrawLatex(cx-dr+dr*.05,cy-dr+dr*.05, Form("YX, Slice %2i",slice->Param().ISlice()));
  } else {
    a0+= raddeg*TMath::ATan2(fSin, fCos );
  }
  fArc.SetLineColor(kBlack);
  fArc.SetFillStyle(0);     
  fCrown.SetLineColor(kBlack);
  fCrown.SetFillStyle(0);
    
  fCrown.DrawCrown(0,0, slice->Param().RMin(),slice->Param().RMax(), a0-da, a0+da );

  fLine.SetLineColor(kBlack);
 
  fZX->cd();

  Double_t cz = .5*(slice->Param().ZMax()+slice->Param().ZMin());
  Double_t dz = .5*(slice->Param().ZMax()-slice->Param().ZMin())*1.2;
  //fLine.DrawLine(cz+dz, cy-dr, cz+dz, cy+dr ); 
  if( fSliceView ) fLatex.DrawLatex(cz-dz+dz*.05,cy-dr+dr*.05, Form("ZX, Slice %2i",slice->Param().ISlice()));
}


void AliHLTTPCCADisplay::DrawHit( Int_t iRow, Int_t iHit, Int_t color )
{
  // draw hit
  if( !fSlice ) return;
  AliHLTTPCCARow &row = fSlice->Rows()[iRow];
  AliHLTTPCCAHit *h = &(row.Hits()[iHit]);
  if( color<0 ) color = GetColor( h->Z() );

  //Double_t dgy = 3.5*TMath::Abs(h->ErrY()*fSlice->Param().CosAlpha() - fSlice->Param().ErrX()*fSlice->Param().SinAlpha() );
  Double_t dx = 0.1;//fSlice->Param().ErrX()*TMath::Sqrt(12.)/2.;
  Double_t dy = h->ErrY()*3.5;
  //Double_t dz = h->ErrZ()*3.5;
  fMarker.SetMarkerSize(.3);
  fMarker.SetMarkerColor(color);
  fArc.SetLineColor(color);
  fArc.SetFillStyle(0);
  Double_t vx, vy, dvx, dvy;
  Slice2View( row.X(), h->Y(), &vx, &vy );
  Slice2View( dx, dy, &dvx, &dvy );
  
  fYX->cd();
  //if( fSliceView ) fArc.DrawEllipse( vx, vy, dvx, dvy, 0,360, 0);
  //else  fArc.DrawEllipse( vx, vy, dx, dy, 0,360, fSlice->Param().Alpha()*180./3.1415);
  fMarker.DrawMarker(vx, vy);
  fZX->cd();
  //if( fSliceView ) fArc.DrawEllipse( h->Z(), vy, dz, dvy, 0,360, 0 );
  //else fArc.DrawEllipse( h->Z(), vy, dz, dgy, 0,360, fSlice->Param().Alpha()*180./3.1415);
  fMarker.DrawMarker(h->Z(), vy); 
}

void AliHLTTPCCADisplay::DrawCell( Int_t iRow, AliHLTTPCCACell &cell, Int_t width, Int_t color )
{
  // draw cell
  AliHLTTPCCARow &row = fSlice->Rows()[iRow];
  Double_t vx, vy, vdx, vdy, vz = cell.Z(), vdz = cell.ErrZ()*3.5;
  Slice2View(row.X(), cell.Y(), &vx, &vy);
  Slice2View(0.2, cell.ErrY()*3.5, &vdx, &vdy);
  if( color<0 ) color = GetColor(cell.Z());
  fLine.SetLineColor(color);
  fLine.SetLineWidth(width);
  fArc.SetLineColor(color);
  fArc.SetFillStyle(0);
  fYX->cd();
  //fLine.DrawLine(vx-vdx,vy-vdy, vx+vdx, vy+vdy );
  fArc.DrawEllipse(vx, vy, vdx, vdy, 0,360, 0);
  fZX->cd();
  //fLine.DrawLine(cell.Z()-3*cell.ErrZ(),vy-vdy, cell.Z()+3*cell.ErrZ(), vy+vdy ); 
  fArc.DrawEllipse(vz, vy, vdz, vdy, 0,360, 0);
  fLine.SetLineWidth(1);
}

void  AliHLTTPCCADisplay::DrawCell( Int_t iRow, Int_t iCell, Int_t width, Int_t color )
{
  // draw cell
  AliHLTTPCCARow &row = fSlice->Rows()[iRow];
  DrawCell( iRow, row.Cells()[iCell], width, color );
}
 

void AliHLTTPCCADisplay::DrawEndPoint( Int_t ID, Float_t R, Int_t width, Int_t color)
{
  // draw endpoint
  if( !fSlice ) return;
  AliHLTTPCCARow &row = fSlice->ID2Row(ID);
  AliHLTTPCCAEndPoint &p = fSlice->ID2Point(ID);
  AliHLTTPCCACell &c = fSlice->ID2Cell(p.CellID());
  if( color<0 ) color = GetColor( c.Z() );

  fArc.SetLineColor(color);
  fArc.SetFillStyle(0);
  fArc.SetLineWidth(width);

  Double_t vx, vy;
  Slice2View( row.X(), c.Y(), &vx, &vy );  
  fYX->cd();
  fArc.DrawEllipse( vx, vy, R, R, 0,360, 90);  
  fZX->cd();
  fArc.DrawEllipse( c.Z(), vy, R, R, 0,360, 90);  
  fArc.SetLineWidth(1);

}
 
void AliHLTTPCCADisplay::ConnectEndPoints( Int_t iID, Int_t jID, Float_t R, Int_t width, Int_t color )
{
   // connect endpoints
  if( !fSlice ) return;
  AliHLTTPCCARow &irow = fSlice->ID2Row(iID);
  AliHLTTPCCAEndPoint &ip = fSlice->ID2Point(iID);
  AliHLTTPCCACell &ic = fSlice->ID2Cell(ip.CellID());
  AliHLTTPCCARow &jrow = fSlice->ID2Row(jID);
  AliHLTTPCCAEndPoint &jp = fSlice->ID2Point(jID);
  AliHLTTPCCACell &jc = fSlice->ID2Cell(jp.CellID());
  if( color<0 ) color = GetColor( ic.Z() );
  
  fArc.SetLineColor(color);
  fArc.SetFillStyle(0);
  fArc.SetLineWidth(width);
  fLine.SetLineWidth(width);
  fLine.SetLineColor(color);
  Double_t ivx, ivy;
  Slice2View( irow.X(), ic.Y(), &ivx, &ivy );  
  Double_t jvx, jvy;
  Slice2View( jrow.X(), jc.Y(), &jvx, &jvy );  

  fYX->cd();
  fArc.DrawEllipse( ivx, ivy, R, R, 0,360, 90);  
  fArc.DrawEllipse( jvx, jvy, R, R, 0,360, 90);  
  fLine.DrawLine(ivx, ivy,jvx, jvy);
  fZX->cd();
  fArc.DrawEllipse( ic.Z(), ivy, R, R, 0,360, 90);  
  fArc.DrawEllipse( jc.Z(), jvy, R, R, 0,360, 90);  
  fLine.DrawLine(ic.Z(), ivy, jc.Z(), jvy);
  fArc.SetLineWidth(1);
  fLine.SetLineWidth(1);
}

void AliHLTTPCCADisplay::ConnectCells( Int_t iRow1, AliHLTTPCCACell &cell1, 
				       Int_t iRow2, AliHLTTPCCACell &cell2, Int_t color )
{
  // connect two cells on display, kind of row is drawing
  AliHLTTPCCARow &row1 = fSlice->Rows()[iRow1];
  AliHLTTPCCARow &row2 = fSlice->Rows()[iRow2];

  AliHLTTPCCAHit &h11 = row1.GetCellHit(cell1,0);
  AliHLTTPCCAHit &h12 = row1.GetCellHit(cell1,cell1.NHits()-1);
  AliHLTTPCCAHit &h21 = row2.GetCellHit(cell2,0);
  AliHLTTPCCAHit &h22=  row2.GetCellHit(cell2,cell2.NHits()-1);

  Double_t x11 = row1.X();
  Double_t x12 = row1.X();
  Double_t y11 = h11.Y() - h11.ErrY()*3;
  Double_t y12 = h12.Y() + h12.ErrY()*3;
  Double_t z11 = h11.Z() - h11.ErrZ()*3;
  Double_t z12 = h12.Z() + h12.ErrZ()*3;
  Double_t x21 = row2.X();
  Double_t x22 = row2.X();
  Double_t y21 = h21.Y() - h21.ErrY()*3;
  Double_t y22 = h22.Y() + h22.ErrY()*3;
  Double_t z21 = h21.Z() - h21.ErrZ()*3;
  Double_t z22 = h22.Z() + h22.ErrZ()*3;

  Double_t vx11, vx12, vy11, vy12, vx21, vx22, vy21, vy22;

  Slice2View(x11,y11, &vx11, &vy11 );
  Slice2View(x12,y12, &vx12, &vy12 );
  Slice2View(x21,y21, &vx21, &vy21 );
  Slice2View(x22,y22, &vx22, &vy22 );

  Double_t lx[] = { vx11, vx12, vx22, vx21, vx11 };
  Double_t ly[] = { vy11, vy12, vy22, vy21, vy11 };
  Double_t lz[] = { z11, z12, z22, z21, z11 };

  if( color<0 ) color = GetColor( (z11+z12+z22+z21)/4. );
  fPLine.SetLineColor(color);    
  fPLine.SetLineWidth(1);        
  //fPLine.SetFillColor(color);
  fPLine.SetFillStyle(-1);
 
  fYX->cd();
  fPLine.DrawPolyLine(5, lx, ly );
  fZX->cd();
  fPLine.DrawPolyLine(5, lz, ly );   
  DrawCell( iRow1, cell1, 1, color );
  DrawCell( iRow2, cell2, 1, color );
}




void AliHLTTPCCADisplay::DrawTrack( AliHLTTPCCATrack &track, Int_t color, Bool_t DrawCells )
{
  // draw track

  if( track.NCells()<2 ) return;
  int width = 1;

  AliHLTTPCCADisplayTmpCell *vCells = new AliHLTTPCCADisplayTmpCell[track.NCells()];
  AliHLTTPCCATrackParam &t = fSlice->ID2Point(track.PointID()[0]).Param();

  Int_t iID = track.FirstCellID();
  {
    Int_t iCell=0;
    while( iID>=0 ){
      AliHLTTPCCACell *c = &(fSlice->ID2Cell( iID )); 
      AliHLTTPCCARow &row = fSlice->ID2Row(iID);
      vCells[iCell].ID() = iID;    
      vCells[iCell].S() = t.GetS( row.X(), c->Y() );
      vCells[iCell].Z() = c->Z();
      iCell++;
      iID = c->Link(); 
    }
  }
  sort(vCells, vCells + track.NCells(), AliHLTTPCCADisplayTmpCell::CompareCellZ );
  
  {
    AliHLTTPCCACell &c1 = fSlice->ID2Cell(vCells[0].ID());
    AliHLTTPCCACell &c2 = fSlice->ID2Cell(vCells[track.NCells()-1].ID());
    if( color<0 ) color = GetColor( (c1.Z()+c2.Z())/2. );
  }
  
  fMarker.SetMarkerColor(color);//kBlue);
  fMarker.SetMarkerSize(1.);
  /*
  for( Int_t i=0; i<3; i++){    
    AliHLTTPCCACell &c1 = fSlice->ID2Cell(track.CellID()[i]);    
    AliHLTTPCCARow &row1 = fSlice->ID2Row(track.CellID()[i]);
    Double_t vx1, vy1;
    Slice2View(row1.X(), c1.Y(), &vx1, &vy1 ); 
    fYX->cd();
    fMarker.DrawMarker(vx1,vy1);
    fZX->cd();
    fMarker.DrawMarker(c1.Z(),vy1);
  }
  */
  DrawTrackletPoint( fSlice->ID2Point(track.PointID()[0]).Param(), kBlack);//color );
  DrawTrackletPoint( fSlice->ID2Point(track.PointID()[1]).Param(), kBlack);//color );

  for( Int_t iCell=0; iCell<track.NCells()-1; iCell++ )
  {
    AliHLTTPCCACell &c1 = fSlice->ID2Cell(vCells[iCell].ID());
    AliHLTTPCCACell &c2 = fSlice->ID2Cell(vCells[iCell+1].ID());
    AliHLTTPCCARow &row1 = fSlice->ID2Row(vCells[iCell].ID());
    AliHLTTPCCARow &row2 = fSlice->ID2Row(vCells[iCell+1].ID());
    Float_t x1, y1, z1, x2, y2, z2;    
    t.GetDCAPoint( row1.X(), c1.Y(), c1.Z(), x1, y1, z1 );
    t.GetDCAPoint( row2.X(), c2.Y(), c2.Z(), x2, y2, z2 );

    //if( color<0 ) color = GetColor( (z1+z2)/2. );
    Double_t vx1, vy1, vx2, vy2;
    Slice2View(x1, y1, &vx1, &vy1 );
    Slice2View(x2, y2, &vx2, &vy2 );
    
    fLine.SetLineColor( color );
    fLine.SetLineWidth( width );
    
    Double_t x0 = t.GetX();
    Double_t y0 = t.GetY();
    Double_t sinPhi = t.GetSinPhi();
    Double_t k = t.GetKappa();
    Double_t ex = t.GetCosPhi();
    Double_t ey = sinPhi;
 
    if( TMath::Abs(k)>1.e-4 ){

      fArc.SetFillStyle(0);
      fArc.SetLineColor(color);    
      fArc.SetLineWidth(width);        
      
      fYX->cd();

      Double_t r = 1/TMath::Abs(k);
      Double_t xc = x0 -ey*(1/k);
      Double_t yc = y0 +ex*(1/k);
     
      Double_t vx, vy;
      Slice2View( xc, yc, &vx, &vy );
      
      Double_t a1 = TMath::ATan2(vy1-vy, vx1-vx)/TMath::Pi()*180.;
      Double_t a2 = TMath::ATan2(vy2-vy, vx2-vx)/TMath::Pi()*180.;
      if( a1<0 ) a1+=360;
      if( a2<0 ) a2+=360;
      if( a2<a1 ) a2+=360;
      Double_t da = TMath::Abs(a2-a1);
      if( da>360 ) da-= 360;
      if( da>180 ){
	da = a1;
	a1 = a2;
	a2 = da;
	if( a2<a1 ) a2+=360;	
      }
      fArc.DrawArc(vx,vy,r, a1,a2,"only");
      //fArc.DrawArc(vx,vy,r, 0,360,"only");
   } else {
      fYX->cd();
      fLine.DrawLine(vx1,vy1, vx2, vy2 );
    }
  }

  for( Int_t iCell=0; iCell<track.NCells()-1; iCell++ ){
    AliHLTTPCCACell &c1 = fSlice->ID2Cell(vCells[iCell].ID());
    AliHLTTPCCACell &c2 = fSlice->ID2Cell(vCells[iCell+1].ID());
    AliHLTTPCCARow &row1 = fSlice->ID2Row(vCells[iCell].ID());
    AliHLTTPCCARow &row2 = fSlice->ID2Row(vCells[iCell+1].ID());

    if( DrawCells ) ConnectCells( fSlice->ID2IRow(vCells[iCell].ID()),c1,
				  fSlice->ID2IRow(vCells[iCell+1].ID()),c2, color );
    Float_t x1, y1, z1, x2, y2, z2;
    t.GetDCAPoint( row1.X(), c1.Y(), c1.Z(), x1, y1, z1 );
    t.GetDCAPoint( row2.X(), c2.Y(), c2.Z(), x2, y2, z2 );

    Double_t vx1, vy1, vx2, vy2;
    Slice2View(x1, y1, &vx1, &vy1 );
    Slice2View(x2, y2, &vx2, &vy2 );
    
    fLine.SetLineColor(color);
    fLine.SetLineWidth(width);    
    
    fZX->cd();
    fLine.DrawLine(z1,vy1, z2, vy2 ); 
  }
  fLine.SetLineWidth(1);     
  delete[] vCells;
}


void AliHLTTPCCADisplay::DrawTrackletPoint( AliHLTTPCCATrackParam &t, Int_t color )
{
  // draw tracklet point

  Double_t x = t.GetX();
  Double_t y = t.GetY();
  Double_t sinPhi = t.GetSinPhi();
  Double_t z = t.GetZ();
  Double_t dzds = t.GetDzDs();
  Double_t ex = t.GetCosPhi();
  Double_t ey = sinPhi;

  int width = 3;

  if( color<0 ) color = GetColor( t.GetZ() );
    
  fMarker.SetMarkerColor(color);
  fMarker.SetMarkerSize(.5);
  fLine.SetLineWidth(width);  
  fLine.SetLineColor(color);

  Double_t vx, vy, vex, vey, vdx, vdy;
  Double_t dz = TMath::Sqrt(t.GetErr2Z());
  Slice2View( x, y, &vx, &vy ); 
  Slice2View( ex, ey, &vex, &vey ); 
  Slice2View( 0, TMath::Sqrt(t.GetErr2Y())*3.5, &vdx, &vdy);
  Double_t d = TMath::Sqrt(vex*vex+vey*vey);
  vex/=d;
  vey/=d;
  fYX->cd();
  fMarker.DrawMarker(vx,vy);
  fLine.DrawLine(vx,vy,vx+vex*4, vy+vey*4);
  fLine.DrawLine(vx-vdx,vy-vdy, vx+vdx, vy+vdy );
  fZX->cd();
  fMarker.DrawMarker(z,vy);
  fLine.DrawLine(z,vy,z+dzds*4, vy+vey*4);
  fLine.DrawLine(z-3.5*dz,vy-vdy, z+3.5*dz, vy+vdy ); 
  fLine.SetLineWidth(1);
}
