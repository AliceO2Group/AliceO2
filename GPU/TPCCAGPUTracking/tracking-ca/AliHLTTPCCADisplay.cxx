// @(#) $Id$
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
//                                                                        *
// Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
//                  for The ALICE HLT Project.                            *
//                                                                        *
// Permission to use, copy, modify and distribute this software and its   *
// documentation strictly for non-commercial purposes is hereby granted   *
// without fee, provided that the above copyright notice appears in all   *
// copies and that both the copyright notice and this permission notice   *
// appear in the supporting documentation. The authors make no claims     *
// about the suitability of this software for any purpose. It is          *
// provided "as is" without express or implied warranty.                  *
//*************************************************************************

#include "AliHLTTPCCADisplay.h"

#include "AliHLTTPCCATracker.h"
#include "TString.h"
#include "Riostream.h"
#include "TMath.h"
#include "TStyle.h"
#include "TCanvas.h"
#include <vector>

ClassImp(AliHLTTPCCADisplay);

AliHLTTPCCADisplay &AliHLTTPCCADisplay::Instance()
{
  // reference to static object
  static AliHLTTPCCADisplay gAliHLTTPCCADisplay;
  return gAliHLTTPCCADisplay; 
}

AliHLTTPCCADisplay::AliHLTTPCCADisplay() : fXY(0), fZY(0), fAsk(1), fSectorView(1), fSector(0), 
					   fCos(1), fSin(0), fZMin(-250), fZMax(250),
					   fRInnerMin(83.65), fRInnerMax(133.3), fROuterMin(133.5), fROuterMax(247.7),
					   fTPCZMin(-250.), fTPCZMax(250), fArc(), fLine(), fPLine(), fMarker(), fBox(), fCrown(), fLatex()
{
  // constructor
}

AliHLTTPCCADisplay::AliHLTTPCCADisplay( const AliHLTTPCCADisplay& ) 
  : fXY(0), fZY(0), fAsk(1), fSectorView(1), fSector(0), 
    fCos(1), fSin(0), fZMin(-250), fZMax(250),
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
  delete fXY; 
  delete fZY;
}

void AliHLTTPCCADisplay::Init()
{
  // initialization
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasBorderSize(1);
  gStyle->SetCanvasColor(0);
  fXY = new TCanvas ("XY", "XY window", -1, 0, 600, 600);
  fZY = new TCanvas ("ZY", "ZY window", -610, 0, 590, 600);  
  fMarker = TMarker(0.0, 0.0, 6);
}

void AliHLTTPCCADisplay::Update()
{
  // update windows
  if( !fAsk ) return;
  fXY->Update();
  fZY->Update();
}

void AliHLTTPCCADisplay::Clear()
{
  // clear windows
  fXY->Clear();
  fZY->Clear();
}

void AliHLTTPCCADisplay::Ask()
{
  // whait for the pressed key, when "r" pressed, don't ask anymore
  char symbol;
  if (fAsk){
    Update();
    cout<<"ask> "<<endl;
    do{
      cin.get(symbol);
      if (symbol == 'r')
	fAsk = false;
    } while (symbol != '\n');
  }
}


void AliHLTTPCCADisplay::SetSectorView()
{
  // switch to sector view
  fSectorView = 1;
}

void AliHLTTPCCADisplay::SetTPCView()
{
  // switch to full TPC view
  fSectorView = 0;
  fCos = 1;
  fSin = 0;
  fZMin = fTPCZMin;
  fZMax = fTPCZMax;
}

void AliHLTTPCCADisplay::SetCurrentSector( AliHLTTPCCATracker *sec )
{
  // set reference to the current CA tracker, and read the current sector geometry
  fSector = sec;
  if( fSectorView ){
    fCos = sec->Param().SinAlpha();
    fSin = - sec->Param().CosAlpha();
    fZMin = sec->Param().ZMin();
    fZMax = sec->Param().ZMax();
    Clear();
    Double_t r0 = .5*(sec->Param().RMax()+sec->Param().RMin());
    Double_t dr = .5*(sec->Param().RMax()-sec->Param().RMin());
    Double_t cx = 0;
    Double_t cy = r0;

    fXY->Range(cx-dr, cy-dr, cx+dr, cy+dr);
    Double_t cz = .5*(sec->Param().ZMax()+sec->Param().ZMin());
    Double_t dz = .5*(sec->Param().ZMax()-sec->Param().ZMin())*1.2;
    fZY->Range(cz-dz, cy-dr, cz+dz, cy+dr);
  }
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
  return kMyColor[iz]-4;
}

void AliHLTTPCCADisplay::Global2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const
{
  // convert coordinates global->view
  *xv = x*fCos + y*fSin;
  *yv = y*fCos - x*fSin;
}

void AliHLTTPCCADisplay::Sec2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const
{
  // convert coordinates sector->view
  Double_t xg = x*fSector->Param().CosAlpha() - y*fSector->Param().SinAlpha();
  Double_t yg = y*fSector->Param().CosAlpha() + x*fSector->Param().SinAlpha();
  *xv = xg*fCos + yg*fSin;
  *yv = yg*fCos - xg*fSin;
}


void AliHLTTPCCADisplay::DrawTPC()
{
  // schematically draw TPC detector
  fXY->Range(-fROuterMax, -fROuterMax, fROuterMax, fROuterMax);
  fXY->Clear();
  {
    fArc.SetLineColor(kBlack);
    fArc.SetFillStyle(0);
    fXY->cd();    
    for( Int_t iSec=0; iSec<18; iSec++){
      fCrown.SetLineColor(kBlack);
      fCrown.SetFillStyle(0);
      fCrown.DrawCrown(0,0,fRInnerMin, fRInnerMax, 360./18.*iSec, 360./18.*(iSec+1) );
      fCrown.DrawCrown(0,0,fROuterMin, fROuterMax, 360./18.*iSec, 360./18.*(iSec+1) );
    }
  }
  fZY->cd();
  fZY->Range( fTPCZMin, -fROuterMax, fTPCZMax, fROuterMax );
  fZY->Clear();
}

void AliHLTTPCCADisplay::DrawSector( AliHLTTPCCATracker *sec )
{     
  // draw current the TPC sector
  fXY->cd();
  Double_t r0 = .5*(sec->Param().RMax()+sec->Param().RMin());
  Double_t dr = .5*(sec->Param().RMax()-sec->Param().RMin());
  Double_t cx = r0*sec->Param().CosAlpha();
  Double_t cy = r0*sec->Param().SinAlpha();
  Double_t raddeg = 180./3.1415;
  Double_t a0 = raddeg*.5*(sec->Param().AngleMax() + sec->Param().AngleMin());
  Double_t da = raddeg*.5*(sec->Param().AngleMax() - sec->Param().AngleMin());
  if( fSectorView ){
    cx = 0; cy = r0;
    a0 = 90.;
    fLatex.DrawLatex(cx-dr+dr*.05,cy-dr+dr*.05, Form("Sec.%2i",sec->Param().ISec()));
  }
  fArc.SetLineColor(kBlack);
  fArc.SetFillStyle(0);     
  fCrown.SetLineColor(kBlack);
  fCrown.SetFillStyle(0);
    
  fCrown.DrawCrown(0,0, sec->Param().RMin(),sec->Param().RMax(), a0-da, a0+da );

  fLine.SetLineColor(kBlack);
 
  fZY->cd();

  Double_t cz = .5*(sec->Param().ZMax()+sec->Param().ZMin());
  Double_t dz = .5*(sec->Param().ZMax()-sec->Param().ZMin())*1.2;
  //fLine.DrawLine(cz+dz, cy-dr, cz+dz, cy+dr ); 
  if( fSectorView ) fLatex.DrawLatex(cz-dz+dz*.05,cy-dr+dr*.05, Form("Sec.%2i",sec->Param().ISec()));
}


void AliHLTTPCCADisplay::DrawHit( Int_t iRow, Int_t iHit, Int_t color )
{
  // draw hit
  if( !fSector ) return;
  AliHLTTPCCARow &row = fSector->Rows()[iRow];
  AliHLTTPCCAHit *h = &(row.Hits()[iHit]);
  if( color<0 ) color = GetColor( h->Z() );

  Double_t dgy = 3.*TMath::Abs(h->ErrY()*fSector->Param().CosAlpha() - fSector->Param().ErrX()*fSector->Param().SinAlpha() );
  Double_t dx = fSector->Param().ErrX()*TMath::Sqrt(12.)/2.;
  Double_t dy = h->ErrY()*3.;
  Double_t dz = h->ErrZ()*3.;
  fMarker.SetMarkerColor(color);
  fArc.SetLineColor(color);
  fArc.SetFillStyle(0);
  Double_t vx, vy;
  Sec2View( row.X(), h->Y(), &vx, &vy );
  
  fXY->cd();
  if( fSectorView ) fArc.DrawEllipse( vx, vy, dx, dy, 0,360, 90);
  else  fArc.DrawEllipse( vx, vy, dx, dy, 0,360, fSector->Param().Alpha()*180./3.1415);
  fMarker.DrawMarker(vx, vy);
  fZY->cd();
  if( fSectorView ) fArc.DrawEllipse( h->Z(), vy, dz, dx, 0,360, 90 );
  else fArc.DrawEllipse( h->Z(), vy, dz, dgy, 0,360, fSector->Param().Alpha()*180./3.1415);
  fMarker.DrawMarker(h->Z(), vy); 
}

void AliHLTTPCCADisplay::DrawCell( Int_t iRow, AliHLTTPCCACell &cell, Int_t width, Int_t color )
{
  // draw cell
  AliHLTTPCCARow &row = fSector->Rows()[iRow];
  Double_t vx, vy, vdx, vdy;
  Sec2View(row.X(), cell.Y(), &vx, &vy);
  Sec2View(0, cell.ErrY()*3, &vdx, &vdy);
  if( color<0 ) color = GetColor(cell.Z());
  fLine.SetLineColor(color);
  fLine.SetLineWidth(width);
  fXY->cd();
  fLine.DrawLine(vx-vdx,vy-vdy, vx+vdx, vy+vdy );
  fZY->cd();
  fLine.DrawLine(cell.Z()-3*cell.ErrZ(),vy-vdy, cell.Z()+3*cell.ErrZ(), vy+vdy ); 
  fLine.SetLineWidth(1);
}

void  AliHLTTPCCADisplay::DrawCell( Int_t iRow, Int_t iCell, Int_t width, Int_t color )
{
  // draw cell
  AliHLTTPCCARow &row = fSector->Rows()[iRow];
  DrawCell( iRow, row.Cells()[iCell], width, color );
}
 
void AliHLTTPCCADisplay::ConnectCells( Int_t iRow1, AliHLTTPCCACell &cell1, 
				       Int_t iRow2, AliHLTTPCCACell &cell2, Int_t color )
{
  // connect two cells on display, kind of row is drawing
  AliHLTTPCCARow &row1 = fSector->Rows()[iRow1];
  AliHLTTPCCARow &row2 = fSector->Rows()[iRow2];

  AliHLTTPCCAHit &h11 = row1.GetCellHit(cell1,0);
  AliHLTTPCCAHit &h12 = row1.GetCellHit(cell1,cell1.NHits()-1);
  AliHLTTPCCAHit &h21 = row2.GetCellHit(cell2,0);
  AliHLTTPCCAHit &h22=  row2.GetCellHit(cell2,cell2.NHits()-1);

  Double_t x11 = row1.X();
  Double_t x12 = row1.X();
  Double_t y11 = h11.Y() - h11.ErrY()*3;
  Double_t y12 = h12.Y() + h12.ErrY()*3;
  Double_t z11 = h11.Z();
  Double_t z12 = h12.Z();
  Double_t x21 = row2.X();
  Double_t x22 = row2.X();
  Double_t y21 = h21.Y() - h21.ErrY()*3;
  Double_t y22 = h22.Y() + h22.ErrY()*3;
  Double_t z21 = h21.Z();
  Double_t z22 = h22.Z();

  Double_t vx11, vx12, vy11, vy12, vx21, vx22, vy21, vy22;

  Sec2View(x11,y11, &vx11, &vy11 );
  Sec2View(x12,y12, &vx12, &vy12 );
  Sec2View(x21,y21, &vx21, &vy21 );
  Sec2View(x22,y22, &vx22, &vy22 );

  Double_t lx[] = { vx11, vx12, vx22, vx21, vx11 };
  Double_t ly[] = { vy11, vy12, vy22, vy21, vy11 };
  Double_t lz[] = { z11, z12, z22, z21, z11 };

  if( color<0 ) color = GetColor( (z11+z12+z22+z21)/4. );
  fPLine.SetLineColor(color);    
  fPLine.SetLineWidth(1);        
  //fPLine.SetFillColor(color);
  fPLine.SetFillStyle(-1);
 
  fXY->cd();
  fPLine.DrawPolyLine(5, lx, ly );
  fZY->cd();
  fPLine.DrawPolyLine(5, lz, ly );   
  DrawCell( iRow1, cell1, 1, color );
  DrawCell( iRow2, cell2, 1, color );
}


Bool_t CompareCellDS( const pair<int,double> &a, const pair<int,double> &b )
{
  // function used to sort cells track along trajectory, pair<cell index, track length>
  return (a.second<b.second);
}

void AliHLTTPCCADisplay::DrawTrack( AliHLTTPCCATrack &track, Int_t color )
{
  // draw track
  if( track.NCells()<2 ) return;

  Double_t b = -5;    

  std::vector<pair<int,double> > vCells;
  AliHLTTPCCATrackPar t = track.Param();
  for( Int_t iCell=0; iCell<track.NCells(); iCell++ ){
    AliHLTTPCCACell &c = fSector->GetTrackCell(track,iCell);
    AliHLTTPCCARow &row = fSector->GetTrackCellRow(track,iCell);
    Double_t xyz[3] = {row.X(), c.Y(), c.Z()};
    if( iCell==0 ) t.TransportBz(-5, xyz);
    pair<int,double> tmp(iCell, t.GetDsToPointBz(-5.,xyz));
    vCells.push_back(tmp);
  }
  sort(vCells.begin(), vCells.end(), CompareCellDS );
  t.Normalize();
  const Double_t kCLight = 0.000299792458;
  Double_t bc = b*kCLight;
  Double_t pt = sqrt(t.Par()[3]*t.Par()[3] +t.Par()[4]*t.Par()[4] );
  //Double_t p = sqrt(pt*pt +t.Par()[5]*t.Par()[5] );
  Double_t q = t.Par()[6];

  //for( Int_t iCell=0; iCell<track.fNCells-1; iCell++ )
  {    
    AliHLTTPCCACell &c1 = fSector->GetTrackCell(track,vCells[0].first);
    AliHLTTPCCACell &c2 = fSector->GetTrackCell(track,vCells[track.NCells()-1].first);
    AliHLTTPCCARow &row1 = fSector->GetTrackCellRow(track,vCells[0].first);
    AliHLTTPCCARow &row2 = fSector->GetTrackCellRow(track,vCells[track.NCells()-1].first);
    if( color<0 ) color = GetColor( (c1.Z()+c2.Z())/2. );
    Double_t vx1, vy1, vx2, vy2;
    Sec2View(row1.X(), c1.Y(), &vx1, &vy1 );
    Sec2View(row2.X(), c2.Y(), &vx2, &vy2 );
    
    fLine.SetLineColor( color );
    fLine.SetLineWidth(3);
    
    if( TMath::Abs(q)>.1 ){
      Double_t qq = pt/q;
      
      Double_t xc = t.Par()[0] + qq*t.Par()[4]/pt/bc;
      Double_t yc = t.Par()[1] - qq*t.Par()[3]/pt/bc;
      Double_t r = TMath::Abs(qq)/fabs(bc);
	
      Double_t vx, vy;
      Sec2View( xc, yc, &vx, &vy );
      
      Double_t a1 = TMath::ATan2(vy1-vy, vx1-vx)/TMath::Pi()*180.;
      Double_t a2 = TMath::ATan2(vy2-vy, vx2-vx)/TMath::Pi()*180.;
      Double_t da= a2-a1;      
      if( da>=180 ) da=360-da;
      if( da<=-180 ) da=360-da;
      a1 = a2-da;
      fArc.SetFillStyle(0);
      fArc.SetLineColor(color);    
      fArc.SetLineWidth(3);        
      
      fXY->cd();
      
      fArc.DrawArc(vx,vy,r, a1,a2,"only");
    } else {
      fXY->cd();
      fLine.DrawLine(vx1,vy1, vx2, vy2 );
    }
  }

  for( Int_t iCell=0; iCell<track.NCells()-1; iCell++ ){
    
    AliHLTTPCCACell &c1 = fSector->GetTrackCell(track,vCells[iCell].first);
    AliHLTTPCCACell &c2 = fSector->GetTrackCell(track,vCells[iCell+1].first);
    AliHLTTPCCARow &row1 = fSector->GetTrackCellRow(track,vCells[iCell].first);
    AliHLTTPCCARow &row2 = fSector->GetTrackCellRow(track,vCells[iCell+1].first);
    Double_t x1, y1, z1, x2, y2, z2;
    {
      Double_t xyz[3] = {row1.X(), c1.Y(), c1.Z()};
      t.TransportBz(-5, xyz);
      x1 = t.Par()[0]; y1 = t.Par()[1]; z1 = t.Par()[2];
    }
    {
      Double_t xyz[3] = {row2.X(), c2.Y(), c2.Z()};
      t.TransportBz(-5, xyz);
      x2 = t.Par()[0]; y2 = t.Par()[1]; z2 = t.Par()[2];
    }
   
    Double_t vx1, vy1, vx2, vy2;
    Sec2View(x1, y1, &vx1, &vy1 );
    Sec2View(x2, y2, &vx2, &vy2 );
    
    fLine.SetLineColor(color);
    fLine.SetLineWidth(1);
    
    fZY->cd();
    fLine.DrawLine(z1,vy1, z2, vy2 ); 
  }
  fLine.SetLineWidth(1);     
}

