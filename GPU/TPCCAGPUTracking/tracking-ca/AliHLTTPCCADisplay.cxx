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


//#include "AliHLTTPCCADisplay.h"

#ifdef XXXX

//#include "AliHLTTPCCATracker.h"
//#include "AliHLTTPCCARow.h"
//#include "AliHLTTPCCATrack.h"

//#include "TString.h"
//#include "Riostream.h"
//#include "TMath.h"
//#include "TStyle.h"
//#include "TCanvas.h"


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

    fYX->Range(cx, cy-dr*0.8, cx+dr/2, cy+dr*0.1);
    fZX->Range(cz, cy-dr*.8, cz+dz, cy+dr*0.1);
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
  Double_t dy = 0.35;//h->ErrY()*3.5;
  //Double_t dz = h->ErrZ()*3.5;
  fMarker.SetMarkerSize(.5);
  //fMarker.SetMarkerSize(.3);
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


 


 



void AliHLTTPCCADisplay::DrawMergedHit( Int_t iRow, Int_t iHit, Int_t color )
{
#ifdef XXX
  // connect two cells on display, kind of row is drawing
  
  AliHLTTPCCARow &row = fSlice->Rows()[iRow];
  AliHLTTPCCAHit &h = row.Hits()[iHit];
  AliHLTTPCCAHit &hyz = row.HitsYZ()[iHit];

  Double_t x = row.X();
  Double_t y = hyz.Y();
  Double_t z = hyz.Z();
  Double_t x1 = x, x2 = x;
  Double_t y1 = y, y2 = y;
  Double_t z1 = z, z2 = z;
  Int_t iRow1 = iRow, iHit1 = iHit;
  Int_t iRow2 = iRow, iHit2 = iHit;

  if( fSlice->HitLinksDown()[]>=0 ){    
    iRow1 = iRow - 1;
    iHit1 = h.LinkDown();
    AliHLTTPCCARow &row1 = fSlice->Rows()[iRow1];
    AliHLTTPCCAHitYZ &h1 = row1.HitsYZ()[iHit1];
    x1 = row1.X();
    y1 = h1.Y();
    z1 = h1.Z();
  }
  if( h.LinkUp()>=0 ){    
    iRow2 = iRow+1;
    iHit2 = h.LinkUp();
    AliHLTTPCCARow &row2 = fSlice->Rows()[iRow2];
    AliHLTTPCCAHitYZ &h2 = row2.HitsYZ()[iHit2];
    x2 = row2.X();
    y2 = h2.Y();
    z2 = h2.Z();
  }
  if( color<0 ) color = GetColor( (z+z1+z2)/3. );


  Slice2View(x,y, &x, &y );
  Slice2View(x1,y1, &x1, &y1 );
  Slice2View(x2,y2, &x2, &y2 );

  Double_t lx[] = { x1, x, x2 };
  Double_t ly[] = { y1, y, y2 };
  Double_t lz[] = { z1, z, z2 };

  fPLine.SetLineColor(color);    
  fPLine.SetLineWidth(1);        
  //fPLine.SetFillColor(color);
  fPLine.SetFillStyle(-1);
 
  fYX->cd();
  fPLine.DrawPolyLine(3, lx, ly );
  fZX->cd();
  fPLine.DrawPolyLine(3, lz, ly );   
  DrawHit( iRow, iHit, color );
  DrawHit( iRow1, iHit1, color );
  DrawHit( iRow2, iHit2, color ); 
#endif
}


void AliHLTTPCCADisplay::DrawTrack( AliHLTTPCCATrack &track, Int_t color, Bool_t DrawHits )
{
  // draw track
  
  if( track.NHits()<2 ) return;
  int width = 2;

  AliHLTTPCCADisplayTmpHit *vHits = new AliHLTTPCCADisplayTmpHit[track.NHits()];
  AliHLTTPCCATrackParam &t = track.Param();

  Int_t iID = track.FirstHitID();
  int nhits = 0;
  { 
    Int_t iHit = 0;
    for( int ih=0; ih<track.NHits(); ih++ ){
      Int_t i = fSlice->TrackHits()[iID];
      AliHLTTPCCAHit *h = &(fSlice->ID2Hit( i )); 
      AliHLTTPCCARow &row = fSlice->ID2Row(i);
      vHits[iHit].ID() = i;    
      vHits[iHit].S() = t.GetS( row.X(), h->Y() );
      vHits[iHit].Z() = h->Z();
      iHit++;
      nhits++;
      iID++;
    }
  }
  sort(vHits, vHits + track.NHits(), AliHLTTPCCADisplayTmpHit::CompareHitZ );
  cout<<"Draw track, nhits = "<<nhits<<endl;
  {
    AliHLTTPCCAHit &c1 = fSlice->ID2Hit(vHits[0].ID());
    AliHLTTPCCAHit &c2 = fSlice->ID2Hit(vHits[track.NHits()-1].ID());
    if( color<0 ) color = GetColor( (c1.Z()+c2.Z())/2. );
  }
  
  fMarker.SetMarkerColor(color);//kBlue);
  fMarker.SetMarkerSize(1.);
  /*
  for( Int_t i=0; i<3; i++){    
    AliHLTTPCCAHit &c1 = fSlice->ID2Hit(track.HitID()[i]);    
    AliHLTTPCCARow &row1 = fSlice->ID2Row(track.HitID()[i]);
    Double_t vx1, vy1;
    Slice2View(row1.X(), c1.Y(), &vx1, &vy1 ); 
    fYX->cd();
    fMarker.DrawMarker(vx1,vy1);
    fZX->cd();
    fMarker.DrawMarker(c1.Z(),vy1);
  }
  */
  
  //DrawTrackletPoint( fSlice->ID2Point(track.PointID()[0]).Param(), kBlack);//color );
  //DrawTrackletPoint( fSlice->ID2Point(track.PointID()[1]).Param(), kBlack);//color );
  //cout<<"DrawTrack end points x = "<<fSlice->ID2Point(track.PointID()[0]).Param().GetX()<<" "<<fSlice->ID2Point(track.PointID()[1]).Param().GetX()<<endl;
  for( Int_t iHit=0; iHit<track.NHits()-1; iHit++ )
  {
    AliHLTTPCCAHit &c1 = fSlice->ID2Hit(vHits[iHit].ID());
    AliHLTTPCCAHit &c2 = fSlice->ID2Hit(vHits[iHit+1].ID());
    AliHLTTPCCARow &row1 = fSlice->ID2Row(vHits[iHit].ID());
    AliHLTTPCCARow &row2 = fSlice->ID2Row(vHits[iHit+1].ID());
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

  for( Int_t iHit=0; iHit<track.NHits()-1; iHit++ ){
    AliHLTTPCCAHit &c1 = fSlice->ID2Hit(vHits[iHit].ID());
    AliHLTTPCCAHit &c2 = fSlice->ID2Hit(vHits[iHit+1].ID());
    AliHLTTPCCARow &row1 = fSlice->ID2Row(vHits[iHit].ID());
    AliHLTTPCCARow &row2 = fSlice->ID2Row(vHits[iHit+1].ID());

    //if( DrawHits ) ConnectHits( fSlice->ID2IRow(vHits[iHit].ID()),c1,
    //fSlice->ID2IRow(vHits[iHit+1].ID()),c2, color );
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
  delete[] vHits;  
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

  int width = 1;

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
#endif //XXXX
