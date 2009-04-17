// $Id$
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


#include "AliHLTTPCCADisplay.h"


#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGBTracker.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCAGBTrack.h"
#include "AliHLTTPCCAGBHit.h"
#include "AliHLTTPCCAPerformance.h"
#include "AliHLTTPCCAMCTrack.h"
#include "AliHLTTPCCAOutTrack.h"

#include "TString.h"
#include "Riostream.h"
#include "TMath.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TApplication.h"


class AliHLTTPCCADisplay::AliHLTTPCCADisplayTmpHit
{

  public:

    int ID() const { return fHitID; }
    double S() const { return fS; }
    double Z() const { return fZ; }

    void SetID( int v ) { fHitID = v; }
    void SetS( double v ) { fS = v; }
    void SetZ( double v ) { fZ = v; }

    static bool CompareHitDS( const AliHLTTPCCADisplayTmpHit &a,
                              const AliHLTTPCCADisplayTmpHit  &b ) {
      return ( a.fS < b.fS );
    }

    static bool CompareHitZ( const AliHLTTPCCADisplayTmpHit &a,
                             const AliHLTTPCCADisplayTmpHit  &b ) {
      return ( a.fZ < b.fZ );
    }

  private:

    int fHitID; // hit ID
    double fS;  // hit position on the XY track curve
    double fZ;  // hit Z position

};



AliHLTTPCCADisplay &AliHLTTPCCADisplay::Instance()
{
  // reference to static object
  static AliHLTTPCCADisplay gAliHLTTPCCADisplay;
  static bool firstCall = 1;
  if ( firstCall ) {
    if ( !gApplication ) new TApplication( "myapp", 0, 0 );
    gAliHLTTPCCADisplay.Init();
    firstCall = 0;
  }
  return gAliHLTTPCCADisplay;
}

AliHLTTPCCADisplay::AliHLTTPCCADisplay() : fYX( 0 ), fZX( 0 ), fAsk( 1 ), fSliceView( 1 ), fSlice( 0 ), fGB( 0 ), fPerf( 0 ),
    fCos( 1 ), fSin( 0 ), fZMin( -250 ), fZMax( 250 ), fYMin( -250 ), fYMax( 250 ), fSliceCos( 1 ), fSliceSin( 0 ),
    fRInnerMin( 83.65 ), fRInnerMax( 133.3 ), fROuterMin( 133.5 ), fROuterMax( 247.7 ),
    fTPCZMin( -250. ), fTPCZMax( 250 ), fArc(), fLine(), fPLine(), fMarker(), fBox(), fCrown(), fLatex(), fDrawOnlyRef( 0 )
{
  fPerf = &( AliHLTTPCCAPerformance::Instance() );
  // constructor
}


AliHLTTPCCADisplay::AliHLTTPCCADisplay( const AliHLTTPCCADisplay& )
    : fYX( 0 ), fZX( 0 ), fAsk( 1 ), fSliceView( 1 ), fSlice( 0 ), fGB( 0 ), fPerf( 0 ),
    fCos( 1 ), fSin( 0 ), fZMin( -250 ), fZMax( 250 ), fYMin( -250 ), fYMax( 250 ), fSliceCos( 1 ), fSliceSin( 0 ),
    fRInnerMin( 83.65 ), fRInnerMax( 133.3 ), fROuterMin( 133.5 ), fROuterMax( 247.7 ),
    fTPCZMin( -250. ), fTPCZMax( 250 ), fArc(), fLine(), fPLine(), fMarker(), fBox(), fCrown(), fLatex(), fDrawOnlyRef( 0 )
{
  // dummy
}

const AliHLTTPCCADisplay& AliHLTTPCCADisplay::operator=( const AliHLTTPCCADisplay& ) const
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
  gStyle->SetCanvasBorderMode( 0 );
  gStyle->SetCanvasBorderSize( 1 );
  gStyle->SetCanvasColor( 0 );
  fYX = new TCanvas ( "YX", "YX window", -1, 0, 600, 600 );
  fZX = new TCanvas ( "ZX", "ZX window", -610, 0, 590, 600 );
  fMarker = TMarker( 0.0, 0.0, 20 );//6);
  fDrawOnlyRef = 0;
}

void AliHLTTPCCADisplay::Update()
{
  // update windows
  if ( !fAsk ) return;
  fYX->Update();
  fZX->Update();
  fYX->Print( "YX.pdf" );
  fZX->Print( "ZX.pdf" );

}

void AliHLTTPCCADisplay::ClearView()
{
  // clear windows
  fYX->Clear();
  fZX->Clear();
}

void AliHLTTPCCADisplay::Ask()
{
  // wait for the pressed key, when "r" pressed, don't ask anymore
  char symbol;
  if ( fAsk ) {
    Update();
    std::cout << "ask> " << std::endl;
    do {
      std::cin.get( symbol );
      if ( symbol == 'r' )
        fAsk = false;
    } while ( symbol != '\n' );
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
  fYMin = -fROuterMax;
  fYMax = fROuterMax;
}


void AliHLTTPCCADisplay::SetGB( AliHLTTPCCAGBTracker * const GBTracker )
{
  fGB = GBTracker;
}

void AliHLTTPCCADisplay::SetCurrentSlice( AliHLTTPCCATracker *slice )
{
  // set reference to the current CA tracker, and read the current slice geometry
  fSlice = slice;
  SetSliceTransform( slice );
  if ( fSliceView ) {
    fCos = slice->Param().SinAlpha();
    fSin = slice->Param().CosAlpha();
    fZMin = slice->Param().ZMin();
    fZMax = slice->Param().ZMax();
    ClearView();
    double r0 = .5 * ( slice->Param().RMax() + slice->Param().RMin() );
    double dr = .5 * ( slice->Param().RMax() - slice->Param().RMin() );
    fYMin = -dr;
    fYMax = dr;
    double cx = 0;
    double cy = r0;
    double cz = .5 * ( slice->Param().ZMax() + slice->Param().ZMin() );
    double dz = .5 * ( slice->Param().ZMax() - slice->Param().ZMin() ) * 1.2;
    fYX->Range( cx - dr, cy - dr*1.05, cx + dr, cy + dr );
    fZX->Range( cz - dz, cy - dr*1.05, cz + dz, cy + dr );

    //fYX->Range(cx-dr*.3, cy-dr*1.05, cx+dr*.3, cy-dr*.35);
    //fZX->Range(cz-dz, cy-dr*1.05, cz+dz, cy-dr*.3);

    //fYX->Range(cx-dr*.3, cy-dr*.8, cx-dr*.1, cy-dr*.75);
    //fZX->Range(cz-dz*0, cy-dr*.8, cz+dz, cy-dr*.75);

    //fYX->Range(cx-dr*.08, cy-dr*1., cx-dr*.02, cy-dr*0.7);
    //fZX->Range(cz-dz*.2, cy-dr*1., cz-dz*.05, cy-dr*0.7);

    //double x0 = cx-dr*.1, x1 = cx-dr*.05;
    //double y0 = cy-dr*1.05, y1 = cy-dr*0.7;
    //double z0 = cz-dz*.3, z1 = cz;
    //double xc = (x0+x1)/2, yc= (y0+y1)/2, zc=(z0+z1)/2;
    //double d = TMath::Max((x1-x0)/2,TMath::Max((y1-y0)/2,(z1-z0)/2));
    //fYX->Range(xc-d, yc-d, xc+d, yc+d);
    //fZX->Range(zc-d, yc-d, zc+d, yc+d);

  }
}

void AliHLTTPCCADisplay::SetSliceTransform( double alpha )
{
  fSliceCos = TMath::Cos( alpha );
  fSliceSin = TMath::Sin( alpha );
}

void AliHLTTPCCADisplay::SetSliceTransform( AliHLTTPCCATracker *slice )
{
  SetSliceTransform( slice->Param().Alpha() );
}


void AliHLTTPCCADisplay::DrawTPC()
{
  // schematically draw TPC detector
  fYX->Range( -fROuterMax, -fROuterMax, fROuterMax, fROuterMax );
  //fYX->Range( -fROuterMax*.7, -fROuterMax, fROuterMax*0., -fROuterMax*.5);
  fYX->Clear();
  {
    fArc.SetLineColor( kBlack );
    fArc.SetFillStyle( 0 );
    fYX->cd();
    for ( int iSlice = 0; iSlice < 18; iSlice++ ) {
      fCrown.SetLineColor( kBlack );
      fCrown.SetFillStyle( 0 );
      fCrown.DrawCrown( 0, 0, fRInnerMin, fRInnerMax, 360. / 18.*iSlice, 360. / 18.*( iSlice + 1 ) );
      fCrown.DrawCrown( 0, 0, fROuterMin, fROuterMax, 360. / 18.*iSlice, 360. / 18.*( iSlice + 1 ) );
    }
  }
  fZX->cd();
  fZX->Range( fTPCZMin, -fROuterMax, fTPCZMax*1.1, fROuterMax );
  //fZX->Range( fTPCZMax*.1, -fROuterMax, fTPCZMax*.3, -fROuterMax*0.5 );
  fZX->Clear();
}

void AliHLTTPCCADisplay::DrawSlice( AliHLTTPCCATracker *slice, bool DrawRows )
{
  // draw current the TPC slice
  fYX->cd();
  double r0 = .5 * ( slice->Param().RMax() + slice->Param().RMin() );
  double dr = .5 * ( slice->Param().RMax() - slice->Param().RMin() );
  double cx = r0 * slice->Param().CosAlpha();
  double cy = r0 * slice->Param().SinAlpha();
  double raddeg = 180. / 3.1415;
  double a0 = raddeg * .5 * ( slice->Param().AngleMax() + slice->Param().AngleMin() );
  double da = raddeg * .5 * ( slice->Param().AngleMax() - slice->Param().AngleMin() );
  if ( fSliceView ) {
    cx = 0; cy = r0;
    a0 = 90.;
    fLatex.DrawLatex( cx - dr + dr*.05, cy - dr + dr*.05, Form( "YX, Slice %2i", slice->Param().ISlice() ) );
  } else {
    a0 += raddeg * TMath::ATan2( fSin, fCos );
  }
  fArc.SetLineColor( kBlack );
  fArc.SetFillStyle( 0 );
  fCrown.SetLineColor( kBlack );
  fCrown.SetFillStyle( 0 );
  fCrown.DrawCrown( 0, 0, fRInnerMin, fRInnerMax, a0 - da, a0 + da );
  fCrown.DrawCrown( 0, 0, fROuterMin, fROuterMax, a0 - da, a0 + da );
  //fCrown.DrawCrown(0,0, slice->Param().RMin(),slice->Param().RMax(), a0-da, a0+da );

  fLine.SetLineColor( kBlack );

  fZX->cd();

  double cz = .5 * ( slice->Param().ZMax() + slice->Param().ZMin() );
  double dz = .5 * ( slice->Param().ZMax() - slice->Param().ZMin() ) * 1.2;
  //fLine.DrawLine(cz+dz, cy-dr, cz+dz, cy+dr );
  if ( fSliceView ) fLatex.DrawLatex( cz - dz + dz*.05, cy - dr + dr*.05, Form( "ZX, Slice %2i", slice->Param().ISlice() ) );

  if ( DrawRows ) {
    fLine.SetLineWidth( 1 );
    fLine.SetLineColor( kBlack );
    SetSliceTransform( fSlice );
    for ( int iRow = 0; iRow < fSlice->Param().NRows(); iRow++ ) {
      double x = fSlice->Row( iRow ).X();
      double y = fSlice->Row( iRow ).MaxY();
      double vx0, vy0, vx1, vy1;
      Slice2View( x, y, &vx0, &vy0 );
      Slice2View( x, -y, &vx1, &vy1 );
      fYX->cd();
      fLine.DrawLine( vx0, vy0, vx1, vy1 );
      fZX->cd();
      fLine.DrawLine( fTPCZMin, vy0, fTPCZMax, vy1 );
    }
  }

}


void AliHLTTPCCADisplay::Set2Slices( AliHLTTPCCATracker * const slice )
{
  //* Set view for two neighbouring slices

  fSlice = slice;
  fSliceView = 0;
  fCos = TMath::Cos( TMath::Pi() / 2 - ( slice->Param().Alpha() + 10. / 180.*TMath::Pi() ) );
  fSin = TMath::Sin( TMath::Pi() / 2 - ( slice->Param().Alpha() + 10. / 180.*TMath::Pi() ) );
  fZMin = slice->Param().ZMin();
  fZMax = slice->Param().ZMax();
  ClearView();
  double r0 = .5 * ( slice->Param().RMax() + slice->Param().RMin() );
  double dr = .5 * ( slice->Param().RMax() - slice->Param().RMin() );
  double cx = 0;
  double cy = r0;
  fYX->Range( cx - 1.3*dr, cy - 1.1*dr, cx + 1.3*dr, cy + 1.1*dr );
  fYX->cd();
  int islice = slice->Param().ISlice();
  int jslice = slice->Param().ISlice() + 1;
  if ( islice == 17 ) jslice = 0;
  else if ( islice == 35 ) jslice = 18;
  fLatex.DrawLatex( cx - 1.3*dr + 1.3*dr*.05, cy - dr + dr*.05, Form( "YX, Slices %2i/%2i", islice, jslice ) );
  double cz = .5 * ( slice->Param().ZMax() + slice->Param().ZMin() );
  double dz = .5 * ( slice->Param().ZMax() - slice->Param().ZMin() ) * 1.2;
  fZX->Range( cz - dz, cy - 1.1*dr, cz + dz, cy + 1.1*dr );//+dr);
  fZX->cd();
  fLatex.DrawLatex( cz - dz + dz*.05, cy - dr + dr*.05, Form( "ZX, Slices %2i/%2i", islice, jslice ) );
}

int AliHLTTPCCADisplay::GetColor( int i ) const
{
  // Get color with respect to Z coordinate
  const Color_t kMyColor[9] = { kGreen, kBlue, kYellow, kCyan, kOrange,
                                kSpring, kTeal, kAzure, kViolet
                              };
  if ( i < 0 ) i = 0;
  if ( i == 0 ) return kBlack;
  return kMyColor[( i-1 )%9];
}

int AliHLTTPCCADisplay::GetColorZ( double z ) const
{
  // Get color with respect to Z coordinate
  const Color_t kMyColor[11] = { kGreen, kBlue, kYellow, kMagenta, kCyan,
                                 kOrange, kSpring, kTeal, kAzure, kViolet, kPink
                               };

  double zz = ( z - fZMin ) / ( fZMax - fZMin );
  int iz = ( int ) ( zz * 11 );
  if ( iz < 0 ) iz = 0;
  if ( iz > 10 ) iz = 10;
  return kMyColor[iz];
}

int AliHLTTPCCADisplay::GetColorY( double y ) const
{
  // Get color with respect to Z coordinate
  const Color_t kMyColor[11] = { kGreen, kBlue, kYellow, kMagenta, kCyan,
                                 kOrange, kSpring, kTeal, kAzure, kViolet, kPink
                               };

  double yy = ( y - fYMin ) / ( fYMax - fYMin );
  int iy = ( int ) ( yy * 11 );
  if ( iy < 0 ) iy = 0;
  if ( iy > 10 ) iy = 10;
  return kMyColor[iy];
}

int AliHLTTPCCADisplay::GetColorK( double k ) const
{
  // Get color with respect to Z coordinate
  const Color_t kMyColor[11] = { kRed, kBlue, kYellow, kMagenta, kCyan,
                                 kOrange, kSpring, kTeal, kAzure, kViolet, kPink
                               };
  const double kCLight = 0.000299792458;
  const double kBz = 5;
  double k2QPt = 100;
  if ( TMath::Abs( kBz ) > 1.e-4 ) k2QPt = 1. / ( kBz * kCLight );
  double qPt = k * k2QPt;
  double pt = 100;
  if ( TMath::Abs( qPt ) > 1.e-4 ) pt = 1. / TMath::Abs( qPt );

  double yy = ( pt - 0.1 ) / ( 1. - 0.1 );
  int iy = ( int ) ( yy * 11 );
  if ( iy < 0 ) iy = 0;
  if ( iy > 10 ) iy = 10;
  return kMyColor[iy];
}

void AliHLTTPCCADisplay::Global2View( double x, double y, double *xv, double *yv ) const
{
  // convert coordinates global->view
  *xv = x * fCos + y * fSin;
  *yv = y * fCos - x * fSin;
}


void AliHLTTPCCADisplay::Slice2View( double x, double y, double *xv, double *yv ) const
{
  // convert coordinates slice->view
  double xg = x * fSliceCos - y * fSliceSin;
  double yg = y * fSliceCos + x * fSliceSin;
  *xv = xg * fCos - yg * fSin;
  *yv = yg * fCos + xg * fSin;
}


void AliHLTTPCCADisplay::DrawGBHit( AliHLTTPCCAGBTracker &tracker, int iHit, int color, Size_t width  )
{
  // draw hit
  const AliHLTTPCCAGBHit &h = tracker.Hits()[iHit];
  AliHLTTPCCATracker &slice = tracker.Slices()[h.ISlice()];
  SetSliceTransform( &slice );

  if ( color < 0 ) {
    if ( fPerf ) {
      int lab = fPerf->HitLabels()[h.ID()].fLab[0];
      color = GetColor( lab + 1 );
      if ( lab >= 0 ) {
        AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lab];
        if ( mc.P() >= 1. ) color = kRed;
        else if ( fDrawOnlyRef ) return;
      }
    } else color = GetColorZ( h.Z() );
  }
  if ( width > 0 )fMarker.SetMarkerSize( width );
  else fMarker.SetMarkerSize( .3 );
  fMarker.SetMarkerColor( color );
  double vx, vy;
  Slice2View( h.X(), h.Y(), &vx, &vy );

  fYX->cd();
  fMarker.DrawMarker( vx, vy );
  fZX->cd();
  fMarker.DrawMarker( h.Z(), vy );
}

void AliHLTTPCCADisplay::DrawGBHits( AliHLTTPCCAGBTracker &tracker, int color, Size_t width )
{
  // draw hits

  if ( !fPerf ) return;
  if ( width < 0 ) width = .3;

  for ( int iHit = 0; iHit < tracker.NHits(); iHit++ ) {
    const AliHLTTPCCAGBHit &h = tracker.Hits()[iHit];
    int imc = fPerf->HitLabels()[h.ID()].fLab[0];
    AliHLTTPCCAMCTrack *mc = ( imc >= 0 ) ? &( fPerf->MCTracks()[imc] ) : 0;
    if ( fDrawOnlyRef && ( !mc || ( mc->P() < 1 ) ) ) continue;
    int col = color;
    if ( color < 0 ) {
      col = GetColor( imc + 1 ) ;
      if ( mc && ( mc->P() >= 1. ) ) col = kRed;
    }

    AliHLTTPCCATracker &slice = tracker.Slices()[h.ISlice()];
    SetSliceTransform( &slice );

    fMarker.SetMarkerSize( width );
    fMarker.SetMarkerColor( col );
    double vx, vy;
    Slice2View( h.X(), h.Y(), &vx, &vy );

    fYX->cd();
    fMarker.DrawMarker( vx, vy );
    fZX->cd();
    fMarker.DrawMarker( h.Z(), vy );
  }
}

void AliHLTTPCCADisplay::DrawSliceHit( int iRow, int iHit, int color, Size_t width )
{
  // draw hit
  if ( !fSlice ) return;
  const AliHLTTPCCARow &row = fSlice->Row( iRow );
  float y0 = row.Grid().YMin();
  float z0 = row.Grid().ZMin();
  float stepY = row.HstepY();
  float stepZ = row.HstepZ();
  float x = row.X();
  float y = y0 + fSlice->HitDataY( row, iHit ) * stepY;
  float z = z0 + fSlice->HitDataZ( row, iHit ) * stepZ;

  SetSliceTransform( fSlice );

  if ( color < 0 ) {
    if ( fPerf && fGB ) {
      int id = fGB->FirstSliceHit()[fSlice->Param().ISlice()] + fSlice->HitInputID( row, iHit );
      const AliHLTTPCCAGBHit &h = fGB->Hits()[id];
      int lab = fPerf->HitLabels()[h.ID()].fLab[0];
      color = GetColor( lab + 1 );
      if ( lab >= 0 ) {
        AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lab];
        if ( mc.P() >= 1. ) color = kRed;
        else if ( fDrawOnlyRef ) return;
      }
    } else color = GetColorZ( z );
  }
  if ( width > 0 )fMarker.SetMarkerSize( width );
  else fMarker.SetMarkerSize( .3 );
  fMarker.SetMarkerColor( color );
  double vx, vy;
  Slice2View( x, y, &vx, &vy );
  fYX->cd();
  fMarker.DrawMarker( vx, vy );
  fZX->cd();
  fMarker.DrawMarker( z, vy );
}

void AliHLTTPCCADisplay::DrawSliceHits( int color, Size_t width )
{

  // draw hits

  for ( int iRow = 0; iRow < fSlice->Param().NRows(); iRow++ ) {
    const AliHLTTPCCARow &row = fSlice->Row( iRow );
    for ( int ih = 0; ih < row.NHits(); ih++ ) {
      DrawSliceHit( iRow, ih, color, width );
    }
  }
}


void AliHLTTPCCADisplay::DrawSliceLink( int iRow, int iHit, int colorUp, int colorDn, int width )
{
  // draw link between clusters

  if ( !fPerf || !fGB ) return;
  AliHLTTPCCAGBTracker &tracker = *fGB;
  if ( width < 0 ) width = 1.;
  fLine.SetLineWidth( width );
  int colUp = colorUp >= 0 ? colorUp : kMagenta;
  int colDn = colorDn >= 0 ? colorDn : kBlack;
  if ( iRow < 2 || iRow >= fSlice->Param().NRows() - 2 ) return;

  const AliHLTTPCCARow& row = fSlice->Row( iRow );
  const AliHLTTPCCARow& rowUp = fSlice->Row( iRow + 2 );
  const AliHLTTPCCARow& rowDn = fSlice->Row( iRow - 2 );

  int id = fSlice->HitInputID( row, iHit );
  const AliHLTTPCCAGBHit &h = tracker.Hits()[tracker.FirstSliceHit()[fSlice->Param().ISlice()] + id];
  short iUp = fSlice->HitLinkUpData( row, iHit );
  short iDn = fSlice->HitLinkDownData( row, iHit );

  if ( iUp >= 0 ) {
    int id1 = fSlice->HitInputID( rowUp, iUp );
    const AliHLTTPCCAGBHit &h1 = tracker.Hits()[tracker.FirstSliceHit()[fSlice->Param().ISlice()] + id1];
    double vx, vy, vx1, vy1;
    Slice2View( h.X(), h.Y(), &vx, &vy );
    Slice2View( h1.X(), h1.Y(), &vx1, &vy1 );
    fLine.SetLineColor( colUp );
    fYX->cd();
    fLine.DrawLine( vx - .1, vy, vx1 - .1, vy1 );
    fZX->cd();
    fLine.DrawLine( h.Z() - 1., vy, h1.Z() - 1., vy1 );
  }
  if ( iDn >= 0 ) {
    int id1 = fSlice->HitInputID( rowDn, iDn );
    const AliHLTTPCCAGBHit &h1 = tracker.Hits()[tracker.FirstSliceHit()[fSlice->Param().ISlice()] + id1];
    double vx, vy, vx1, vy1;
    Slice2View( h.X(), h.Y(), &vx, &vy );
    Slice2View( h1.X(), h1.Y(), &vx1, &vy1 );
    fLine.SetLineColor( colDn );
    fYX->cd();
    fLine.DrawLine( vx + .1, vy, vx1 + .1, vy1 );
    fZX->cd();
    fLine.DrawLine( h.Z() + 1., vy, h1.Z() + 1., vy1 );
  }
}


void AliHLTTPCCADisplay::DrawSliceLinks( int colorUp, int colorDn, int width )
{
  // draw links between clusters

  for ( int iRow = 1; iRow < fSlice->Param().NRows() - 1; iRow++ ) {
    const AliHLTTPCCARow& row = fSlice->Row( iRow );
    for ( int ih = 0; ih < row.NHits(); ih++ ) {
      DrawSliceLink( iRow, ih, colorUp, colorDn, width );
    }
  }
}



int AliHLTTPCCADisplay::GetTrackMC( const AliHLTTPCCADisplayTmpHit *vHits, int NHits )
{
  // get MC label for the track

  AliHLTTPCCAGBTracker &tracker = *fGB;

  int label = -1;
  double purity = 0;
  int *lb = new int[NHits*3];
  int nla = 0;
  //std::cout<<"\n\nTrack hits mc: "<<std::endl;
  for ( int ihit = 0; ihit < NHits; ihit++ ) {
    const AliHLTTPCCAGBHit &h = tracker.Hits()[vHits[ihit].ID()];
    AliHLTTPCCAPerformance::AliHLTTPCCAHitLabel &l = fPerf->HitLabels()[h.ID()];
    if ( l.fLab[0] >= 0 ) lb[nla++] = l.fLab[0];
    if ( l.fLab[1] >= 0 ) lb[nla++] = l.fLab[1];
    if ( l.fLab[2] >= 0 ) lb[nla++] = l.fLab[2];
    //std::cout<<ihit<<":  "<<l.fLab[0]<<" "<<l.fLab[1]<<" "<<l.fLab[2]<<std::endl;
  }
  sort( lb, lb + nla );
  int labmax = -1, labcur = -1, lmax = 0, lcurr = 0, nh = 0;
  //std::cout<<"MC track IDs :"<<std::endl;
  for ( int i = 0; i < nla; i++ ) {
    if ( lb[i] != labcur ) {
      if ( 0 && i > 0 && lb[i-1] >= 0 ) {
        AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lb[i-1]];
        std::cout << lb[i-1] << ": nhits=" << nh << ", pdg=" << mc.PDG() << ", Pt=" << mc.Pt() << ", P=" << mc.P()
                  << ", par=" << mc.Par()[0] << " " << mc.Par()[1] << " " << mc.Par()[2]
                  << " " << mc.Par()[3] << " " << mc.Par()[4] << " " << mc.Par()[5] << " " << mc.Par()[6] << std::endl;

      }
      nh = 0;
      if ( labcur >= 0 && lmax < lcurr ) {
        lmax = lcurr;
        labmax = labcur;
      }
      labcur = lb[i];
      lcurr = 0;
    }
    lcurr++;
    nh++;
  }
  if ( 0 && nla - 1 > 0 && lb[nla-1] >= 0 ) {
    AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lb[nla-1]];
    std::cout << lb[nla-1] << ": nhits=" << nh << ", pdg=" << mc.PDG() << ", Pt=" << mc.Pt() << ", P=" << mc.P()
              << ", par=" << mc.Par()[0] << " " << mc.Par()[1] << " " << mc.Par()[2]
              << " " << mc.Par()[3] << " " << mc.Par()[4] << " " << mc.Par()[5] << " " << mc.Par()[6] << std::endl;

  }
  if ( labcur >= 0 && lmax < lcurr ) {
    lmax = lcurr;
    labmax = labcur;
  }
  lmax = 0;
  for ( int ihit = 0; ihit < NHits; ihit++ ) {
    const AliHLTTPCCAGBHit &h = tracker.Hits()[vHits[ihit].ID()];
    AliHLTTPCCAPerformance::AliHLTTPCCAHitLabel &l = fPerf->HitLabels()[h.ID()];
    if ( l.fLab[0] == labmax || l.fLab[1] == labmax || l.fLab[2] == labmax
       ) lmax++;
  }
  label = labmax;
  purity = ( ( NHits > 0 ) ? double( lmax ) / double( NHits ) : 0 );
  if ( lb ) delete[] lb;
  if ( purity < .9 ) label = -1;
  return label;
}

bool AliHLTTPCCADisplay::DrawTrack( AliHLTTPCCATrackParam t, double Alpha, const AliHLTTPCCADisplayTmpHit *vHits,
                                    int NHits, int color, int width, bool pPoint )
{
  // draw track

  if ( NHits < 2 ) return 0;

  AliHLTTPCCAGBTracker &tracker = *fGB;
  if ( width < 0 ) width = 2;

  if ( fDrawOnlyRef ) {
    int lab = GetTrackMC( vHits, NHits );
    if ( lab < 0 ) return 0;
    AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lab];
    if ( mc.P() < 1 ) return 0;
  }

  if ( color < 0 ) {
    //color = GetColorZ( (vz[0]+vz[mHits-1])/2. );
    //color = GetColorK(t.GetKappa());
    int lab = GetTrackMC( vHits, NHits );
    color = GetColor( lab + 1 );
    if ( lab >= 0 ) {
      AliHLTTPCCAMCTrack &mc = fPerf->MCTracks()[lab];
      if ( mc.P() >= 1. ) color = kRed;
    }
  }

  if ( t.SinPhi() > .999 )  t.SetSinPhi( .999 );
  else if ( t.SinPhi() < -.999 )  t.SetSinPhi( -.999 );

  //  int iSlice = fSlice->Param().ISlice();

  //sort(vHits, vHits + NHits, AliHLTTPCCADisplayTmpHit::CompareHitZ );

  double vx[2000], vy[2000], vz[2000];
  int mHits = 0;

  //int oldSlice = -1;
  double alpha = ( TMath::Abs( Alpha + 1 ) < 1.e-4 ) ? fSlice->Param().Alpha() : Alpha;
  AliHLTTPCCATrackParam tt = t;

  for ( int iHit = 0; iHit < NHits; iHit++ ) {

    const AliHLTTPCCAGBHit &h = tracker.Hits()[vHits[iHit].ID()];

    double hCos = TMath::Cos( alpha - tracker.Slices()[h.ISlice()].Param().Alpha() );
    double hSin = TMath::Sin( alpha - tracker.Slices()[h.ISlice()].Param().Alpha() );
    double x0 = h.X(), y0 = h.Y(), z1 = h.Z();
    double x1 = x0 * hCos + y0 * hSin;
    double y1 = y0 * hCos - x0 * hSin;

    {
      double dx = x1 - tt.X();
      double dy = y1 - tt.Y();
      if ( dx*dx + dy*dy > 1. ) {
        double dalpha = TMath::ATan2( dy, dx );
        if ( tt.Rotate( dalpha ) ) {
          alpha += dalpha;
          hCos = TMath::Cos( alpha - tracker.Slices()[h.ISlice()].Param().Alpha() );
          hSin = TMath::Sin( alpha - tracker.Slices()[h.ISlice()].Param().Alpha() );
          x1 = x0 * hCos + y0 * hSin;
          y1 = y0 * hCos - x0 * hSin;
        }
      }
    }
    SetSliceTransform( alpha );

    //t.GetDCAPoint( x1, y1, z1, x1, y1, z1 );
    std::cout << "mark 3" << std::endl;
    bool ok = tt.TransportToX( x1, .999 );
    std::cout << "mark 4" << std::endl;
    if ( 1 || ok ) {
      x1 = tt.X();
      y1 = tt.Y();
      z1 = tt.Z();
    }

    Slice2View( x1, y1, &x1, &y1 );
    vx[mHits] = x1;
    vy[mHits] = y1;
    vz[mHits] = z1;
    mHits++;
    for ( int j = 0; j < 0; j++ ) {
      x0 = h.X() + j; y0 = h.Y(); z1 = h.Z();
      x1 = x0 * hCos + y0 * hSin;
      y1 = y0 * hCos - x0 * hSin;
      ok = tt.TransportToX( x1, .999 );
      if ( ok ) {
        x1 = tt.X();
        y1 = tt.Y();
        z1 = tt.Z();
      }

      Slice2View( x1, y1, &x1, &y1 );
      vx[mHits] = x1;
      vy[mHits] = y1;
      vz[mHits] = z1;
      mHits++;
    }
  }
  if ( pPoint ) {
    double x1 = t.X(), y1 = t.Y(), z1 = t.Z();
    double a = ( TMath::Abs( Alpha + 1 ) < 1.e-4 ) ? fSlice->Param().Alpha() : Alpha;
    SetSliceTransform( a );

    Slice2View( x1, y1, &x1, &y1 );
    double dx = x1 - vx[0];
    double dy = y1 - vy[0];
    //std::cout<<x1<<" "<<y1<<" "<<vx[0]<<" "<<vy[0]<<" "<<dx<<" "<<dy<<std::endl;
    double d0 = dx * dx + dy * dy;
    dx = x1 - vx[mHits-1];
    dy = y1 - vy[mHits-1];
    //std::cout<<x1<<" "<<y1<<" "<<vx[mHits-1]<<" "<<vy[mHits-1]<<" "<<dx<<" "<<dy<<std::endl;
    double d1 = dx * dx + dy * dy;
    //std::cout<<"d0, d1="<<d0<<" "<<d1<<std::endl;
    if ( d1 < d0 ) {
      vx[mHits] = x1;
      vy[mHits] = y1;
      vz[mHits] = z1;
      mHits++;
    } else {
      for ( int i = mHits; i > 0; i-- ) {
        vx[i] = vx[i-1];
        vy[i] = vy[i-1];
        vz[i] = vz[i-1];
      }
      vx[0] = x1;
      vy[0] = y1;
      vz[0] = z1;
      mHits++;
    }
  }


  fLine.SetLineColor( color );
  fLine.SetLineWidth( width );
  fArc.SetFillStyle( 0 );
  fArc.SetLineColor( color );
  fArc.SetLineWidth( width );
  TPolyLine pl;
  pl.SetLineColor( color );
  pl.SetLineWidth( width );
  TPolyLine plZ;
  plZ.SetLineColor( color );
  plZ.SetLineWidth( width );

  fMarker.SetMarkerSize( width / 2. );
  fMarker.SetMarkerColor( color );

  fYX->cd();
  pl.DrawPolyLine( mHits, vx, vy );
  {
    fMarker.DrawMarker( vx[0], vy[0] );
    fMarker.DrawMarker( vx[mHits-1], vy[mHits-1] );
  }
  fZX->cd();
  plZ.DrawPolyLine( mHits, vz, vy );
  fMarker.DrawMarker( vz[0], vy[0] );
  fMarker.DrawMarker( vz[mHits-1], vy[mHits-1] );

  fLine.SetLineWidth( 1 );
  return 1;
}


bool AliHLTTPCCADisplay::DrawTracklet( AliHLTTPCCATrackParam &track, const int *hitstore, int color, int width, bool pPoint )
{
  // draw tracklet
  AliHLTTPCCAGBTracker &tracker = *fGB;
  AliHLTTPCCADisplayTmpHit vHits[200];
  int nHits = 0;
  for ( int iRow = 0; iRow < fSlice->Param().NRows(); iRow++ ) {
    int iHit = hitstore[iRow];
    if ( iHit < 0 ) continue;
    const AliHLTTPCCARow &row = fSlice->Row( iRow );
    int id = fSlice->HitInputID( row, iHit );
    int iGBHit = tracker.FirstSliceHit()[fSlice->Param().ISlice()] + id;
    const AliHLTTPCCAGBHit &h = tracker.Hits()[iGBHit];
    vHits[nHits].SetID( iGBHit );
    vHits[nHits].SetS( 0 );
    vHits[nHits].SetZ( h.Z() );
    nHits++;
  }
  return DrawTrack( track, -1, vHits, nHits, color, width, pPoint );
}


void AliHLTTPCCADisplay::DrawSliceOutTrack( AliHLTTPCCATrackParam &t, double alpha, int itr, int color, int width )
{
  // draw slice track

  AliHLTTPCCAOutTrack &track = fSlice->OutTracks()[itr];
  if ( track.NHits() < 2 ) return;

  AliHLTTPCCAGBTracker &tracker = *fGB;
  AliHLTTPCCADisplayTmpHit vHits[200];

  for ( int ih = 0; ih < track.NHits(); ih++ ) {
    int id = tracker.FirstSliceHit()[fSlice->Param().ISlice()] + fSlice->OutTrackHits()[track.FirstHitRef()+ih];
    const AliHLTTPCCAGBHit &h = tracker.Hits()[id];
    vHits[ih].SetID( id );
    vHits[ih].SetS( 0 );
    vHits[ih].SetZ( h.Z() );
  }

  DrawTrack( t, alpha, vHits, track.NHits(), color, width, 1 );
}

void AliHLTTPCCADisplay::DrawSliceOutTrack( int itr, int color, int width )
{
  // draw slice track

  AliHLTTPCCAOutTrack &track = fSlice->OutTracks()[itr];
  if ( track.NHits() < 2 ) return;

  AliHLTTPCCAGBTracker &tracker = *fGB;
  AliHLTTPCCADisplayTmpHit vHits[200];

  for ( int ih = 0; ih < track.NHits(); ih++ ) {
    int id = tracker.FirstSliceHit()[fSlice->Param().ISlice()] + fSlice->OutTrackHits()[track.FirstHitRef()+ih];
    const AliHLTTPCCAGBHit &h = tracker.Hits()[id];
    vHits[ih].SetID( id );
    vHits[ih].SetS( 0 );
    vHits[ih].SetZ( h.Z() );
  }

  DrawTrack( track.StartPoint(), -1, vHits, track.NHits(), color, width );
}


void AliHLTTPCCADisplay::DrawSliceTrack( int itr, int color )
{
  // draw slice track

  const AliHLTTPCCATrack &track = fSlice->Tracks()[itr];
  if ( track.NHits() < 2 ) return;

  AliHLTTPCCAGBTracker &tracker = *fGB;
  AliHLTTPCCADisplayTmpHit vHits[200];
  for ( int ith = 0; ith < track.NHits(); ith++ ) {
    AliHLTTPCCAHitId ic = ( fSlice->TrackHits()[track.FirstHitID()+ith] );
    const AliHLTTPCCARow &row = fSlice->Row( ic );
    int ih = ic.HitIndex();
    int id = fSlice->HitInputID( row, ih );
    int gbID = tracker.FirstSliceHit()[fSlice->Param().ISlice()] + id;
    const AliHLTTPCCAGBHit &h = tracker.Hits()[gbID];
    vHits[ith].SetID( gbID );
    vHits[ith].SetS( 0 );
    vHits[ith].SetZ( h.Z() );
  }

  DrawTrack( track.Param(), -1, vHits, track.NHits(), color, -1 );
  //track.Param().Print();
}


void AliHLTTPCCADisplay::DrawGBTrack( int itr, int color, int width )
{
  // draw global track

  AliHLTTPCCAGBTracker &tracker = *fGB;
  AliHLTTPCCADisplayTmpHit vHits[1000];

  AliHLTTPCCAGBTrack &track = tracker.Tracks()[itr];
  if ( track.NHits() < 2 ) return;

  for ( int ih = 0; ih < track.NHits(); ih++ ) {
    int i = tracker.TrackHits()[ track.FirstHitRef() + ih];
    const AliHLTTPCCAGBHit &h = tracker.Hits()[i];
    vHits[ih].SetID( i );
    vHits[ih].SetS( 0 );
    vHits[ih].SetZ( h.Z() );
  }

  DrawTrack( track.Param(), track.Alpha(), vHits, track.NHits(), color, width );
}


void AliHLTTPCCADisplay::DrawGBTrackFast( AliHLTTPCCAGBTracker &tracker, int itr, int color )
{
  // draw global track

  AliHLTTPCCAGBTrack &track = tracker.Tracks()[itr];
  if ( track.NHits() < 2 ) return;
  int width = 1;

  AliHLTTPCCADisplayTmpHit *vHits = new AliHLTTPCCADisplayTmpHit[track.NHits()];
  AliHLTTPCCATrackParam t = track.Param();

  for ( int ih = 0; ih < track.NHits(); ih++ ) {
    int i = tracker.TrackHits()[ track.FirstHitRef() + ih];
    const AliHLTTPCCAGBHit *h = &( tracker.Hits()[i] );
    vHits[ih].SetID( i );
    vHits[ih].SetS( 0 );
    vHits[ih].SetZ( h->Z() );
  }

  sort( vHits, vHits + track.NHits(), AliHLTTPCCADisplayTmpHit::CompareHitZ );
  int colorY = color;
  {
    const AliHLTTPCCAGBHit &h1 = tracker.Hits()[ vHits[0].ID()];
    const AliHLTTPCCAGBHit &h2 = tracker.Hits()[ vHits[track.NHits()-1].ID()];
    if ( color < 0 ) color = GetColorZ( ( h1.Z() + h2.Z() ) / 2. );
    double gx1, gy1, gx2, gy2;
    Slice2View( h1.X(), h1.Y(), &gx1, &gy1 );
    Slice2View( h2.X(), h2.Y(), &gx2, &gy2 );
    if ( colorY < 0 ) colorY = GetColorY( ( gy1 + gy2 ) / 2. );
    color = colorY = GetColorK( t.GetQPt() );
  }

  fMarker.SetMarkerColor( color );//kBlue);
  fMarker.SetMarkerSize( 1. );
  fLine.SetLineColor( color );
  fLine.SetLineWidth( width );
  fArc.SetFillStyle( 0 );
  fArc.SetLineColor( color );
  fArc.SetLineWidth( width );
  TPolyLine pl;
  pl.SetLineColor( colorY );
  pl.SetLineWidth( width );

  int oldSlice = -1;
  double alpha = track.Alpha();
  // YX
  {

    const AliHLTTPCCAGBHit &h1 = tracker.Hits()[vHits[0].ID()];
    const AliHLTTPCCAGBHit &h2 = tracker.Hits()[vHits[track.NHits()-1].ID()];
    float x1, y1, z1, x2, y2, z2;
    double vx1, vy1, vx2, vy2;

    if ( h1.ISlice() != oldSlice ) {
      t.Rotate( tracker.Slices()[h1.ISlice()].Param().Alpha() - alpha );
      oldSlice = h1.ISlice();
      alpha = tracker.Slices()[h1.ISlice()].Param().Alpha();
      SetSliceTransform( &( tracker.Slices()[oldSlice] ) );
    }
    t.GetDCAPoint( h1.X(), h1.Y(), h1.Z(), x1, y1, z1, fSlice->Param().Bz()  );
    Slice2View( x1, y1, &vx1, &vy1 );

    if ( h2.ISlice() != oldSlice ) {
      t.Rotate( tracker.Slices()[h2.ISlice()].Param().Alpha() - alpha );
      oldSlice = h2.ISlice();
      alpha = tracker.Slices()[h2.ISlice()].Param().Alpha();
      SetSliceTransform( &( tracker.Slices()[oldSlice] ) );
    }
    t.GetDCAPoint( h2.X(), h2.Y(), h2.Z(), x2, y2, z2, fSlice->Param().Bz()  );
    Slice2View( x2, y2, &vx2, &vy2 );

    double x0 = t.GetX();
    double y0 = t.GetY();
    double sinPhi = t.GetSinPhi();
    double k = t.GetKappa( fSlice->Param().Bz() );
    double ex = t.GetCosPhi();
    double ey = sinPhi;

    if ( TMath::Abs( k ) > 1.e-4 ) {

      fYX->cd();

      double r = 1 / TMath::Abs( k );
      double xc = x0 - ey * ( 1 / k );
      double yc = y0 + ex * ( 1 / k );

      double vx, vy;
      Slice2View( xc, yc, &vx, &vy );

      double a1 = TMath::ATan2( vy1 - vy, vx1 - vx ) / TMath::Pi() * 180.;
      double a2 = TMath::ATan2( vy2 - vy, vx2 - vx ) / TMath::Pi() * 180.;
      if ( a1 < 0 ) a1 += 360;
      if ( a2 < 0 ) a2 += 360;
      if ( a2 < a1 ) a2 += 360;
      double da = TMath::Abs( a2 - a1 );
      if ( da > 360 ) da -= 360;
      if ( da > 180 ) {
        da = a1;
        a1 = a2;
        a2 = da;
        if ( a2 < a1 ) a2 += 360;
      }
      fArc.DrawArc( vx, vy, r, a1, a2, "only" );
      //fArc.DrawArc(vx,vy,r, 0,360,"only");
    } else {
      fYX->cd();
      fLine.DrawLine( vx1, vy1, vx2, vy2 );
    }
  }

  // ZX
  AliHLTResizableArray<double> py( track.NHits() ), pz( track.NHits() );

  for ( int iHit = 0; iHit < track.NHits(); iHit++ ) {

    const AliHLTTPCCAGBHit &h1 = tracker.Hits()[vHits[iHit].ID()];
    float x1, y1, z1;
    double vx1, vy1;
    if ( h1.ISlice() != oldSlice ) {
      t.Rotate( tracker.Slices()[h1.ISlice()].Param().Alpha() - alpha );
      oldSlice = h1.ISlice();
      alpha = tracker.Slices()[h1.ISlice()].Param().Alpha();
      SetSliceTransform( &( tracker.Slices()[oldSlice] ) );
    }
    t.GetDCAPoint( h1.X(), h1.Y(), h1.Z(), x1, y1, z1, fSlice->Param().Bz()  );
    Slice2View( x1, y1, &vx1, &vy1 );
    py[iHit] = vy1;
    pz[iHit] = z1;
  }


  fZX->cd();
  pl.DrawPolyLine( track.NHits(), pz.Data(), py.Data() );

  fLine.SetLineWidth( 1 );
  delete[] vHits;
}





#ifdef XXXX




void AliHLTTPCCADisplay::DrawMergedHit( int iRow, int iHit, int color )
{
  // connect two cells on display

#ifdef XXX

  const AliHLTTPCCARow &row = fSlice->Row( iRow );
  const AliHLTTPCCAHit &h = row.Hits()[iHit];
  const AliHLTTPCCAHit &hyz = row.HitsYZ()[iHit];

  double x = row.X();
  double y = hyz.Y();
  double z = hyz.Z();
  double x1 = x, x2 = x;
  double y1 = y, y2 = y;
  double z1 = z, z2 = z;
  int iRow1 = iRow, iHit1 = iHit;
  int iRow2 = iRow, iHit2 = iHit;

  if ( fSlice->HitLinksDown()[] >= 0 ) {
    iRow1 = iRow - 1;
    iHit1 = h.LinkDown();
    const AliHLTTPCCARow &row1 = fSlice->Rows()[iRow1];
    AliHLTTPCCAHitYZ &h1 = row1.HitsYZ()[iHit1];
    x1 = row1.X();
    y1 = h1.Y();
    z1 = h1.Z();
  }
  if ( h.LinkUp() >= 0 ) {
    iRow2 = iRow + 1;
    iHit2 = h.LinkUp();
    const AliHLTTPCCARow &row2 = fSlice->Rows()[iRow2];
    AliHLTTPCCAHitYZ &h2 = row2.HitsYZ()[iHit2];
    x2 = row2.X();
    y2 = h2.Y();
    z2 = h2.Z();
  }
  if ( color < 0 ) color = GetColorZ( ( z + z1 + z2 ) / 3. );


  Slice2View( x, y, &x, &y );
  Slice2View( x1, y1, &x1, &y1 );
  Slice2View( x2, y2, &x2, &y2 );

  double lx[] = { x1, x, x2 };
  double ly[] = { y1, y, y2 };
  double lz[] = { z1, z, z2 };

  fPLine.SetLineColor( color );
  fPLine.SetLineWidth( 1 );
  //fPLine.SetFillColor(color);
  fPLine.SetFillStyle( -1 );

  fYX->cd();
  fPLine.DrawPolyLine( 3, lx, ly );
  fZX->cd();
  fPLine.DrawPolyLine( 3, lz, ly );
  DrawHit( iRow, iHit, color );
  DrawHit( iRow1, iHit1, color );
  DrawHit( iRow2, iHit2, color );
#endif
}


void AliHLTTPCCADisplay::DrawTrack( AliHLTTPCCATrack &track, int color, bool DrawHits )
{
  // draw track

  if ( track.NHits() < 2 ) return;
  int width = 2;

  AliHLTTPCCADisplayTmpHit *vHits = new AliHLTTPCCADisplayTmpHit[track.NHits()];
  AliHLTTPCCATrackParam &t = track.Param();

  int iID = track.FirstHitID();
  int nhits = 0;
  {
    int iHit = 0;
    for ( int ih = 0; ih < track.NHits(); ih++ ) {
      int i = fSlice->TrackHits()[iID];
      const AliHLTTPCCAHit *h = &( fSlice->ID2Hit( i ) );
      const AliHLTTPCCARow &row = fSlice->ID2Row( i );
      vHits[iHit].ID() = i;
      vHits[iHit].S() = t.GetS( row.X(), h->Y() );
      vHits[iHit].Z() = h->Z();
      iHit++;
      nhits++;
      iID++;
    }
  }
  sort( vHits, vHits + track.NHits(), AliHLTTPCCADisplayTmpHit::CompareHitZ );
  //cout<<"Draw track, nhits = "<<nhits<<endl;
  {
    const AliHLTTPCCAHit &c1 = fSlice->ID2Hit( vHits[0].ID() );
    const AliHLTTPCCAHit &c2 = fSlice->ID2Hit( vHits[track.NHits()-1].ID() );
    if ( color < 0 ) color = GetColorZ( ( c1.Z() + c2.Z() ) / 2. );
  }

  fMarker.SetMarkerColor( color );//kBlue);
  fMarker.SetMarkerSize( 1. );
  /*
  for( int i=0; i<3; i++){
    const AliHLTTPCCAHit &c1 = fSlice->ID2Hit(track.HitID()[i]);
    const AliHLTTPCCARow &row1 = fSlice->ID2Row(track.HitID()[i]);
    double vx1, vy1;
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
  for ( int iHit = 0; iHit < track.NHits() - 1; iHit++ ) {
    const AliHLTTPCCAHit &c1 = fSlice->ID2Hit( vHits[iHit].ID() );
    const AliHLTTPCCAHit &c2 = fSlice->ID2Hit( vHits[iHit+1].ID() );
    const AliHLTTPCCARow &row1 = fSlice->ID2Row( vHits[iHit].ID() );
    const AliHLTTPCCARow &row2 = fSlice->ID2Row( vHits[iHit+1].ID() );
    float x1, y1, z1, x2, y2, z2;
    t.GetDCAPoint( row1.X(), c1.Y(), c1.Z(), x1, y1, z1, fSlice->Param().Bz()  );
    t.GetDCAPoint( row2.X(), c2.Y(), c2.Z(), x2, y2, z2, fSlice->Param().Bz()  );

    //if( color<0 ) color = GetColorZ( (z1+z2)/2. );
    double vx1, vy1, vx2, vy2;
    Slice2View( x1, y1, &vx1, &vy1 );
    Slice2View( x2, y2, &vx2, &vy2 );

    fLine.SetLineColor( color );
    fLine.SetLineWidth( width );

    double x0 = t.GetX();
    double y0 = t.GetY();
    double sinPhi = t.GetSinPhi();
    double k = t.GetKappa();
    double ex = t.GetCosPhi();
    double ey = sinPhi;

    if ( TMath::Abs( k ) > 1.e-4 ) {

      fArc.SetFillStyle( 0 );
      fArc.SetLineColor( color );
      fArc.SetLineWidth( width );

      fYX->cd();

      double r = 1 / TMath::Abs( k );
      double xc = x0 - ey * ( 1 / k );
      double yc = y0 + ex * ( 1 / k );

      double vx, vy;
      Slice2View( xc, yc, &vx, &vy );

      double a1 = TMath::ATan2( vy1 - vy, vx1 - vx ) / TMath::Pi() * 180.;
      double a2 = TMath::ATan2( vy2 - vy, vx2 - vx ) / TMath::Pi() * 180.;
      if ( a1 < 0 ) a1 += 360;
      if ( a2 < 0 ) a2 += 360;
      if ( a2 < a1 ) a2 += 360;
      double da = TMath::Abs( a2 - a1 );
      if ( da > 360 ) da -= 360;
      if ( da > 180 ) {
        da = a1;
        a1 = a2;
        a2 = da;
        if ( a2 < a1 ) a2 += 360;
      }
      fArc.DrawArc( vx, vy, r, a1, a2, "only" );
      //fArc.DrawArc(vx,vy,r, 0,360,"only");
    } else {
      fYX->cd();
      fLine.DrawLine( vx1, vy1, vx2, vy2 );
    }
  }

  for ( int iHit = 0; iHit < track.NHits() - 1; iHit++ ) {
    const AliHLTTPCCAHit &c1 = fSlice->ID2Hit( vHits[iHit].ID() );
    const AliHLTTPCCAHit &c2 = fSlice->ID2Hit( vHits[iHit+1].ID() );
    const AliHLTTPCCARow &row1 = fSlice->ID2Row( vHits[iHit].ID() );
    const AliHLTTPCCARow &row2 = fSlice->ID2Row( vHits[iHit+1].ID() );

    //if( DrawHits ) ConnectHits( fSlice->ID2IRow(vHits[iHit].ID()),c1,
    //fSlice->ID2IRow(vHits[iHit+1].ID()),c2, color );
    float x1, y1, z1, x2, y2, z2;
    t.GetDCAPoint( row1.X(), c1.Y(), c1.Z(), x1, y1, z1, fSlice->Param().Bz()  );
    t.GetDCAPoint( row2.X(), c2.Y(), c2.Z(), x2, y2, z2, fSlice->Param().Bz()  );

    double vx1, vy1, vx2, vy2;
    Slice2View( x1, y1, &vx1, &vy1 );
    Slice2View( x2, y2, &vx2, &vy2 );

    fLine.SetLineColor( color );
    fLine.SetLineWidth( width );

    fZX->cd();
    fLine.DrawLine( z1, vy1, z2, vy2 );
  }
  fLine.SetLineWidth( 1 );
  delete[] vHits;
}


void AliHLTTPCCADisplay::DrawTrackletPoint( AliHLTTPCCATrackParam &t, int color )
{
  // draw tracklet point

  double x = t.GetX();
  double y = t.GetY();
  double sinPhi = t.GetSinPhi();
  double z = t.GetZ();
  double dzds = t.GetDzDs();
  double ex = t.GetCosPhi();
  double ey = sinPhi;

  int width = 1;

  if ( color < 0 ) color = GetColorZ( t.GetZ() );

  fMarker.SetMarkerColor( color );
  fMarker.SetMarkerSize( .5 );
  fLine.SetLineWidth( width );
  fLine.SetLineColor( color );

  double vx, vy, vex, vey, vdx, vdy;
  double dz = TMath::Sqrt( t.GetErr2Z() );
  Slice2View( x, y, &vx, &vy );
  Slice2View( ex, ey, &vex, &vey );
  Slice2View( 0, TMath::Sqrt( t.GetErr2Y() )*3.5, &vdx, &vdy );
  double d = TMath::Sqrt( vex * vex + vey * vey );
  vex /= d;
  vey /= d;
  fYX->cd();
  fMarker.DrawMarker( vx, vy );
  fLine.DrawLine( vx, vy, vx + vex*4, vy + vey*4 );
  fLine.DrawLine( vx - vdx, vy - vdy, vx + vdx, vy + vdy );
  fZX->cd();
  fMarker.DrawMarker( z, vy );
  fLine.DrawLine( z, vy, z + dzds*4, vy + vey*4 );
  fLine.DrawLine( z - 3.5*dz, vy - vdy, z + 3.5*dz, vy + vdy );
  fLine.SetLineWidth( 1 );
}
#endif //XXXX
