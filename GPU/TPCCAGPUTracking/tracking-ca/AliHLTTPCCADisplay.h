//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project         * 
//* ALICE Experiment at CERN, All rights reserved.                          *
//* See cxx source for full Copyright notice                                *

//*                                                                         *
//*  AliHLTTPCCADisplay class is a debug utility.                           *
//*  It is not used in the normal data processing.                          *
//*                                                                         *

#ifndef ALIHLTTPCCADISPLAY_H
#define ALIHLTTPCCADISPLAY_H


class AliHLTTPCCATracker;
class AliHLTTPCCAGBTracker;
class AliHLTTPCCATrack;
class AliHLTTPCCATrackParam;
class TCanvas;
#include "TArc.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TBox.h"
#include "TCrown.h"
#include "TMarker.h"
#include "TLatex.h"

/**
 * @class AliHLTTPCCADisplay
 */
class AliHLTTPCCADisplay
{
 public:
  static AliHLTTPCCADisplay &Instance();
  
  AliHLTTPCCADisplay();
  AliHLTTPCCADisplay( const AliHLTTPCCADisplay& );
  AliHLTTPCCADisplay& operator=(const AliHLTTPCCADisplay&);

  virtual ~AliHLTTPCCADisplay();

  void Init();
  void Update();
  void ClearView();
  void Ask();
  void SetSliceView();
  void SetTPCView();
  void SetCurrentSlice( AliHLTTPCCATracker *slice ); 
  void Set2Slices( AliHLTTPCCATracker *slice );

  Int_t GetColor( Double_t z ) const ;
  Int_t GetColorY( Double_t y ) const ;
  Int_t GetColorK( Double_t k ) const ;
  void Global2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const ;
  void Slice2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const ;

  void DrawTPC();
  void DrawSlice( AliHLTTPCCATracker *slice ); 
  void DrawGBTrack( AliHLTTPCCAGBTracker &tracker, Int_t itr, Int_t color=-1 );
  void DrawGBTrackFast( AliHLTTPCCAGBTracker &tracker, Int_t itr, Int_t color=-1 );

  void DrawGBHit( AliHLTTPCCAGBTracker &tracker, Int_t iHit, Int_t color=-1 );

#ifdef XXXX

  void DrawHit( Int_t iRow,Int_t iHit, Int_t color=-1 );
  void DrawMergedHit( Int_t iRow, Int_t iHit, Int_t color=-1 );

  void DrawTrack( AliHLTTPCCATrack &track, Int_t color=-1, Bool_t DrawCells=1 );
  void DrawTrackletPoint( AliHLTTPCCATrackParam &t, Int_t color=-1 );
#endif // XXXX

  void SetSliceTransform( Double_t alpha );

  void SetSliceTransform( AliHLTTPCCATracker *slice );

  TCanvas *CanvasYX(){ return fYX; }
  TCanvas *CanvasZX(){ return fZX; }

 protected:

  TCanvas *fYX, *fZX;               // two views
  Bool_t fAsk;                      // flag to ask for the pressing key
  Bool_t fSliceView;               // switch between slice/TPC zoom
  AliHLTTPCCATracker *fSlice;      // current CA tracker, includes slice geometry
  Double_t fCos, fSin, fZMin, fZMax, fYMin, fYMax;// view parameters
  Double_t fSliceCos, fSliceSin;        // current slice angle
  Double_t fRInnerMin, fRInnerMax, fROuterMin, fROuterMax,fTPCZMin, fTPCZMax; // view parameters

  TArc fArc;       // parameters of drawing objects are copied from this members
  TLine fLine;     //!
  TPolyLine fPLine;//!
  TMarker fMarker; //!
  TBox fBox;       //!
  TCrown fCrown;   //!
  TLatex fLatex;   //!

  class AliHLTTPCCADisplayTmpHit{  

  public:
    Int_t &ID(){ return fHitID; }
    Double_t &S(){ return fS; }
    Double_t &Z(){ return fZ; }

    static Bool_t CompareHitDS( const AliHLTTPCCADisplayTmpHit &a, 
				 const AliHLTTPCCADisplayTmpHit  &b )
    {    
      return (a.fS < b.fS);
    }
    static Bool_t CompareHitZ( const AliHLTTPCCADisplayTmpHit &a, 
				 const AliHLTTPCCADisplayTmpHit  &b )
    {    
      return (a.fZ < b.fZ);
    }
  protected:
    Int_t fHitID; // cell ID
    Double_t fS;  // cell position on the XY track curve 
    Double_t fZ;  // cell Z position
  };


};

#endif
