//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCADISPLAY_H
#define ALIHLTTPCCADISPLAY_H

class AliHLTTPCCATracker;
class AliHLTTPCCACell;
class AliHLTTPCCATrack;

#include "TCanvas.h"
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
  void Clear();
  void Ask();
  void SetSectorView();
  void SetTPCView();
  void SetCurrentSector( AliHLTTPCCATracker *sec ); 

  Int_t GetColor( Double_t z ) const ;
  void Global2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const ;
  void Sec2View( Double_t x, Double_t y, Double_t *xv, Double_t *yv ) const ;

  void DrawTPC();
  void DrawSector( AliHLTTPCCATracker *sec ); 

  void DrawHit( Int_t iRow,Int_t iHit, Int_t color=-1 );
  void DrawCell( Int_t iRow, AliHLTTPCCACell &cell, Int_t width=1, Int_t color=-1 );
  void DrawCell( Int_t iRow, Int_t iCell, Int_t width=1, Int_t color=-1 );

  void ConnectCells( Int_t iRow1, AliHLTTPCCACell &cell1, Int_t iRow2, AliHLTTPCCACell &cell2, Int_t color=-1 );

  void DrawTrack( AliHLTTPCCATrack &track, Int_t color=-1 );

 protected:

  TCanvas *fXY, *fZY;               // two views
  Bool_t fAsk;                      // flag to ask for the pressing key
  Bool_t fSectorView;               // switch between sector/TPC zoomv
  AliHLTTPCCATracker *fSector;      // current CA tracker, includes sector geometry
  Double_t fCos, fSin, fZMin, fZMax;// view parameters
  Double_t fRInnerMin, fRInnerMax, fROuterMin, fROuterMax,fTPCZMin, fTPCZMax; // view parameters

  TArc fArc;       // parameters of drawing objects are copied from this members
  TLine fLine;     //!
  TPolyLine fPLine;//!
  TMarker fMarker; //!
  TBox fBox;       //!
  TCrown fCrown;   //!
  TLatex fLatex;   //!

  ClassDef(AliHLTTPCCADisplay,1);

};

#endif
