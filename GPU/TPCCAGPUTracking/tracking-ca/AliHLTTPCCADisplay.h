//-*- Mode: C++ -*-
// @(#) $Id$
//  *************************************************************************
//  This file is property of and copyright by the ALICE HLT Project         *
//  ALICE Experiment at CERN, All rights reserved.                          *
//  See cxx source for full Copyright notice                                *
//                                                                          *
//  AliHLTTPCCADisplay class is a debug utility.                            *
//  It is not used in the normal data processing.                           *
//                                                                          *
//***************************************************************************

#ifndef ALIHLTTPCCADISPLAY_H
#define ALIHLTTPCCADISPLAY_H


class AliHLTTPCCATracker;
class AliHLTTPCCATrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCAPerformance;

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

    class AliHLTTPCCADisplayTmpHit;

    static AliHLTTPCCADisplay &Instance();

    AliHLTTPCCADisplay();

    virtual ~AliHLTTPCCADisplay();

    void Init();
    void Update();
    void ClearView();
    void Ask();
    void SetSliceView();
    void SetTPCView();
    void SetCurrentSlice( AliHLTTPCCATracker *slice );
    void Set2Slices( AliHLTTPCCATracker * const slice );

    int GetColor( int i ) const;
    int GetColorZ( double z ) const ;
    int GetColorY( double y ) const ;
    int GetColorK( double k ) const ;
    void Global2View( double x, double y, double *xv, double *yv ) const ;
    void Slice2View( double x, double y, double *xv, double *yv ) const ;
    int GetTrackMC( const AliHLTTPCCADisplayTmpHit *vHits, int NHits );

    void DrawTPC();
    void DrawSlice( AliHLTTPCCATracker *slice, bool DrawRows = 0 );
    void DrawSliceOutTrack( int itr, int color = -1, int width = -1  );
    void DrawSliceOutTrack( AliHLTTPCCATrackParam &t, double Alpha, int itr, int color = -1, int width = -1  );
    void DrawSliceTrack( int itr, int color = -1 );
    bool DrawTrack( AliHLTTPCCATrackParam t, double Alpha, const AliHLTTPCCADisplayTmpHit *vHits,
                    int NHits, int color = -1, int width = -1, bool pPoint = 0 );

    bool DrawTracklet( AliHLTTPCCATrackParam &track, const int *hitstore, int color = -1, int width = -1, bool pPoint = 0 );

    void DrawSliceHit( int iRow, int iHit, int color = -1, Size_t width = -1 );
    void DrawSliceHits( int color = -1, Size_t width = -1 );
    void DrawSliceLinks( int colorUp = -1, int colorDn = -1, int width = -1 );
    void DrawSliceLink( int iRow, int iHit, int colorUp = -1, int colorDn = -1, int width = -1 );
    void SliceHitXYZ(int iRow, int iHit, double &x, double &y, double &z );


    void SetSliceTransform( double alpha );

    void SetSliceTransform( AliHLTTPCCATracker *slice );

    TCanvas *CanvasYX() const { return fYX; }
    TCanvas *CanvasZX() const { return fZX; }

  protected:

    TCanvas *fYX, *fZX;               // two views
    bool fAsk;                      // flag to ask for the pressing key
    bool fSliceView;               // switch between slice/TPC zoom
    AliHLTTPCCATracker *fSlice;      // current CA tracker, includes slice geometry
    AliHLTTPCCAPerformance *fPerf; // Performance class (mc labels etc)
    double fCos, fSin, fZMin, fZMax, fYMin, fYMax;// view parameters
    double fSliceCos, fSliceSin;        // current slice angle
    double fRInnerMin, fRInnerMax, fROuterMin, fROuterMax, fTPCZMin, fTPCZMax; // view parameters

    TArc fArc;       // parameters of drawing objects are copied from this members
    TLine fLine;     //!
    TPolyLine fPLine;//!
    TMarker fMarker; //!
    TBox fBox;       //!
    TCrown fCrown;   //!
    TLatex fLatex;   //!

    bool fDrawOnlyRef; // draw only clusters from ref. mc tracks

private:
  /// copy constructor prohibited
  AliHLTTPCCADisplay( const AliHLTTPCCADisplay& );
  /// assignment operator prohibited
  AliHLTTPCCADisplay& operator=( const AliHLTTPCCADisplay& ) const ;

};

#endif //ALIHLTTPCCADISPLAY_H
