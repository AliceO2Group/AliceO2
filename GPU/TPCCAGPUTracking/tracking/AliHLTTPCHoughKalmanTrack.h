// @(#) $Id$
// origin: hough/AliL3HoughKalmanTrack.h,v 1.1 Thu Mar 31 04:48:58 2005 UTC by cvetan 

#ifndef ALIHLTTPCHOUGHKALMANTRACK_H
#define ALIHLTTPCHOUGHKALMANTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
//             High Level Trigger TPC Hough Kalman Track Class
//
//        Origin: Cvetan Cheshkov, CERN, Cvetan.Cheshkov@cern.ch 
//-------------------------------------------------------------------------


/*****************************************************************************
 *                          October 11, 2004                                 *
 * The class inherits from the off-line AliTPCtrack class.                   *
 * It is used to transform AliHLTHoughTrack into AliTPCTrack, which is        *
 * then stored as AliESDtrack object in the ESD                              *
 *****************************************************************************/

#include <AliTPCtrack.h>

class AliHLTTPCHoughTrack;
class AliHLTTPCHoughTransformer;

class AliHLTTPCHoughKalmanTrack : public AliTPCtrack {
public:
  AliHLTTPCHoughKalmanTrack(const AliHLTTPCHoughTrack& t) throw (const Char_t *);

  ClassDef(AliHLTTPCHoughKalmanTrack,1)   //TPC TPC Hough track
};

#endif
