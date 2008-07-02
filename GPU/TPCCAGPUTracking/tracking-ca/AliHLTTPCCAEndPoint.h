//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

//*                                                                        *
//* The AliHLTTPCCAEndPoint class describes the end point of the track     *
//* it contains the fitted track parameters at this point                  *
//* The class is used for the matching of tracks withing one TPC slice     *
//*                                                                        *

#ifndef ALIHLTTPCCAENDPOINT_H
#define ALIHLTTPCCAENDPOINT_H


#include "Rtypes.h"
#include "AliHLTTPCCATrackParam.h"

/**
 * @class AliHLTTPCCAEndPoint
 */
class AliHLTTPCCAEndPoint
{
 public:
  
  AliHLTTPCCAEndPoint() :fCellID(0),fTrackID(0),fLink(0),fParam(){}
  Int_t &CellID()  { return fCellID; }
  Int_t &TrackID() { return fTrackID; }
  Int_t &Link()    { return fLink; }

  AliHLTTPCCATrackParam &Param(){ return fParam; };

 protected:

  Int_t fCellID;                //* index of the cell
  Int_t fTrackID;               //* index of the track
  Int_t fLink;                  //* link to the neighbour (if found)
  AliHLTTPCCATrackParam fParam; //* track parameters at the end point

private:
  void Dummy(); //* to make rulechecker happy by having something in .cxx file

};


#endif
