#ifndef ALIALGVTX_H
#define ALIALGVTX_H

/*--------------------------------------------------------
  Special fake "sensor" for event vertex.
  It is needed to allow adjustement of the global IP position
  if the event event is used as a measured point.
  Its degrees of freedom of LOCAL X,Y,Z, coinciding with
  GLOBAL X,Y,Z. 
  Since the vertex added to the track as a mesured point must be
  defined in the frame with X axis along the tracks, the T2L
  matrix of this sensor need to be recalculated for each track!
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


#include "AliAlgSens.h"
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;


class AliAlgVtx : public AliAlgSens
{
 public:
  AliAlgVtx();
  //
  void           ApplyCorrection(double *vtx) const;
  virtual Bool_t IsSensor()                   const {return kTRUE;}
  //
  void SetAlpha(double alp)              {fAlp=alp; PrepareMatrixT2L();}
  virtual void   PrepareMatrixL2G(Bool_t=0)      {fMatL2G.Clear();} // unit matrix
  virtual void   PrepareMatrixL2GIdeal() {fMatL2GIdeal.Clear();} // unit matrix
  virtual void   PrepareMatrixT2L();
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //
 protected:
  AliAlgVtx(const AliAlgVtx&);
  AliAlgVtx& operator=(const AliAlgVtx&);
  //
 protected:
  //
  ClassDef(AliAlgVtx,1);
};


#endif
