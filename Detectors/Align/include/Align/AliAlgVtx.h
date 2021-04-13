// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgVtx.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Special fake "sensor" for event vertex.

/**
  * Special fake "sensor" for event vertex.
  * It is needed to allow adjustement of the global IP position
  * if the event event is used as a measured point.
  * Its degrees of freedom of LOCAL X,Y,Z, coinciding with
  * GLOBAL X,Y,Z.
  * Since the vertex added to the track as a mesured point must be
  * defined in the frame with X axis along the tracks, the T2L
  * matrix of this sensor need to be recalculated for each track!
  */

#ifndef ALIALGVTX_H
#define ALIALGVTX_H

#include "Align/AliAlgSens.h"
// class AliTrackPointArray; FIXME(milettri): needs AliTrackPointArray
//class AliESDtrack; FIXME(milettri): needs AliESDtrack
class AliAlgPoint;

namespace o2
{
namespace align
{

class AliAlgVtx : public AliAlgSens
{
 public:
  AliAlgVtx();
  //
  void applyCorrection(double* vtx) const;
  virtual bool isSensor() const { return true; }
  //
  void setAlpha(double alp)
  {
    mAlp = alp;
    prepareMatrixT2L();
  }
  virtual void prepareMatrixL2G(bool = 0) { mMatL2G.Clear(); }   // unit matrix
  virtual void prepareMatrixL2GIdeal() { mMatL2GIdeal.Clear(); } // unit matrix
  virtual void prepareMatrixT2L();
  //
  //  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t); FIXME(milettri): needs AliTrackPointArray, AliESDtrack
  //
 protected:
  AliAlgVtx(const AliAlgVtx&);
  AliAlgVtx& operator=(const AliAlgVtx&);
  //
 protected:
  //
  ClassDef(AliAlgVtx, 1);
};
} // namespace align
} // namespace o2
#endif
