// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EventVertex.h
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

#ifndef EVENTVERTEX_H
#define EVENTVERTEX_H

#include "Align/AlignableSensor.h"

class AlignmentPoint;

namespace o2
{
namespace align
{

class Controller;

class EventVertex : public AlignableSensor
{
 public:
  EventVertex() = default;
  EventVertex(Controller* ctr);
  //
  void applyCorrection(double* vtx) const;
  bool isSensor() const final { return true; }
  //
  void setAlpha(double alp)
  {
    mAlp = alp;
    prepareMatrixT2L();
  }
  void prepareMatrixL2G(bool = 0) final { mMatL2G.Clear(); }   // unit matrix
  void prepareMatrixL2GIdeal() final { mMatL2GIdeal.Clear(); } // unit matrix
  void prepareMatrixT2L() final;
  //
 protected:
  EventVertex(const EventVertex&);
  EventVertex& operator=(const EventVertex&);
  //
 protected:
  //
  ClassDef(EventVertex, 1);
};
} // namespace align
} // namespace o2
#endif
