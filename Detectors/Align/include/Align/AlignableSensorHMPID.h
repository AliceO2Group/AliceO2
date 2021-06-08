// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableSensorHMPID.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  HMPID sensor (chamber)

#ifndef ALIGNABLESENSORHMPID_H
#define ALIGNABLESENSORHMPID_H

#include "Align/AlignableSensor.h"

class TObjArray;
//class AliTrackPointArray;
//class AliESDtrack;
class AlignmentPoint;

namespace o2
{
namespace align
{

class AlignableSensorHMPID : public AlignableSensor
{
 public:
  AlignableSensorHMPID(const char* name = 0, int vid = 0, int iid = 0, int isec = 0);
  virtual ~AlignableSensorHMPID();
  //
  virtual AlignmentPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //  virtual void   setTrackingFrame();
  virtual void prepareMatrixT2L();
  //
 protected:
  //
  ClassDef(AlignableSensorHMPID, 1)
};
} // namespace align
} // namespace o2
#endif
