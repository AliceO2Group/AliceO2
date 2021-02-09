// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSensHMPID.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  HMPID sensor (chamber)

#ifndef ALIALGSENSHMPID_H
#define ALIALGSENSHMPID_H

#include "AliAlgSens.h"

class TObjArray;
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;

namespace o2
{
namespace align
{

class AliAlgSensHMPID : public AliAlgSens
{
 public:
  AliAlgSensHMPID(const char* name = 0, Int_t vid = 0, Int_t iid = 0, Int_t isec = 0);
  virtual ~AliAlgSensHMPID();
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //  virtual void   SetTrackingFrame();
  virtual void PrepareMatrixT2L();
  //
 protected:
  //
  ClassDef(AliAlgSensHMPID, 1)
};
} // namespace align
} // namespace o2
#endif
