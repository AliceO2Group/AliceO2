// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetHMPID.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  HMPID detector wrapper

#ifndef ALIALGDETHMPID_H
#define ALIALGDETHMPID_H

#include "AliAlgDet.h"

namespace o2
{
namespace align
{
class AliAlgDetHMPID : public AliAlgDet
{
 public:
  AliAlgDetHMPID(const char* title = "");
  virtual ~AliAlgDetHMPID();
  //
  virtual void DefineVolumes();
  //
  Bool_t AcceptTrack(const AliESDtrack* trc, Int_t trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetHMPID(const AliAlgDetHMPID&);
  AliAlgDetHMPID& operator=(const AliAlgDetHMPID&);
  //
 protected:
  ClassDef(AliAlgDetHMPID, 1);
};
} // namespace align
} // namespace o2
#endif
