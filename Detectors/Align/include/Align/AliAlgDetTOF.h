// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Wrapper for TOF detector

#ifndef ALIALGDETTOF_H
#define ALIALGDETTOF_H

#include "AliAlgDet.h"

namespace o2
{
namespace align
{

class AliAlgDetTOF : public AliAlgDet
{
 public:
  AliAlgDetTOF(const char* title = "");
  virtual ~AliAlgDetTOF();
  //
  virtual void DefineVolumes();
  //
  Bool_t AcceptTrack(const AliESDtrack* trc, Int_t trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetTOF(const AliAlgDetTOF&);
  AliAlgDetTOF& operator=(const AliAlgDetTOF&);
  //
 protected:
  ClassDef(AliAlgDetTOF, 1);
};
} // namespace align
} // namespace o2
#endif
