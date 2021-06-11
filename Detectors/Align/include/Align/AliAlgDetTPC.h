// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTPC.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TPC detector wrapper

#ifndef ALIALGDETTPC_H
#define ALIALGDETTPC_H

#include "Align/AliAlgDet.h"

namespace o2
{
namespace align
{

class AliAlgDetTPC : public AliAlgDet
{
 public:
  AliAlgDetTPC(const char* title = "");
  virtual ~AliAlgDetTPC();
  //
  virtual void defineVolumes();
  //
  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetTPC(const AliAlgDetTPC&);
  AliAlgDetTPC& operator=(const AliAlgDetTPC&);
  //
 protected:
  ClassDef(AliAlgDetTPC, 1);
};
} // namespace align
} // namespace o2
#endif
