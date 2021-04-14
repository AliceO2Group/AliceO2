// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetectorHMPID.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  HMPID detector wrapper

#ifndef ALIGNABLEDETECTORHMPID_H
#define ALIGNABLEDETECTORHMPID_H

#include "Align/AlignableDetector.h"

namespace o2
{
namespace align
{
class AlignableDetectorHMPID : public AlignableDetector
{
 public:
  AlignableDetectorHMPID(const char* title = "");
  virtual ~AlignableDetectorHMPID();
  //
  virtual void defineVolumes();
  //
  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AlignableDetectorHMPID(const AlignableDetectorHMPID&);
  AlignableDetectorHMPID& operator=(const AlignableDetectorHMPID&);
  //
 protected:
  ClassDef(AlignableDetectorHMPID, 1);
};
} // namespace align
} // namespace o2
#endif
