// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_VMCREPLAY_H
#define O2_VMCREPLAY_H

#include "TGeant3TGeo.h"

class VMCReplay : public TGeant3TGeo
{
 public:
  using TGeant3TGeo::TGeant3TGeo;

  ClassDef(VMCReplay, 1); // needed as long we inherit from TObject
};

#endif //O2_VMCREPLAY_H
