// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigContext.h"

namespace o2::framework
{

bool ConfigContext::helpOnCommandLine() const
{
  bool helpasked = false;
  for (int argi = 0; argi < mArgc; ++argi) {
    if (strcmp(mArgv[argi], "--help") == 0 || (strcmp(mArgv[argi], "-h") == 0)) {
      helpasked = true;
      break;
    }
  }
  return helpasked;
}

} // namespace o2::framework
