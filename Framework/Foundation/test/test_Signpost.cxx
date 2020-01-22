// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>
#define O2_SIGNPOST_DEFINE_CONTEXT
#include "Framework/Signpost.h"

int main(int argc, char** argv)
{
  // To be run inside some profiler (e.g. instruments) to make sure it actually
  // works.
  O2_SIGNPOST_INIT();
  O2_SIGNPOST(dpl, 1000, 0, 0, 0);
  O2_SIGNPOST_START(dpl, 1, 0, 0, 0);
  O2_SIGNPOST_END(dpl, 1, 0, 0, 0);
}
