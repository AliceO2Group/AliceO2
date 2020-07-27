// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <stdexcept>
#include <cstring>
#include "Framework/Logger.h"
#include "DataFormatsFT0/CTF.h"

using namespace o2::ft0;

///________________________________
void CompressedDigits::clear()
{
  trigger.clear();
  bcInc.clear();
  orbitInc.clear();
  nChan.clear();
  //  eventFlags.clear();

  idChan.clear();
  qtcChain.clear();
  cfdTime.clear();
  qtcAmpl.clear();

  header.nTriggers = 0;
  header.firstOrbit = 0;
  header.firstBC = 0;
}
