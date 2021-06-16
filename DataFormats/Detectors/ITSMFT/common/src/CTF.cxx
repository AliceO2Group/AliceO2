// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <stdexcept>
#include <cstring>
#include "Framework/Logger.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"

using namespace o2::itsmft;

///________________________________
void CompressedClusters::clear()
{
  firstChipROF.clear();
  bcIncROF.clear();
  orbitIncROF.clear();
  nclusROF.clear();
  row.clear();
  colInc.clear();
  chipInc.clear();
  chipMul.clear();
  pattID.clear();
  pattMap.clear();
}
