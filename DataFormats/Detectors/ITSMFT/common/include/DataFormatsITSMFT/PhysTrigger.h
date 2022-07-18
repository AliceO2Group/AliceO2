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

/// \file PhysTrigger.h
/// \brief Definition Physics trigger record extracted from the ITS/MFT stream

#ifndef ALICEO2_ITSMFT_PHYSTRIGGER_H
#define ALICEO2_ITSMFT_PHYSTRIGGER_H

#include "CommonDataFormat/InteractionRecord.h"

namespace o2::itsmft
{
// root friendly version of the trigger (root does not support anonymous structs)
struct PhysTrigger {
  o2::InteractionRecord ir{};
  uint64_t data = 0;

  ClassDefNV(PhysTrigger, 1);
};

} // namespace o2::itsmft

#endif
