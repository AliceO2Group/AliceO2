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

#ifndef O2_MCH_DEVIO_DIGITS_ROFRECORD_R0_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2::mch::io::impl
{

struct ROFRecordR0 {
  o2::InteractionRecord ir;
  o2::dataformats::RangeReference<int, int> ref;
};

} // namespace o2::mch::io::impl

#endif
