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

/// \file FT0DCSConfigReader.h
/// \brief DCS configuration reader for FT0
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FT0_DCSCONFIGREADER_H
#define O2_FT0_DCSCONFIGREADER_H

#include "FITDCSMonitoring/FITDCSConfigReader.h"
#include "Rtypes.h"

namespace o2
{
namespace ft0
{

/// DCS configuration reader for FT0
///
/// At the moment this class doesn't differ from the base class o2::fit::FITDCSConfigReader,
/// which makes it obsolete. It exists only as an example for how to create detector specific
/// DCS configuration readers later.
class FT0DCSConfigReader : public o2::fit::FITDCSConfigReader
{
  // For FT0 specific processing of DCS configurations, override base class methods here.

  ClassDefNV(FT0DCSConfigReader, 0);
};

} // namespace ft0
} // namespace o2

#endif // O2_FT0_DCSCONFIGREADER_H