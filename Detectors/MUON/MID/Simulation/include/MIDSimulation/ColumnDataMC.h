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

/// \file   MIDSimulation/ColumnDataMC.h
/// \brief  Strip pattern (aka digits) for simulations
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2019

#ifndef O2_MID_COLUMNDATAMC_H
#define O2_MID_COLUMNDATAMC_H

#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{
/// Column data structure for MID simulations
class ColumnDataMC : public ColumnData
{
  ClassDefNV(ColumnDataMC, 1);
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATAMC_H */
