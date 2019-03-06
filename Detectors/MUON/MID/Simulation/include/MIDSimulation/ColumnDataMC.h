// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{
/// Column data structure for MID simulations
class ColumnDataMC : public ColumnData, public o2::dataformats::TimeStamp<int>
{
  ClassDefNV(ColumnDataMC, 1);
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATAMC_H */
