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

/// \file CalibdEdxDataContainer.h
/// \brief
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDXDATACONTAINER_H_
#define ALICEO2_TPC_CALIBDEDXDATACONTAINER_H_

#include <array>
#include <cstddef>

#include "DataFormatsTPC/Defs.h"
#include "Framework/Logger.h"

namespace o2::tpc
{

/// Define the commom properties of dEdx containers.
template <typename Entry>
class CalibdEdxDataContainer
{
 public:
  static const size_t stacksCount = SECTORSPERSIDE * SIDES * GEMSTACKSPERSECTOR;
  static const size_t size = stacksCount * DEDXCHARGETYPES; // both Tot and Max charge data in a single array
  using Container = std::array<Entry, size>;

  /// Fill the undelying arrays with the passed value
  void init(const Entry&);

  /// \brief Find the index of a stack.
  /// \param sector from 0 to 17
  static size_t stackIndex(size_t sector, Side, GEMstack, dEdxCharge);

  /// \brief data from a specific stack.
  Entry& at(size_t sector, Side side, GEMstack type, dEdxCharge charge)
  {
    const size_t index = stackIndex(sector, side, type, charge);
    return mEntries[index];
  }

  const Container& container() const { return mEntries; };
  // We need a non const version to manipulate the data
  Container& container() { return mEntries; };

 private:
  Container mEntries;
};

template <typename Entry>
void CalibdEdxDataContainer<Entry>::init(const Entry& entry)
{
  mEntries.fill(entry);
}

template <typename Entry>
size_t CalibdEdxDataContainer<Entry>::stackIndex(size_t sector, const Side side, const GEMstack type, const dEdxCharge charge)
{
  // Limit sector value
  if (sector >= SECTORSPERSIDE) {
    LOGP(error, "Sector number should be less than {}", SECTORSPERSIDE);
    sector %= SECTORSPERSIDE;
  }

  // Account for the readout type: IROC, OROC1, ...
  sector += type * SECTORSPERSIDE;

  // Account for side
  constexpr size_t sideOffset = SECTORSPERSIDE * GEMSTACKSPERSECTOR;
  sector += side * sideOffset;

  // Account for charge type
  sector += charge * stacksCount;

  return sector;
}

} // namespace o2::tpc

#endif
