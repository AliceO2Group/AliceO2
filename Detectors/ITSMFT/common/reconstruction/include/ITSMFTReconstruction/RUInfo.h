// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_RUINFO_H
#define ALICEO2_RUINFO_H

// \file RUInfo.h
// \brief Transient structures for ITS and MFT HW -> SW mapping

#include <Rtypes.h>
#include <cstdint>

namespace o2
{
namespace itsmft
{

struct RUInfo {
  ///< provides details of the RU (e.g.stave)
  const std::uint16_t DUMMY16 = 0xffff;
  const std::uint8_t DUMMY8 = 0xff;
  uint16_t idSW = DUMMY16;         // software ID
  uint16_t idHW = DUMMY16;         // hardware ID
  uint16_t firstChipIDSW = 0xffff; // SW ID of the 1st chip of the module served by this RU
  uint8_t layer = DUMMY8;          // layer
  uint8_t ruType = DUMMY8;         // RU type (=subbarel)
  uint8_t nCables = DUMMY8;        // by how many cables it is served
};

struct ChipOnRUInfo {
  ///< provides details on chip location and HW labeling within the RU (stave)
  const std::uint8_t DUMMY8 = 0xff;
  std::uint8_t id = DUMMY8;             // chip ID within the stave (RU)
  std::uint8_t moduleSW = DUMMY8;       // sequential ID of the chip's module on stave
  std::uint8_t moduleHW = DUMMY8;       // HW ID of the chip's module on stave
  std::uint8_t chipOnModuleSW = DUMMY8; // sequential ID of the chip on the module
  std::uint8_t chipOnModuleHW = DUMMY8; // sequential ID of the chip on the module
  std::uint8_t cableSW = DUMMY8;        // cable SW ID
  std::uint8_t cableHW = DUMMY8;        // cable HW ID
  std::uint8_t chipOnCable = DUMMY8;    // chip within the cable (0 = master)

  void print() const;
};

struct ChipInfo {
  const std::uint16_t DUMMY16 = 0xffff;
  const std::uint8_t DUMMY8 = 0xff;
  const ChipOnRUInfo* chOnRU = nullptr; // pointer on chip detailed info within the stave

  std::uint16_t user = DUMMY16; // reserved for the user ?
  std::uint16_t id = DUMMY16;
  std::uint16_t ru = DUMMY16;     // RU sequential id
  std::uint16_t ruType = DUMMY16; // RU (or subBarrel) type

  void print() const;
};
} // namespace itsmft
} // namespace o2

#endif
