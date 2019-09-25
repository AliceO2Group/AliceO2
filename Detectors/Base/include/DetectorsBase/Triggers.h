// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Triggers.h
/// \brief Definition of the Alpide triggers
/// defined in https://twiki.cern.ch/twiki/pub/ALICE/NoteForFEDevelopers/CTS_CRU_FE_interface.pdf

#ifndef ALICEO2_ITSMFT_TRIGGERS_H
#define ALICEO2_ITSMFT_TRIGGERS_H

#include <cstdint>
#include <Rtypes.h>

namespace o2
{
namespace trigger
{

constexpr uint32_t ORBIT = 0x1 << 0; // Orbit
constexpr uint32_t HB = 0x1 << 1;    // Heart bit flag
constexpr uint32_t HBr = 0x1 << 2;   // Heart bit reject flag
constexpr uint32_t HC = 0x1 << 3;    // Heart Check
constexpr uint32_t PhT = 0x1 << 4;   // Physics Trigger
constexpr uint32_t PP = 0x1 << 5;    // Pre Pulse for calibration
constexpr uint32_t Cal = 0x1 << 6;   // Calibration trigger
constexpr uint32_t SOT = 0x1 << 7;   // Start of Triggered Data
constexpr uint32_t EOT = 0x1 << 8;   // EOT End of Triggered Data
constexpr uint32_t SOC = 0x1 << 9;   // Start of Continuous Data
constexpr uint32_t EOC = 0x1 << 10;  // End of Continuous Data
constexpr uint32_t TF = 0x1 << 11;   // Time Frame delimiter
// spare
constexpr uint32_t TPC = 0x1 << 29;    // TPC syncTPC synchronisation
constexpr uint32_t TPCrst = 0x1 << 30; // TPC reset
constexpr uint32_t TOF = 0x1 << 31;    // TOF special trigger
} // namespace trigger
} // namespace o2

#endif
