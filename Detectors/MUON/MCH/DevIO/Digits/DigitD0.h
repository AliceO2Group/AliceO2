// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include <fstream>
#include <vector>

namespace o2::mch::io::impl
{
struct DigitD0 {
  int32_t tfTime{0};      /// time since the beginning of the time frame, in bunch crossing units
  uint16_t nofSamples{0}; /// number of samples in the signal + saturated bit
  int detID{0};           /// ID of the Detection Element to which the digit corresponds to
  int padID{0};           /// PadIndex to which the digit corresponds to
  uint32_t adc{0};        /// Amplitude of signal

  void setNofSamples(uint16_t n) { nofSamples = (nofSamples & 0x8000) + (n & 0x7FFF); }
  uint16_t getNofSamples() const { return (nofSamples & 0x7FFF); }

  void setSaturated(bool sat) { nofSamples = sat ? nofSamples | 0x8000 : nofSamples & 0x7FFF; }
  bool isSaturated() const { return ((nofSamples & 0x8000) > 0); }
};

} // namespace o2::mch::io::impl
