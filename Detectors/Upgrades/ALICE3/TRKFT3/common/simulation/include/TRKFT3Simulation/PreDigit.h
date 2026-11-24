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

/// \file PreDigit.h
/// \brief Definition of the digit contribution from single hit for TRK/FT3 (based on o2::itsmft::PreDigit, with time of arrival)
#ifndef ALICEO2_TRKFT3_PREDIGIT_H
#define ALICEO2_TRKFT3_PREDIGIT_H

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::trkft3
{

// PreDigit: pixel row/column/roframe with at least one contribution (hit) in that ROF.
// Each contribution (label, arrival time, charge) is a PreDigitContribution; the first one
// is stored in the PreDigit, the others in a chain pointed by the "next" index.
// Contributions are kept separately (not summed) because, with edge detection, the digit is
// built at the flush by processing them in time order together with the pixel state.

struct PreDigitContribution {
  o2::MCCompLabel label; ///< hit label
  float toa = 0.f;       ///< arrival time w.r.t. ROF start (ns)
  int charge = 0;        ///< N electrons of this contribution
  int next = -1;         ///< eventual next contribution to the same pixel
  PreDigitContribution(o2::MCCompLabel lbl = 0, float t = 0.f, int q = 0, int nxt = -1) : label(lbl), toa(t), charge(q), next(nxt) {}

  ClassDefNV(PreDigitContribution, 1);
};

struct PreDigit {
  UShort_t row = 0;             ///< Pixel index in X
  UShort_t col = 0;             ///< Pixel index in Z
  UInt_t roFrame = 0;           ///< Readout frame
  PreDigitContribution contrib; ///< first contribution and reference to the next one

  PreDigit(UInt_t rf = 0, UShort_t rw = 0, UShort_t cl = 0, int nele = 0, o2::MCCompLabel lbl = 0, float t = 0.f)
    : row(rw), col(cl), roFrame(rf), contrib(lbl, t, nele) {}

  ClassDefNV(PreDigit, 2);
};

} // namespace o2::trkft3

#endif /* ALICEO2_TRKFT3_PREDIGIT_H */
