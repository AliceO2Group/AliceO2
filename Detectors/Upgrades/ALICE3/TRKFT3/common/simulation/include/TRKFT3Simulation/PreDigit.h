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

// PreDigit is a contribution to the digit from a single hit which happen
// to be registered 1st in given chip row/column/roframe
// Eventual extra contributions will be registered as PreDigitExtra, pointed by
// the "next" index

// PreDigitExtra registers additional contributions to the same row/column/roframe
struct PreDigitLabelRef {
  o2::MCCompLabel label; ///< hit label
  int next = -1;         ///< eventual next contribution to the same pixel
  PreDigitLabelRef(o2::MCCompLabel lbl = 0, int nxt = -1) : label(lbl), next(nxt) {}

  ClassDefNV(PreDigitLabelRef, 1);
};

struct PreDigit {
  UShort_t row = 0;          ///< Pixel index in X
  UShort_t col = 0;          ///< Pixel index in Z
  UInt_t roFrame = 0;        ///< Readout frame
  int charge = 0;            ///< N electrons
  float toa = 0.f;           ///< Time of arrival in ns (earliest contribution)
  PreDigitLabelRef labelRef; ///< label and reference to the next one

  PreDigit(UInt_t rf = 0, UShort_t rw = 0, UShort_t cl = 0, int nele = 0, o2::MCCompLabel lbl = 0, float t = 0.f)
    : row(rw), col(cl), roFrame(rf), charge(nele), toa(t), labelRef(lbl) {}

  ClassDefNV(PreDigit, 1);
};

} // namespace o2::trkft3

#endif /* ALICEO2_TRKFT3_PREDIGIT_H */
