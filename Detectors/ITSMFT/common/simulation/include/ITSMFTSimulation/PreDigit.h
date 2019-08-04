// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreDigit.h
/// \brief Definition of the digit contribution from single hit
#ifndef ALICEO2_ITSMFT_PREDIGIT_H
#define ALICEO2_ITSMFT_PREDIGIT_H

#include "SimulationDataFormat/MCCompLabel.h"

/// \file PreDigit.h
/// \brief Contribution of single hit to digit

namespace o2
{

namespace itsmft
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
  int charge = 0.f;          ///< N electrons
  PreDigitLabelRef labelRef; ///< label and reference to the next one

  PreDigit(UInt_t rf = 0, UShort_t rw = 0, UShort_t cl = 0, int nele = 0, o2::MCCompLabel lbl = 0)
    : row(rw), col(cl), roFrame(rf), charge(nele), labelRef(lbl) {}

  ClassDefNV(PreDigit, 1);
};
} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITSMFT_PREDIGIT_H */
