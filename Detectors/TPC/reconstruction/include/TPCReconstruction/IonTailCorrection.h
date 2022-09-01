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

/// \file IonTailCorrection.h
/// \brief Implementation of the ion tail correction from TPC digits
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
/// \author Marian Ivanov, m.ivanov@gsi.de

#ifndef TPC_IonTailCorrection_H_
#define TPC_IonTailCorrection_H_

#include <vector>

#include "Rtypes.h"
#include "DataFormatsTPC/Digit.h"
#include "CommonUtils/DebugStreamer.h"

namespace o2::tpc
{

class IonTailCorrection
{
 public:
  IonTailCorrection();
  void filterDigitsDirect(std::vector<Digit>& digits);

  /// Apply exponential filter to expanded array
  /// \param in input array, of `out` is null, directly operate on the vector
  /// \param out if specified, write filtered digits to this output instead of overwriting
  void exponentialFilter(std::vector<float>& in, std::vector<float>* out = nullptr);

  /// Sort digits of a single sector per pad in increasing time bin order
  static void sortDigitsOneSectorPerPad(std::vector<Digit>& digits);

  /// Sort digits of a single sector time bin by time bin
  /// within each time bin row by row, pad by pad
  static void sortDigitsOneSectorPerTimeBin(std::vector<Digit>& digits);

  void setITMultFactor(float multFactor) { mITMultFactor = multFactor; }
  float getITMultFactor() const { return mITMultFactor; }

  void setSign(float sign) { mSign = sign; }
  float getSign() const { return mSign; }

 private:
  float mITMultFactor = 1;            ///< fudge factor to tune IT correction
  float mKTime = 0.0515;              ///< kTime constant for ion tail filter
  float mSign = -1.f;                 ///< -1 do correction, +1 add tail
  o2::utils::DebugStreamer mStreamer; ///< debug streaming

  ClassDefNV(IonTailCorrection, 0);
};
} // namespace o2::tpc
#endif
