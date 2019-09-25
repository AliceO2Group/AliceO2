// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   PadROCPos.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_PADROCPOS_H_
#define ALICEO2_TPC_PADROCPOS_H_

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/ROC.h"
#include "TPCBase/PadPos.h"

namespace o2
{
namespace tpc
{

/// \brief  Pad and row inside a ROC
///
/// This class encapsulates the pad and row inside a ROC
/// \see TPCBase/PadPos.h
/// \see TPCBase/ROC.h
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class PadROCPos
{
 public:
  /// default constructor
  PadROCPos() = default;

  /// constructor from roc, row and pad
  /// \param roc ROC number
  /// \param rowInROC row in the readout chamber
  /// \param padInRow pad in row
  PadROCPos(const int roc, const int rowInROC, const int padInRow) : mROC(roc), mPadPos(PadPos(rowInROC, padInRow)) {}

  /// constructor from ROC and PadPos types
  /// \param roc ROC type
  /// \param padPosition row and pad
  PadROCPos(const ROC& roc, const PadPos& padPosition) : mROC(roc), mPadPos(padPosition) {}

  /// get ROC
  /// \return ROC
  const ROC& getROC() const { return mROC; }

  /// get the sector
  /// \return sector
  const Sector getSector() const { return mROC.getSector(); }

  /// get ROC
  /// \return ROC
  ROC& getROC() { return mROC; }

  /// get ROC type
  /// \return ROC type
  RocType getROCType() const { return mROC.rocType(); }

  /// get pad and row position
  /// \return pad and row position
  const PadPos& getPadPos() const { return mPadPos; }

  /// get the pad row
  /// \return pad row
  int getRow() const { return mPadPos.getRow(); }

  /// get the pad number
  /// \return pad number
  int getPad() const { return mPadPos.getPad(); }

  PadPos& getPadPos() { return mPadPos; }

  /// check if is valid
  /// @return pad valid
  bool isValid() const { return mPadPos.isValid(); }

  /// equal operator
  bool operator==(const PadROCPos& other) const { return (mROC == other.mROC) && (mPadPos == other.mPadPos); }

  /// smaller operator
  bool operator<(const PadROCPos& other) const
  {
    if (mROC < other.mROC)
      return true;
    if (mROC == other.mROC && mPadPos < other.mPadPos)
      return true;
    return false;
  }

 private:
  ROC mROC{};
  PadPos mPadPos{};
};
} // namespace tpc
} // namespace o2
#endif
