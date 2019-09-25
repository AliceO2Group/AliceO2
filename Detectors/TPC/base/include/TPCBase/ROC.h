// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   ROC.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  ReadOut Chamber (ROC) type
///
/// This class represents a Readout chamber
/// It provides functionality to get the side, chamber type, etc.
/// Inner ReadOut Chambers (IROC) are counted
/// from 0-17 (A-Side)
/// and 18-35 (C-Side)
/// Outer ReadOut Chambers (OROC) are counted
/// from 36-53 (A-Side)
/// and  54-71 (C-Side)
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_ROC_H
#define AliceO2_TPC_ROC_H

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"

namespace o2
{
namespace tpc
{
//   enum RocType {IROC=0, OROC=1};

class ROC
{
 public:
  enum {
    MaxROC = 72
  };

  /// default constructor
  ROC() = default;

  /// constructor
  /// @param [in] roc readout chamber number
  ROC(unsigned char roc) : mROC(roc % MaxROC) { ; }

  /// constructor from sector and ROC type
  /// @param [in] sec sector
  /// @param [in] type ROC type
  ROC(const Sector& sec, const RocType type) : mROC(sec.getSector() + (type == RocType::OROC) * 2 * SECTORSPERSIDE) {}

  /// comparison operator
  bool operator==(const ROC& other) { return mROC == other.mROC; }

  /// unequal operator
  bool operator!=(const ROC& other) { return mROC != other.mROC; }

  /// smaller operator
  bool operator<(const ROC& other) { return mROC < other.mROC; }

  /// increment operator
  /// This operator can be used to iterate over all ROCs e.g.
  /// ROC r;
  /// while (++r) { std::cout << "ROC: " << r.getRoc() << std::endl; }
  bool operator++()
  {
    mLoop = ++mROC >= MaxROC;
    mROC %= MaxROC;
    return mLoop;
  }

  /// int return operator to use similar as integer
  /// \return roc number
  operator int() const { return int(mROC); }

  /// numerical ROC value
  /// \return numerical ROC value
  unsigned char getRoc() const { return mROC; }

  /// side of the ROC
  /// \return side of the sector
  Side side() const { return (mROC / SECTORSPERSIDE) % SIDES ? Side::C : Side::A; }

  /// ROC type
  /// \return ROC type
  RocType rocType() const { return mROC < MaxROC / SIDES ? RocType::IROC : RocType::OROC; }

  /// get sector
  Sector getSector() const { return Sector(mROC); }

  /// if increment operator went above MaxROC
  bool looped() const { return mLoop; }

 private:
  unsigned char mROC{0}; ///< ROC representation 0-MaxROC-1
  bool mLoop{false};     ///< if increment operator resulted in looping
};
} // namespace tpc
} // namespace o2

#endif
