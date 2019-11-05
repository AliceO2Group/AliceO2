// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitMCMetaData.h
/// \brief Definition of the Meta Data object of the Monte Carlo Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_CommonMode_H_
#define ALICEO2_TPC_CommonMode_H_

#include <Rtypes.h>
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

/// \class CommonMode
/// This is the definition of a very simple object used to write out
/// the common mode value in each GEM stack

class CommonMode
{
 public:
  /// Default constructor
  CommonMode() = default;

  /// Constructor, initializing values for common mode value in GEM stack in a given time bin
  /// \param commonMode Common mode signal on that GEM stack
  /// \param timeBin Time bin
  /// \param gemStack GEm stack
  CommonMode(float commonMode, TimeBin timeBin, unsigned char gemStack);

  /// Destructor
  ~CommonMode() = default;

  /// Get the common mode value
  /// \return Common mode signal
  float getCommonMode() const { return mCommonMode; }

  /// Get the time bin
  /// \return Time bin
  TimeBin getTimeBin() const { return mTimebin; }

  /// Get the GEM stack
  /// \return GEM stack
  unsigned char getGEMstack() const { return mGEMstack; }

 private:
  float mCommonMode = 0.f;      ///< Common mode value
  TimeBin mTimebin = -1;        ///< Time bin
  unsigned char mGEMstack = -1; ///< GEM stack

  ClassDefNV(CommonMode, 1);
};

inline CommonMode::CommonMode(float commonMode, TimeBin timeBin, unsigned char gemStack)
  : mCommonMode(commonMode),
    mTimebin(timeBin),
    mGEMstack(gemStack)
{
}

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_CommonMode_H_
