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
/// @file   RandomRing.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Ring with random number
///
/// This class creates a set of random numbers.
/// The idea is to create a set of random numbers that can be
/// reused in order to save computing time.
/// The numbers can then be used as a continuous stream in
/// a ring buffer
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@cern.ch

#ifndef ALICEO2_TPC_RANDOMRING_H_
#define ALICEO2_TPC_RANDOMRING_H_

#include <boost/format.hpp>

#include "Vc/Vc"
#include <array>

#include "TF1.h"
#include "TRandom.h"

using float_v = Vc::float_v;

namespace o2
{
namespace tpc
{

template <size_t N = float_v::size() * 100000>
class RandomRing
{
 public:
  enum class RandomType : char {
    Gaus,     ///< Gaussian distribution
    Flat,     ///< Flat distribution
    CustomTF1 ///< Custom TF1 function to be used
  };

  /// disallow copy constructor
  RandomRing(const RandomRing&) = delete;

  /// disallow assignment operator
  void operator=(const RandomRing&) = delete;

  /// constructor
  /// @param [in] randomType type of the random generator
  RandomRing(const RandomType randomType = RandomType::Gaus);

  /// constructor accepting TF1
  /// @param [in] function TF1 function
  RandomRing(TF1& function);

  /// initialisation of the random ring
  /// @param [in] randomType type of the random generator
  void initialize(const RandomType randomType = RandomType::Gaus);

  /// initialisation of the random ring
  /// @param [in] randomType type of the random generator
  void initialize(TF1& function);

  /// next random value from the ring buffer
  /// This function return a value from the ring buffer
  /// and increases the buffer position
  /// @return next random value
  float getNextValue()
  {
    const float value = mRandomNumbers[mRingPosition];
    ++mRingPosition;
    if (mRingPosition >= mRandomNumbers.size()) {
      mRingPosition = 0;
    }
    return value;
  }

  /// next vector with random values
  /// This function retuns a Vc vector with random numbers to be
  /// used for vectorised programming and increases the buffer
  /// position by the size of the vector
  /// @return vector with random values
  float_v getNextValueVc()
  {
    const float_v value = float_v(&mRandomNumbers[mRingPosition]);
    mRingPosition += float_v::size();
    if (mRingPosition >= mRandomNumbers.size()) {
      mRingPosition = 0;
    }
    return value;
  }

  /// position in the ring buffer
  /// @return position in the ring buffer
  unsigned int getRingPosition() const { return mRingPosition; }

 private:
  // =========================================================================
  // ===| members |===========================================================
  //

  RandomType mRandomType;              ///< Type of random numbers used
  std::array<float, N> mRandomNumbers; ///< Ring with random gaus numbers
  size_t mRingPosition = 0;            ///< presently accessed position in the ring

}; // end class RandomRing

//______________________________________________________________________________
template <size_t N>
inline RandomRing<N>::RandomRing(const RandomType randomType)
  : mRandomType(randomType),
    mRandomNumbers()

{
  initialize(randomType);
}

//______________________________________________________________________________
template <size_t N>
inline RandomRing<N>::RandomRing(TF1& function)
  : mRandomType(RandomType::CustomTF1),
    mRandomNumbers()
{
  initialize(function);
}

//______________________________________________________________________________
template <size_t N>
inline void RandomRing<N>::initialize(const RandomType randomType)
{

  for (auto& v : mRandomNumbers) {
    // TODO: configurable mean and sigma
    switch (randomType) {
      case RandomType::Gaus: {
        v = gRandom->Gaus(0, 1);
        break;
      }
      case RandomType::Flat: {
        v = gRandom->Rndm();
        break;
      }
      default: {
        v = 0;
        break;
      }
    }
  }
}

//______________________________________________________________________________
template <size_t N>
inline void RandomRing<N>::initialize(TF1& function)
{
  for (auto& v : mRandomNumbers) {
    v = function.GetRandom();
  }
}

} // namespace tpc
} // namespace o2
#endif
