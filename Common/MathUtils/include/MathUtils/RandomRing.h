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
/// This class creates a set of random (or any sort of pregenerated) numbers
/// in a ring buffer.

/// The idea is to create a set of random numbers that can be
/// reused in order to save computing time.
/// The numbers can then be used as a continuous stream in
/// a ring buffer
///
/// @author Jens Wiechula, Jens.Wiechula@cern.ch

#ifndef ALICEO2_MATHUTILS_RANDOMRING_H_
#define ALICEO2_MATHUTILS_RANDOMRING_H_

#include <array>

#include "TF1.h"
#include "TRandom.h"
#include <functional>


namespace o2
{
namespace math_utils
{

template <size_t N = 4 * 100000>
class RandomRing
{
 public:
  enum class RandomType : char {
    Gaus,         ///< Gaussian distribution
    Flat,         ///< Flat distribution
    CustomTF1,    ///< Custom TF1 function to be used
    CustomLambda, ///< Initialized through external lambda
  };

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

  /// initialisation of the random ring
  /// @param [in] randomType type of the random generator
  void initialize(std::function<float()> function);

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
  template <typename VcType>
  VcType getNextValueVc()
  {
    // This function is templated so that we don't need to include the <Vc/Vc> header
    // within this header file (to reduce memory problems during compilation).
    // The hope is that the calling user calls this with a
    // correct Vc type (Vc::float_v) in a source file.
    const VcType value = VcType(&mRandomNumbers[mRingPosition]);
    mRingPosition += VcType::size();
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
  mRandomType = RandomType::CustomTF1;
  for (auto& v : mRandomNumbers) {
    v = function.GetRandom();
  }
}

//______________________________________________________________________________
template <size_t N>
inline void RandomRing<N>::initialize(std::function<float()> function)
{
  mRandomType = RandomType::CustomLambda;
  for (auto& v : mRandomNumbers) {
    v = function();
  }
}

} // namespace math_utils
} // namespace o2
#endif
