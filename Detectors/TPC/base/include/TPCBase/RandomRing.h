///
/// @file   RandomRing.h
/// @author Jens Wiechula, Jens.Wiechula@cern.ch
///

/// @brief  Ring with random number following a gaussian distribution
///
/// This class creates a set of random gaus numbers.
/// The idea is to create a set of random numbers that can be
/// reused in order to save computing time.
/// The numbers can then be used as a continuous stream in
/// a ring buffer
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@cern.ch

#ifndef AliceO2_TPC_RandomRing_H
#define AliceO2_TPC_RandomRing_H

#include <boost/format.hpp>

#include "Vc/Vc"
#include <vector>

#include "TF1.h"
#include "TRandom.h"

using float_v=Vc::float_v;

namespace AliceO2 {
namespace TPC {

class RandomRing
{
  public:
    enum class RandomType : char {
      Gaus,                  ///< Gaussian distribution
      Flat,                  ///< Flat distribution
      CustomTF1              ///< Custom TF1 function to be used
    };
    /// constructor
    /// @param [in] size size of the ring buffer
    RandomRing(const RandomType randomType = RandomType::Gaus, const size_t size = 500000);

    /// constructor accepting TF1
    /// @param [in] function TF1 function
    RandomRing(TF1 &function, const size_t size);

    /// next random value from the ring buffer
    /// This function return a value from the ring buffer
    /// and increases the buffer position
    /// @return next random value
    float getNextValue()
    {
      const float value = mRandomNumbers[mRingPosition];
      ++mRingPosition %= mRandomNumbers.size();
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
      mRingPosition %= mRandomNumbers.size();
      return value;
    }

    /// position in the ring buffer
    /// @return position in the ring buffer
    unsigned int getRingPosition() const { return  mRingPosition; }

  private:
    // =========================================================================
    // ===| members |===========================================================
    //

    RandomType mRandomType;                                   ///< Type of random numbers used
    std::vector<float, Vc::Allocator<float>> mRandomNumbers;  ///< Ring with random gaus numbers
    unsigned int mRingPosition;                               ///< presently accessed position in the ring

    // =========================================================================
    // ===| functions |=========================================================
    //

    /// disallow copy constructor
    RandomRing(const RandomRing &);

    /// disallow assignment operator
    void operator=(const RandomRing &) {}

}; // end class RandomRing

//______________________________________________________________________________
inline RandomRing::RandomRing(const RandomType randomType, const size_t size)
  : mRandomType(randomType),
    mRandomNumbers(size),
    mRingPosition(0)
{
  for (auto &v : mRandomNumbers) {
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
      case RandomType::CustomTF1: {
        v = 0;
        break;
      }
    }
  }
}

//______________________________________________________________________________
inline RandomRing::RandomRing(TF1 &function, const size_t size)
  : mRandomType(RandomType::CustomTF1),
    mRandomNumbers(size),
    mRingPosition(0)
{
  for (auto &v : mRandomNumbers) {
    v = function.GetRandom();
  }
}

} // namespace TPC
} // namespace AliceO2
#endif
