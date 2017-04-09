///
/// @file  Baseline.h
/// @author Jens Wiechula, Jens.Wiechula@cern.ch
///

/// @brief TPC Baseline (Noise and Pedestal) description
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@cern.ch
///
/// @todo Vc functionality to be implemented


#ifndef ALICEO2_TPC_BASELINE_H_
#define ALICEO2_TPC_BASELINE_H_

#include "Vc/Vc"

#include "TPCBase/PadSecPos.h"
#include "TPCBase/RandomRing.h"

using float_v = Vc::float_v;

namespace o2 {
namespace TPC {

class PadSecPos;

class Baseline {
  public:
    enum class BaselineType : char {
        Random,           ///< Simple randomly distributed numbers
        PseudoRealistic,  ///< More realistic noise and pedestal patterns from run1
        DataBase          ///< Baseline values as stored in the data base
    };

    /// Default constructor
    Baseline() : mBaselineType{BaselineType::Random}, mMeanNoise{0.8}, mMeanPedestal{70}, mPedestalSpread{10}, mRandomNoiseRing(RandomRing::RandomType::Gaus) {};

    /// setter for mean noise
    void setMeanNoise(float meanNoise)
    {
        mMeanNoise = meanNoise;
    }
    /// Noise for specific pad in a sector, for configured BaselineType
    /// @param [in] padSecPos sector, pad and row information
    /// @return noise value for specific pad
    float getNoise(const PadSecPos& padSecPos);

    /// Vector with noise values for specific single pad, for configured BaselineType
    /// This function retuns a vector with noise values, e.g. for following
    /// time bins of one specific pad
    /// @param [in] padSecPos sector, pad and row information
    /// @return noise value for specific pad
    float_v getNoiseVc(const PadSecPos& padSecPos);

  private:
    // =========================================================================
    // ===| members |===========================================================
    //

    BaselineType  mBaselineType;    ///< Type of the base line
    float         mMeanNoise;       ///< Average value for random noise generation
    float         mMeanPedestal;    ///< Average pedestal value
    float         mPedestalSpread;  ///< Spread of the pedestal values
    RandomRing    mRandomNoiseRing; ///< Ring with random number for noise


    // =========================================================================
    // ===| functions |=========================================================
    //

    /// disallow copy constructor
    Baseline(const Baseline &);

    /// disallow assignment operator
    void operator=(const Baseline &) {}

    /// Radom noise value with sigma mMeanNoise
    /// @return Radom noise value with sigma mMeanNoise
    float getRandomNoise();

    /// Radom noise value with sigma taken from a pseudo realistic map
    /// Pad-wise pseudo realistic noise map from run1
    /// @return Radom noise value with sigma from a pseudo realistic map
    float getPseudoRealisticNoise(const PadSecPos& padSecPos);

    /// Radom noise value with sigma taken from the calibration data base
    /// Pad-wise noise map from data base
    /// @return Radom noise value with sigma taken from the calibration data base
    float getNoiseFromDataBase(const PadSecPos& padSecPos);

    /// Vector of radom noise value with sigma mMeanNoise
    /// @return Vector of random noise value with sigma mMeanNoise
    float_v getRandomNoiseVc();

    /// Vector of radom noise values with sigma taken from a pseudo realistic map
    /// Pad-wise pseudo realistic noise map from run1
    /// @return Vector of radom noise values with sigma from a pseudo realistic map
    float_v getPseudoRealisticNoiseVc(const PadSecPos& padSecPos);

    /// Vector of radom noise values with sigma taken from the calibration data base
    /// Pad-wise noise map from data base
    /// @return Vector of radom noise values with sigma taken from the calibration data base
    float_v getNoiseFromDataBaseVc(const PadSecPos& padSecPos);

    float getRandomPedestal(const PadSecPos& padSecPos);
    float getPseudoRealisticPedestal(const PadSecPos& padSecPos);
    float getPedestalFromDataBase(const PadSecPos& padSecPos);

}; // class baseline

} // namespace TPC
} // namespace AliceO2
#endif
