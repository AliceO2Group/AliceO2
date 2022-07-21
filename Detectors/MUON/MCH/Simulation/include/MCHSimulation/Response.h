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

#ifndef O2_MCH_SIMULATION_RESPONSE_H_
#define O2_MCH_SIMULATION_RESPONSE_H_

#include "DataFormatsMCH/Digit.h"
#include "MCHBase/MathiesonOriginal.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"

namespace o2
{
namespace mch
{

enum class Station {
  Type1,
  Type2345
};

class Response
{
 public:
  Response(Station station);
  ~Response() = default;

  float getChargeSpread() const { return mChargeSpread; }
  float getSigmaIntegration() const { return mSigmaIntegration; }
  bool isAboveThreshold(float charge) const { return charge > mChargeThreshold; }

  /** Converts energy deposition into a charge.
   *
   * @param edepos deposited energy from Geant (in GeV)
   * @returns an equivalent charge (roughyl in ADC units)
   *
   */
  float etocharge(float edepos) const;

  /** Compute the charge fraction in a rectangle area for a unit charge
   * occuring at position (0,0)
   *
   * @param xmin, xmax, ymin, ymax coordinates (in cm) defining the area
   */
  float chargePadfraction(float xmin, float xmax, float ymin, float ymax) const
  {
    return mMathieson.integrate(xmin, ymin, xmax, ymax);
  }

  /// return wire coordinate closest to x
  float getAnod(float x) const;

  /// return a randomized charge correlation between cathodes
  float chargeCorr() const;

  /// compute the number of samples corresponding to the ADC value
  uint32_t nSamples(uint32_t adc) const;

 private:
  MathiesonOriginal mMathieson{}; ///< Mathieson function
  float mPitch = 0.f;             ///< anode-cathode pitch (cm)
  float mChargeSlope = 0.f;       ///< charge slope used in E to charge conversion
  float mChargeSpread = 0.f;      ///< width of the charge distribution (cm)
  float mSigmaIntegration = 0.f;  ///< number of sigmas used for charge distribution
  float mChargeCorr = 0.f;        ///< amplitude of charge correlation between cathodes
  float mChargeThreshold = 0.f;   ///< minimum fraction of charge considered
};
} // namespace mch
} // namespace o2
#endif
