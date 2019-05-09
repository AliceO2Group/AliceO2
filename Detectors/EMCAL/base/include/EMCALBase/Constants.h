// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_CONSTANTS_H_
#define ALICEO2_EMCAL_CONSTANTS_H_

#include <Rtypes.h>

namespace o2
{
namespace emcal
{
enum {
  EMCAL_MODULES = 22,   ///< Number of modules, 12 for EMCal + 8 for DCAL
  EMCAL_ROWS = 24,      ///< Number of rows per module for EMCAL
  EMCAL_COLS = 48,      ///< Number of columns per module for EMCAL
  EMCAL_LEDREFS = 24,   ///< Number of LEDs (reference/monitors) per module for EMCAL; one per StripModule
  EMCAL_TEMPSENSORS = 8 ///< Number Temperature sensors per module for EMCAL
};

namespace constants
{
constexpr Double_t EMCAL_TIMESAMPLE = 100.;  ///< Width of a timebin in nanoseconds
constexpr Double_t EMCAL_ADCENERGY = 0.0167; ///< Energy of one ADC count in GeV/c^2
} // namespace constants

} // namespace emcal
} // namespace o2
#endif
