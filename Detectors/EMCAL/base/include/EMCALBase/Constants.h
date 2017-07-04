// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_CONSTANTS_H_
#define ALICEO2_EMCAL_CONSTANTS_H_

namespace o2
{
namespace EMCAL
{
enum {
  EMCAL_MODULES = 22,   ///< Number of modules, 12 for EMCal + 8 for DCAL
  EMCAL_ROWS = 24,      ///< Number of rows per module for EMCAL
  EMCAL_COLS = 48,      ///< Number of columns per module for EMCAL
  EMCAL_LEDREFS = 24,   ///< Number of LEDs (reference/monitors) per module for EMCAL; one per StripModule
  EMCAL_TEMPSENSORS = 8 ///< Number Temperature sensors per module for EMCAL
};
}
}
#endif
