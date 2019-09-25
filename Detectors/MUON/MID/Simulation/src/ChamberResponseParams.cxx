// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/ChamberResponseParams.cxx
/// \brief  Implementation of the parameters for MID RPC response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 April 2018

/// This class implements the parameters for the parameterization of the RPC spatial resolution.
/// The parameters were tuned by Massimiliano Marchisone in his PhD thesis:
/// http://www.theses.fr/2013CLF22406
/// See ChamberResponse for further details

#include "MIDSimulation/ChamberResponseParams.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
double ChamberResponseParams::getParA(double hv) const
{
  /// Gets first parameter
  /// \par hv RPC HV in volts
  return mParA[1] * hv + mParA[0];
}

//______________________________________________________________________________
double ChamberResponseParams::getParC(double hv) const
{
  /// Get third parameter
  /// \par hv RPC HV in volts
  return mParC[1] * hv + mParC[0];
}

//______________________________________________________________________________
double ChamberResponseParams::getParB(int cathode, int deId) const
{
  /// Gets the second parameter
  /// \par cathode Cathode
  /// \par deId Detection element ID
  return mParB[72 * cathode + deId];
}

//______________________________________________________________________________
void ChamberResponseParams::setParA(double a0, double a1)
{
  /// Sets parameter A
  mParA[0] = a0;
  mParA[1] = a1;
}

//______________________________________________________________________________
void ChamberResponseParams::setParC(double c0, double c1)
{
  /// Sets parameter C
  mParC[0] = c0;
  mParC[1] = c1;
}

//______________________________________________________________________________
void ChamberResponseParams::setParB(int cathode, int deId, double val)
{
  /// Sets parameter B
  mParB[72 * cathode + deId] = val;
}

ChamberResponseParams createDefaultChamberResponseParams()
{
  /// Creates the default parameters
  ChamberResponseParams params;
  params.setParA(-52.70, 6.089 / 1000.);   // par1 in 1/V
  params.setParC(-0.5e-3, 8.3e-4 / 1000.); // par1 in 1/V

  // if (isStreamer) {
  //   mParB.fill(2.966);
  //   return;
  // }

  // BP
  // MT11R
  params.setParB(0, 0, 2.97);
  params.setParB(0, 1, 2.47);
  params.setParB(0, 2, 2.47);
  params.setParB(0, 3, 1.97);
  params.setParB(0, 4, 1.97);
  params.setParB(0, 5, 2.47);
  params.setParB(0, 6, 2.47);
  params.setParB(0, 7, 2.47);
  params.setParB(0, 8, 2.97);
  // MT12R
  params.setParB(0, 9, 2.97);
  params.setParB(0, 10, 1.97);
  params.setParB(0, 11, 1.97);
  params.setParB(0, 12, 1.97);
  params.setParB(0, 13, 2.22);
  params.setParB(0, 14, 2.22);
  params.setParB(0, 15, 1.97);
  params.setParB(0, 16, 2.47);
  params.setParB(0, 17, 2.97);
  // MT21R
  params.setParB(0, 18, 2.97);
  params.setParB(0, 19, 1.97);
  params.setParB(0, 20, 1.97);
  params.setParB(0, 21, 1.97);
  params.setParB(0, 22, 2.22);
  params.setParB(0, 23, 2.22);
  params.setParB(0, 24, 2.47);
  params.setParB(0, 25, 2.47);
  params.setParB(0, 26, 2.97);
  // MT22R
  params.setParB(0, 27, 2.97);
  params.setParB(0, 28, 1.97);
  params.setParB(0, 29, 1.97);
  params.setParB(0, 30, 1.97);
  params.setParB(0, 31, 1.97);
  params.setParB(0, 32, 1.97);
  params.setParB(0, 33, 2.97);
  params.setParB(0, 34, 2.97);
  params.setParB(0, 35, 2.97);
  // MT11L
  params.setParB(0, 36, 2.97);
  params.setParB(0, 37, 1.97);
  params.setParB(0, 38, 2.47);
  params.setParB(0, 39, 1.97);
  params.setParB(0, 40, 2.22);
  params.setParB(0, 41, 1.97);
  params.setParB(0, 42, 2.47);
  params.setParB(0, 43, 2.47);
  params.setParB(0, 44, 2.97);
  // MT12L
  params.setParB(0, 45, 2.97);
  params.setParB(0, 46, 1.97);
  params.setParB(0, 47, 2.47);
  params.setParB(0, 48, 1.97);
  params.setParB(0, 49, 1.97);
  params.setParB(0, 50, 1.97);
  params.setParB(0, 51, 2.47);
  params.setParB(0, 52, 1.97);
  params.setParB(0, 53, 2.97);
  // MT21L
  params.setParB(0, 54, 2.97);
  params.setParB(0, 55, 1.97);
  params.setParB(0, 56, 2.47);
  params.setParB(0, 57, 1.97);
  params.setParB(0, 58, 1.97);
  params.setParB(0, 59, 2.22);
  params.setParB(0, 60, 2.47);
  params.setParB(0, 61, 2.47);
  params.setParB(0, 62, 2.97);
  // MT22L
  params.setParB(0, 63, 2.97);
  params.setParB(0, 64, 2.22);
  params.setParB(0, 65, 2.47);
  params.setParB(0, 66, 1.72);
  params.setParB(0, 67, 1.97);
  params.setParB(0, 68, 1.97);
  params.setParB(0, 69, 1.97);
  params.setParB(0, 70, 2.47);
  params.setParB(0, 71, 2.97);

  // NBP
  // MT11R
  params.setParB(1, 0, 2.97);
  params.setParB(1, 1, 2.97);
  params.setParB(1, 2, 1.97);
  params.setParB(1, 3, 1.72);
  params.setParB(1, 4, 1.97);
  params.setParB(1, 5, 2.47);
  params.setParB(1, 6, 2.47);
  params.setParB(1, 7, 2.97);
  params.setParB(1, 8, 2.97);
  // MT12R
  params.setParB(1, 9, 2.97);
  params.setParB(1, 10, 2.97);
  params.setParB(1, 11, 1.97);
  params.setParB(1, 12, 1.97);
  params.setParB(1, 13, 2.47);
  params.setParB(1, 14, 1.97);
  params.setParB(1, 15, 2.22);
  params.setParB(1, 16, 2.97);
  params.setParB(1, 17, 2.97);
  // MT21R
  params.setParB(1, 18, 2.97);
  params.setParB(1, 19, 2.47);
  params.setParB(1, 20, 1.97);
  params.setParB(1, 21, 1.97);
  params.setParB(1, 22, 1.97);
  params.setParB(1, 23, 2.47);
  params.setParB(1, 24, 2.47);
  params.setParB(1, 25, 2.97);
  params.setParB(1, 26, 2.97);
  // MT22R
  params.setParB(1, 27, 2.97);
  params.setParB(1, 28, 1.97);
  params.setParB(1, 29, 1.97);
  params.setParB(1, 30, 1.97);
  params.setParB(1, 31, 1.72);
  params.setParB(1, 32, 1.97);
  params.setParB(1, 33, 2.97);
  params.setParB(1, 34, 2.97);
  params.setParB(1, 35, 2.97);
  // MT11L
  params.setParB(1, 36, 2.97);
  params.setParB(1, 37, 2.97);
  params.setParB(1, 38, 2.47);
  params.setParB(1, 39, 2.22);
  params.setParB(1, 40, 1.97);
  params.setParB(1, 41, 1.97);
  params.setParB(1, 42, 2.47);
  params.setParB(1, 43, 2.97);
  params.setParB(1, 44, 2.97);
  // MT12L
  params.setParB(1, 45, 2.97);
  params.setParB(1, 46, 2.97);
  params.setParB(1, 47, 2.97);
  params.setParB(1, 48, 1.97);
  params.setParB(1, 49, 1.97);
  params.setParB(1, 50, 1.97);
  params.setParB(1, 51, 2.97);
  params.setParB(1, 52, 2.47);
  params.setParB(1, 53, 2.97);
  // MT21L
  params.setParB(1, 54, 2.97);
  params.setParB(1, 55, 2.97);
  params.setParB(1, 56, 2.47);
  params.setParB(1, 57, 2.22);
  params.setParB(1, 58, 1.97);
  params.setParB(1, 59, 2.22);
  params.setParB(1, 60, 2.47);
  params.setParB(1, 61, 2.97);
  params.setParB(1, 62, 2.97);
  // MT22L
  params.setParB(1, 63, 2.47);
  params.setParB(1, 64, 2.97);
  params.setParB(1, 65, 2.47);
  params.setParB(1, 66, 1.97);
  params.setParB(1, 67, 2.22);
  params.setParB(1, 68, 1.72);
  params.setParB(1, 69, 1.97);
  params.setParB(1, 70, 2.97);
  params.setParB(1, 71, 2.97);

  return std::move(params);
}

} // namespace mid
} // namespace o2
