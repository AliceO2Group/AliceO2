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
  params.setParA(-20.0, 6.089 / 1000.);   // par1 in 1/V (par0 updated from Run 3 fit)
  params.setParC(-4.2e-3, 4.6e-4 / 1000.); // par1 in 1/V (par0 & par1 updated from Run 3 fit)

  // if (isStreamer) {
  //   mParB.fill(2.966);
  //   return;
  // }

  // Updated b-params from Run 3 fit
  params.setParB(1, 0, 1.91);
  params.setParB(0, 1, 2.37);
  params.setParB(1, 1, 1.94);
  params.setParB(0, 2, 2.21);
  params.setParB(1, 2, 1.87);
  params.setParB(0, 3, 2.39);
  params.setParB(1, 3, 1.81);
  params.setParB(0, 4, 2.49);
  params.setParB(1, 4, 1.76);
  params.setParB(0, 5, 3.06);
  params.setParB(1, 5, 2.01);
  params.setParB(0, 6, 2.24);
  params.setParB(1, 6, 2.01);
  params.setParB(0, 7, 2.39);
  params.setParB(1, 7, 2.03);
  params.setParB(0, 8, 2.00);
  params.setParB(1, 8, 1.88);
  params.setParB(0, 9, 2.15);
  params.setParB(1, 9, 2.05);
  params.setParB(0, 10, 2.42);
  params.setParB(1, 10, 1.88);
  params.setParB(0, 11, 2.17);
  params.setParB(1, 11, 1.85);
  params.setParB(0, 12, 2.47);
  params.setParB(1, 12, 1.77);
  params.setParB(0, 13, 2.22);
  params.setParB(1, 13, 1.81);
  params.setParB(0, 14, 2.89);
  params.setParB(1, 14, 2.08);
  params.setParB(0, 15, 2.24);
  params.setParB(1, 15, 1.98);
  params.setParB(0, 16, 2.29);
  params.setParB(1, 16, 1.74);
  params.setParB(0, 17, 2.04);
  params.setParB(1, 17, 2.13);
  params.setParB(0, 18, 2.07);
  params.setParB(1, 18, 2.01);
  params.setParB(0, 19, 2.28);
  params.setParB(1, 19, 1.85);
  params.setParB(0, 20, 2.22);
  params.setParB(1, 20, 1.88);
  params.setParB(0, 21, 2.83);
  params.setParB(1, 21, 2.05);
  params.setParB(0, 22, 2.27);
  params.setParB(1, 22, 2.05);
  params.setParB(0, 23, 2.64);
  params.setParB(1, 23, 2.01);
  params.setParB(0, 24, 2.20);
  params.setParB(1, 24, 2.01);
  params.setParB(0, 25, 2.38);
  params.setParB(1, 25, 2.01);
  params.setParB(0, 26, 2.12);
  params.setParB(1, 26, 2.09);
  params.setParB(0, 27, 2.12);
  params.setParB(1, 27, 1.98);
  params.setParB(0, 28, 2.41);
  params.setParB(1, 28, 1.99);
  params.setParB(0, 29, 2.35);
  params.setParB(1, 29, 2.39);
  params.setParB(0, 30, 2.70);
  params.setParB(1, 30, 1.96);
  params.setParB(0, 31, 2.23);
  params.setParB(1, 31, 1.99);
  params.setParB(0, 32, 2.38);
  params.setParB(1, 32, 2.06);
  params.setParB(0, 33, 2.37);
  params.setParB(1, 33, 2.15);
  params.setParB(0, 34, 2.38);
  params.setParB(1, 34, 2.01);
  params.setParB(0, 35, 2.08);
  params.setParB(1, 35, 1.89);
  params.setParB(0, 36, 2.05);
  params.setParB(1, 36, 1.88);
  params.setParB(0, 37, 2.31);
  params.setParB(1, 37, 1.89);
  params.setParB(0, 38, 2.26);
  params.setParB(1, 38, 1.84);
  params.setParB(0, 39, 2.57);
  params.setParB(1, 39, 2.17);
  params.setParB(0, 40, 2.52);
  params.setParB(1, 40, 1.78);
  params.setParB(0, 41, 2.29);
  params.setParB(1, 41, 1.68);
  params.setParB(0, 42, 2.28);
  params.setParB(1, 42, 1.91);
  params.setParB(0, 43, 2.31);
  params.setParB(1, 43, 1.78);
  params.setParB(0, 44, 2.06);
  params.setParB(1, 44, 1.88);
  params.setParB(0, 45, 2.12);
  params.setParB(1, 45, 2.08);
  params.setParB(0, 46, 1.82);
  params.setParB(1, 46, 1.74);
  params.setParB(0, 47, 2.22);
  params.setParB(1, 47, 2.01);
  params.setParB(0, 48, 2.45);
  params.setParB(1, 48, 1.90);
  params.setParB(0, 49, 2.58);
  params.setParB(1, 49, 1.77);
  params.setParB(0, 50, 2.33);
  params.setParB(1, 50, 1.77);
  params.setParB(0, 51, 2.31);
  params.setParB(1, 51, 2.11);
  params.setParB(0, 52, 1.74);
  params.setParB(1, 52, 1.84);
  params.setParB(0, 53, 2.07);
  params.setParB(1, 53, 2.09);
  params.setParB(0, 54, 2.10);
  params.setParB(1, 54, 2.13);
  params.setParB(0, 55, 2.23);
  params.setParB(1, 55, 1.88);
  params.setParB(0, 56, 2.22);
  params.setParB(1, 56, 1.93);
  params.setParB(0, 57, 2.60);
  params.setParB(1, 57, 2.01);
  params.setParB(0, 58, 2.29);
  params.setParB(1, 58, 2.16);
  params.setParB(0, 59, 2.75);
  params.setParB(1, 59, 2.19);
  params.setParB(0, 60, 2.29);
  params.setParB(1, 60, 2.01);
  params.setParB(0, 61, 2.22);
  params.setParB(1, 61, 1.90);
  params.setParB(0, 62, 2.01);
  params.setParB(1, 62, 1.96);
  params.setParB(0, 63, 2.06);
  params.setParB(1, 63, 1.94);
  params.setParB(0, 64, 2.18);
  params.setParB(1, 64, 1.79);
  params.setParB(0, 65, 2.30);
  params.setParB(1, 65, 2.04);
  params.setParB(0, 66, 2.66);
  params.setParB(1, 66, 1.98);
  params.setParB(0, 67, 2.26);
  params.setParB(1, 67, 1.97);
  params.setParB(0, 68, 2.72);
  params.setParB(1, 68, 1.89);
  params.setParB(0, 69, 2.23);
  params.setParB(1, 69, 1.85);
  params.setParB(0, 70, 2.25);
  params.setParB(1, 70, 1.94);
  params.setParB(0, 71, 2.00);
  params.setParB(1, 71, 1.99);

  return std::move(params);
}

} // namespace mid
} // namespace o2
