// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/ChamberResponse.cxx
/// \brief  Implementation MID RPC response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 April 2018

/// This class implements the RPC spatial resolution.
/// The original functional form is based on this work:
/// R.~Arnaldi {\it et al.} [ALICE Collaboration],
/// %``Spatial resolution of RPC in streamer mode,''
/// Nucl.\ Instrum.\ Meth.\ A {\bf 490} (2002) 51.
/// doi:10.1016/S0168-9002(02)00917-8
/// The parameters were further tuned by Massimiliano Marchisone in his PhD thesis:
/// http://www.theses.fr/2013CLF22406

#include "MIDSimulation/ChamberResponse.h"

#include <cmath>

namespace o2
{
namespace mid
{
//______________________________________________________________________________
ChamberResponse::ChamberResponse(const ChamberResponseParams& params, const ChamberHV& hv) : mParams(params), mHV(hv)
{
  /// Constructor
}

//______________________________________________________________________________
double ChamberResponse::getFiredProbability(double distance, int cathode, int deId, double theta) const
{
  /// Get fired probability

  // Need to convert the distance from cm to mm
  double distMM = distance * 10.;
  double parA = mParams.getParA(mHV.getHV(deId));
  double parB = mParams.getParB(cathode, deId);
  double parC = mParams.getParC(mHV.getHV(deId));
  double costheta = std::cos(theta);
  return (parC + parA / (parA + costheta * std::pow(distMM, parB))) / (1 + parC);
}

//______________________________________________________________________________
double ChamberResponse::firedProbabilityFunction(double* var, double* par)
{
  /// Member function that can be used to tune the parameters
  /// @param var Variables
  /// @param par Parameters
  /// The variables are:
  /// var[0] -> distance from impact point (in cm)
  /// The parameters are:
  /// par[0] -> cathode
  /// par[1] -> deId
  /// par[2] -> impact angle (set to 0.)
  /// par[3] -> parameter B
  /// par[4] -> parameter A0
  /// par[5] -> parameter A1
  /// par[6] -> parameter C0
  /// par[7] -> parameter C1

  int cathode = (int)par[0];
  int deId = (int)par[1];

  mParams.setParB(cathode, deId, par[3]);
  mParams.setParA(par[4], par[5]);
  mParams.setParC(par[6], par[7]);

  return getFiredProbability(var[0], cathode, deId, par[2]);
}

//______________________________________________________________________________
ChamberResponse createDefaultChamberResponse()
{
  /// Returns the default chamber response
  ChamberResponseParams params = createDefaultChamberResponseParams();
  ChamberHV hv = createDefaultChamberHV();
  return ChamberResponse(params, hv);
}

} // namespace mid
} // namespace o2
