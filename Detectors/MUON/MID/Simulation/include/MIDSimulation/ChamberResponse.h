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

/// \file   MIDSimulation/ChamberResponse.h
/// \brief  MID RPC response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 April 2018

#ifndef O2_MID_CHAMBERRESPONSE_H
#define O2_MID_CHAMBERRESPONSE_H

#include <unordered_map>

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "MIDSimulation/ChamberResponseParams.h"
#include "MIDSimulation/ChamberHV.h"

namespace o2
{
namespace mid
{
class ChamberResponse
{
 public:
  /// @brief Constructor
  /// @param params Chamber response parameters
  /// @param hv Chamber high-voltage values
  ChamberResponse(const ChamberResponseParams& params, const ChamberHV& hv);
  /// Default destructor
  virtual ~ChamberResponse() = default;

  /// @brief Checks if the strip at a certain distance from the impact point is fired given a probability prob
  /// @param prob Probability to be fired
  /// @param distance Distance between the hit and the current strip
  /// @param cathode Anode or cathode
  /// @param deId Detection element ID
  /// @param theta Particle impact angle
  /// @return true if the strip is fired
  inline bool isFired(double prob, double distance, int cathode, int deId, double theta = 0.) const
  {
    return (prob < getFiredProbability(distance, cathode, deId, theta));
  }

  /// @brief Returns the fired probability
  /// @param distance Distance between the hit and the current strip
  /// @param cathode Anode or cathode
  /// @param deId Detection element ID
  /// @param theta Particle impact angle
  /// @return The probability that the strip is fired
  double getFiredProbability(double distance, int cathode, int deId, double theta = 0.) const;

  /// @brief Fired probability distribution
  /// @param var Pointer with function variables
  /// @param par Pointer with function parameters
  /// @return Fired probability
  double firedProbabilityFunction(double* var, double* par);

  /// @brief Gets the response parameters
  /// @return Response function parameters
  ChamberResponseParams getResponseParams() const { return mParams; }

  /// @brief Sets the HV from the DCS data points
  /// @param dpMap Map with DCS data points
  inline void setHV(const std::unordered_map<o2::dcs::DataPointIdentifier, std::vector<o2::dcs::DataPointValue>>& dpMap) { mHV.setHV(dpMap); }

 private:
  ChamberResponseParams mParams; ///< Chamber response parameters
  ChamberHV mHV;                 ///< HV values for chambers
};

ChamberResponse createDefaultChamberResponse();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBERRESPONSE_H */
