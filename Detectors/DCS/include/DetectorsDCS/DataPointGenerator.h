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

#ifndef O2_DCS_DATAPOINT_GENERATOR_H
#define O2_DCS_DATAPOINT_GENERATOR_H

#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include <vector>

namespace o2::dcs
{
/**
* Generate random data points, uniformly distributed between two values.
*
* @tparam T the type of value of the data points to be generated. Only
*  a few types are supported : double, float, uint32_t, int32_t, short, bool
*
* @param aliases the list of aliases to be generated. Those can use
*   patterns that will be expanded, @see AliasExpander
* @param minValue the minimum value of the values to be generated
* @param maxValue the maximum value of the values to be generated
* @param refDate the date to be associated with all data points 
*        in `%Y-%b-%d %H:%M:%S` format. If refDate="" the current date is used.
*
* @returns a vector of DataPointCompositeObject objects
*/
template <typename T>
std::vector<DataPointCompositeObject> generateRandomDataPoints(const std::vector<std::string>& aliases,
                                                               T min,
                                                               T max,
                                                               std::string refDate = "");

} // namespace o2::dcs

#endif
