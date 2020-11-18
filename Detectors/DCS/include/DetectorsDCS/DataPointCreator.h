// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_DATAPOINT_CREATOR_H
#define O2_DCS_DATAPOINT_CREATOR_H

#include "DataPointCompositeObject.h"

namespace o2::dcs
{
/**
  * createDataPointCompositeObject is a convenience function to 
  * simplify the creation of a DataPointCompositeObject.
  *
  * @param alias the DataPoint alias name (max 56 characters)
  * @param val the value of the datapoint
  * @param flags value for ADAPOS flags.
  * @param milliseconds value for milliseconds.
  * @param seconds value for seconds.
  *
  * @returns a DataPointCompositeObject
  *
  * The actual DeliveryType of the returned 
  * DataPointCompositeObject is deduced from the type of val. 
  *
  * Note that only a few relevant specialization are actually provided
  *
  * - T=int32_t : DeliveryType = RAW_INT
  * - T=uint32_t : DeliveryType = RAW_UINT
  * - T=double : DeliveryType = RAW_DOUBLE
  * - T=bool : DeliveryType = RAW_BOOL
  * - T=char : DeliveryType = RAW_CHAR
  * - T=std::string : DeliveryType = RAW_STRING
  *
  */
template <typename T>
o2::dcs::DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, T val, uint32_t seconds, uint16_t msec, uint16_t flags = 0);
} // namespace o2::dcs

#endif
