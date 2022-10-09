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

#ifndef O2_MCH_CONDITIONS_SIDE_H
#define O2_MCH_CONDITIONS_SIDE_H

#include <string>

namespace o2::mch::dcs
{

/** Side describes on which side (inside or outside) a detection element
 * (slat or quadrant) is.
 * Note that MCH DCS uses the very old left-right convention instead of the
 * agreed-upon inside-outside.
 */
enum class Side {
  Left,
  Right
};

/** name of Side (Left or Right) */
std::string name(Side side);

/** extract the side information from the alias.
 * alias must be valid otherwise the method throws an exception. */
Side aliasToSide(std::string_view alias);

} // namespace o2::mch::dcs

#endif
