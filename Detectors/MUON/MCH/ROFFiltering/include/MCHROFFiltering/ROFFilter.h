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

#ifndef O2_MCH_ROFFILTERING_ROF_FILTER_H_
#define O2_MCH_ROFFILTERING_ROF_FILTER_H_

#include <functional>
#include "DataFormatsMCH/ROFRecord.h"
#include <gsl/span>

namespace o2::mch
{

/* A ROFFilter is a function that takes a ROFRecord and returns true
 * if that ROF satisfies the filter criteria.
 */
typedef std::function<bool(const ROFRecord&)> ROFFilter;

/** createROFFilter returns a filter that is the AND of the filters in vector.*/
ROFFilter createROFFilter(gsl::span<const ROFFilter> filters);

} // namespace o2::mch

#endif
