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

#ifndef O2_MCH_ROFFILTERING_IRFRAME_FILTER_H_
#define O2_MCH_ROFFILTERING_IRFRAME_FILTER_H_

#include <functional>
#include <gsl/span>
#include "MCHROFFiltering/ROFFilter.h"
#include "CommonDataFormat/IRFrame.h"

namespace o2::mch
{
/** Returns a ROFRecord filter that selects ROFs that overlap
 * one of the given IRFrame.
 *
 * The returned filter is a function that takes a ROFRecord and returns
 * a boolean.
 *
 * @param irframes : the IRFrames (intervals of interaction records)
 * used to select ROFs
 */

ROFFilter createIRFrameFilter(gsl::span<const o2::dataformats::IRFrame> irframes);

} // namespace o2::mch

#endif
