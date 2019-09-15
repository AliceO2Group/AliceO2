// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/View.h>


namespace gpucf
{

std::vector<Digit> findPeaks(View<Digit>, const Map<float> &);

RowMap<std::vector<Digit>> findPeaksByRow(View<Digit>, const Map<float> &);

RowMap<Map<bool>> makePeakMapByRow(const RowMap<std::vector<Digit>> &);

bool isPeak(const Digit &, const Map<float> &, float cutoff=0.f);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
