// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MCANALYSIS_UTILITIES_H_
#define MCANALYSIS_UTILITIES_H_

#include <unordered_map>
#include <vector>
#include <string>

#include "TH1.h"

namespace o2
{
namespace mcstepanalysis
{
namespace utilities
{
/// compressing a histogram with alphanumeric bins and sorting accordingly
/// for sorting option see ROOT's TH1::LabelsOption
void ompressHistogram(TH1* histo, const char* sortOption);
/// scale bin i by scaleVector[i-1]
void scalePerBin(TH1* histo, const std::vector<float>& scaleVector);
/// for histograms with alphanumeric labels scale bin with name "name" by scaleMap["name"]
void scalePerBin(TH1* histo, const std::unordered_map<std::string, float>& scaleMap);
} // namespace utilities
} // namespace mstepanalysis
} // o2
#endif /* MCANALYSIS_UTILITIES_H_ */
