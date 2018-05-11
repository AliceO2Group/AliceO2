// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MC_META_INFO_H_
#define MC_META_INFO_H_

#include <string>
#include <vector>
#include <iostream>

namespace o2
{
namespace mcstepanalysis
{
namespace defaults
{

const std::string mcAnalysisMetaInfoName = "MCAnalysisMetaInfo";
const std::string mcAnalysisObjectsDirName = "MCAnalysisObjects";
const std::string defaultMCAnalysisName = "defaultMCAnalysisName";
const std::string defaultLabel = "defaultLabel";
const std::string defaultStepLoggerTTreeName = "StepLoggerTree";

} // end namespace metainfonames

struct MCAnalysisMetaInfo {
  MCAnalysisMetaInfo()
    : MCAnalysisMetaInfo(defaults::defaultMCAnalysisName, defaults::defaultLabel)
  {
  }
  MCAnalysisMetaInfo(const std::string& an, const std::string& la)
    : analysisName(an), nHistograms(0), label(la)
  {
  }
  /// name of the analysis
  std::string analysisName;
  /// provide number of histograms for later sanity check
  int nHistograms;
  /// label, also shown in plots, especially useful for comparison plots
  std::string label;
  /// verbosity
  void print() const
  {
    std::cout << "Analysis name: " << analysisName << "\n";
    std::cout << "Label " << label << "\n";
  }

  ClassDefNV(MCAnalysisMetaInfo, 1);
};

} // end namespace mcstepanalysis
} // end namespace o2
#endif /* MC_META_INFO_H_ */
