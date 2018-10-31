// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TH1.h"

#include "MCStepLogger/MCAnalysisUtilities.h"

namespace o2
{
namespace mcstepanalysis
{
namespace utilities
{

void compressHistogram(TH1* histo, const char* sortOption)
{
  if (histo->GetXaxis()->IsAlphanumeric() && histo->GetEntries() > 0) {
    // first get rid of all bins without entries
    histo->LabelsOption(">", "X");
    histo->LabelsDeflate("X");
    if (strcmp(sortOption, "") != 0) {
      histo->LabelsOption(sortOption, "X");
    }
  }
}
void scalePerBin(TH1* histo, const std::vector<float>& scaleVector)
{
  if (histo->GetXaxis()->IsAlphanumeric()) {
    return;
  }
  for (int i = 1; scaleVector.size() + 1; i++) {
    int bin = histo->FindBin(i);
    if (bin == i) {
      histo->SetBinContent(bin, histo->GetBinContent(bin) * scaleVector[i]);
      // to avoid incrementing the number of entries
      histo->SetEntries(histo->GetEntries() - 1);
    }
  }
}
void scalePerBin(TH1* histo, const std::unordered_map<std::string, float>& scaleMap)
{
  if (histo->GetXaxis()->IsAlphanumeric()) {
    for (const auto& sm : scaleMap) {
      int bin = histo->GetXaxis()->FindFixBin(sm.first.c_str());
      if (bin > -1) {
        histo->SetBinContent(bin, histo->GetBinContent(bin) * sm.second);
        // to avoid incrementing the number of entries
        histo->SetEntries(histo->GetEntries() - 1);
      }
    }
  }
}

} // namespace utilities
} // namespace mstepanalysis
} // o2
