// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/AnalysisCompositeCut.h"

ClassImp(AnalysisCompositeCut)

  //____________________________________________________________________________
  AnalysisCompositeCut::AnalysisCompositeCut(bool useAND) : AnalysisCut(),
                                                            fOptionUseAND(useAND),
                                                            fCutList()
{
  //
  // default constructor
  //
}

//____________________________________________________________________________
AnalysisCompositeCut::AnalysisCompositeCut(const char* name, const char* title, bool useAND) : AnalysisCut(name, title),
                                                                                               fOptionUseAND(useAND),
                                                                                               fCutList()
{
  //
  // named constructor
  //
}

//____________________________________________________________________________
AnalysisCompositeCut::~AnalysisCompositeCut() = default;

//____________________________________________________________________________
bool AnalysisCompositeCut::IsSelected(float* values)
{
  //
  // apply cuts
  //
  std::vector<AnalysisCut>::iterator it = fCutList.begin();
  for (std::vector<AnalysisCut>::iterator it = fCutList.begin(); it < fCutList.end(); ++it) {
    if (fOptionUseAND && !(*it).IsSelected(values))
      return kFALSE;
    if (!fOptionUseAND && (*it).IsSelected(values))
      return kTRUE;
  }

  if (fOptionUseAND)
    return kTRUE;
  else
    return kFALSE;
}
