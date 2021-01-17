// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PWGDQCore/AnalysisCompositeCut.h"

ClassImp(AnalysisCompositeCut)

  //____________________________________________________________________________
  AnalysisCompositeCut::AnalysisCompositeCut(bool useAND) : AnalysisCut(),
                                                            fOptionUseAND(useAND),
                                                            fCutList(),
                                                            fCompositeCutList()
{
  //
  // default constructor
  //
}

//____________________________________________________________________________
AnalysisCompositeCut::AnalysisCompositeCut(const char* name, const char* title, bool useAND) : AnalysisCut(name, title),
                                                                                               fOptionUseAND(useAND),
                                                                                               fCutList(),
                                                                                               fCompositeCutList()
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
  for (std::vector<AnalysisCut>::iterator it = fCutList.begin(); it < fCutList.end(); ++it) {
    if (fOptionUseAND && !(*it).IsSelected(values)) {
      return false;
    }
    if (!fOptionUseAND && (*it).IsSelected(values)) {
      return true;
    }
  }

  for (std::vector<AnalysisCompositeCut>::iterator it = fCompositeCutList.begin(); it < fCompositeCutList.end(); ++it) {
    if (fOptionUseAND && !(*it).IsSelected(values)) {
      return false;
    }
    if (!fOptionUseAND && (*it).IsSelected(values)) {
      return true;
    }
  }

  if (fOptionUseAND) {
    return true;
  } else {
    return false;
  }
}
