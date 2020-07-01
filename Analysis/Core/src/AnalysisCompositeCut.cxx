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

#include <TIterator.h>

ClassImp(AnalysisCompositeCut)

//____________________________________________________________________________
AnalysisCompositeCut::AnalysisCompositeCut(bool useAND) :
  AnalysisCut(),
  fOptionUseAND(useAND)
{
  //
  // default constructor
  //
  fCuts.SetOwner(kTRUE);
}

//____________________________________________________________________________
AnalysisCompositeCut::AnalysisCompositeCut(const char* name, const char* title, bool useAND) :
  AnalysisCut(name, title),
  fOptionUseAND(useAND)
{
  //
  // named constructor
  //
  fCuts.SetOwner(kTRUE);
}

//____________________________________________________________________________
AnalysisCompositeCut::~AnalysisCompositeCut() = default;

//____________________________________________________________________________
bool AnalysisCompositeCut::IsSelected(float* values) {
  //
  // apply cuts
  //
  TIter next(&fCuts);
  for(int iCut=0; iCut<fCuts.GetEntries(); ++iCut) {
     AnalysisCut* cut = (AnalysisCut*)next();
     if(fOptionUseAND && !cut->IsSelected(values)) return kFALSE;
     if(!fOptionUseAND && cut->IsSelected(values)) return kTRUE;
  }
  if(fOptionUseAND) 
    return kTRUE;
  else 
    return kFALSE;
}
