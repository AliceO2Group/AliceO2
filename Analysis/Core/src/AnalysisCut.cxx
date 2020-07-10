// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/AnalysisCut.h"

#include <iostream>
using std::cout;
using std::endl;

ClassImp(AnalysisCut);

vector<int> AnalysisCut::fgUsedVars = {};

//____________________________________________________________________________
AnalysisCut::AnalysisCut() : TNamed(),
                             fCuts()
{
  //
  // default constructor
  //
}

//____________________________________________________________________________
AnalysisCut::AnalysisCut(const char* name, const char* title) : TNamed(name, title),
                                                                fCuts()
{
  //
  // named constructor
  //
}

//____________________________________________________________________________
AnalysisCut::AnalysisCut(const AnalysisCut& c) = default;

//____________________________________________________________________________
AnalysisCut& AnalysisCut::operator=(const AnalysisCut& c)
{
  //
  // assignment
  //
  if (this != &c) {
    TNamed::operator=(c);
    fCuts = c.fCuts;
  }
  return (*this);
}

//____________________________________________________________________________
AnalysisCut::~AnalysisCut() = default;

//____________________________________________________________________________
bool AnalysisCut::IsSelected(float* values)
{
  //
  // apply the configured cuts
  //
  // iterate over cuts
  for (std::vector<CutContainer>::iterator it = fCuts.begin(); it != fCuts.end(); ++it) {
    // check whether a dependent variables were enabled and if they are in the requested range
    if ((*it).fDepVar != -1) {
      bool inRange = (values[(*it).fDepVar] > (*it).fDepLow && values[(*it).fDepVar] <= (*it).fDepHigh);
      if (!inRange && !((*it).fDepExclude))
        continue;
      if (inRange && (*it).fDepExclude)
        continue;
    }
    if ((*it).fDepVar2 != -1) {
      bool inRange = (values[(*it).fDepVar2] > (*it).fDep2Low && values[(*it).fDepVar2] <= (*it).fDep2High);
      if (!inRange && !((*it).fDep2Exclude))
        continue;
      if (inRange && (*it).fDep2Exclude)
        continue;
    }
    // obtain the low and high cut values (either directly as a value or from a function)
    float cutLow, cutHigh;
    if ((*it).fFuncLow)
      cutLow = ((*it).fFuncLow)->Eval(values[(*it).fDepVar]);
    else
      cutLow = ((*it).fLow);
    if ((*it).fFuncHigh)
      cutHigh = ((*it).fFuncHigh)->Eval(values[(*it).fDepVar]);
    else
      cutHigh = ((*it).fHigh);
    // apply the cut and return the decision
    bool inRange = (values[(*it).fVar] >= cutLow && values[(*it).fVar] <= cutHigh);
    if (!inRange && !((*it).fExclude))
      return false;
    if (inRange && ((*it).fExclude))
      return false;
  }

  return true;
}
