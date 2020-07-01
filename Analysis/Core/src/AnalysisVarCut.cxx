// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/AnalysisVarCut.h"

ClassImp(AnalysisVarCut);

bool* AnalysisVarCut::fgUsedVars = nullptr;

//____________________________________________________________________________
AnalysisVarCut::AnalysisVarCut(int nvars) :
  AnalysisCut(),
  fNVars(nvars),
  fCutVariables(),
  fDependentVariable(),
  fDependentVariable2(),
  fCutLow(),
  fCutHigh(),
  fCutExclude(),
  fDependentVariableCutLow(),
  fDependentVariableCutHigh(),
  fDependentVariableExclude(),   
  fDependentVariable2CutLow(),
  fDependentVariable2CutHigh(),
  fDependentVariable2Exclude(),
  fFuncCutLow(),
  fFuncCutHigh()
{
  //
  // default constructor
  //
  if(fgUsedVars==nullptr) {
    fgUsedVars = new bool[nvars];
    for(int i=0; i<nvars; i++)
      fgUsedVars[i] = false;
  }
}

//____________________________________________________________________________
AnalysisVarCut::AnalysisVarCut(const char* name, const char* title, int nvars) :
  AnalysisCut(name, title),
  fNVars(nvars),
  fCutVariables(),
  fDependentVariable(),
  fDependentVariable2(),
  fCutLow(),
  fCutHigh(),
  fCutExclude(),
  fDependentVariableCutLow(),
  fDependentVariableCutHigh(),
  fDependentVariableExclude(),   
  fDependentVariable2CutLow(),
  fDependentVariable2CutHigh(),
  fDependentVariable2Exclude(),
  fFuncCutLow(),
  fFuncCutHigh()
{
  //
  // named constructor
  //
  if(fgUsedVars==nullptr) {
    fgUsedVars = new bool[nvars];
    for(int i=0; i<nvars; i++)
      fgUsedVars[i] = false;
  }
}

//____________________________________________________________________________
AnalysisVarCut::~AnalysisVarCut() = default;

//____________________________________________________________________________
void AnalysisVarCut::AddCut(int var, float cutLow, float cutHigh, bool exclude, 
                            int dependentVar, float depCutLow, float depCutHigh, bool depCutExclude,
                            int dependentVar2, float depCut2Low, float depCut2High, bool depCut2Exclude) {
  //
  //  Add a cut
  //
  // check whether the variable identifiers are out of range
  // TODO: check the behaviour; throw warnings so the user is aware
  if(var<0 || var>=fNVars)   
    return;
  if(dependentVar != -1 && (dependentVar<0 || dependentVar>=fNVars))
    return;
  if(dependentVar2 != -1 && (dependentVar2<0 || dependentVar2>=fNVars))
    return;
  
  fCutVariables.push_back(var); 
  fCutLow.push_back(cutLow); 
  fCutHigh.push_back(cutHigh); 
  fCutExclude.push_back(exclude);
  fFuncCutLow.push_back(nullptr);
  fFuncCutHigh.push_back(nullptr);
  fgUsedVars[var] = true;

  fDependentVariable.push_back(dependentVar); 
  fDependentVariableCutLow.push_back(depCutLow); 
  fDependentVariableCutHigh.push_back(depCutHigh); 
  fDependentVariableExclude.push_back(depCutExclude);
  if(dependentVar!=-1) 
    fgUsedVars[dependentVar] = true;
  
  fDependentVariable2.push_back(dependentVar2); 
  fDependentVariable2CutLow.push_back(depCut2Low); 
  fDependentVariable2CutHigh.push_back(depCut2High); 
  fDependentVariable2Exclude.push_back(depCut2Exclude);
  if(dependentVar2!=-1) 
    fgUsedVars[dependentVar2] = true;
}

//____________________________________________________________________________
void AnalysisVarCut::AddCut(int var, float cutLow, TF1* funcCutHigh, bool exclude,
                            int dependentVar, float depCutLow, float depCutHigh, bool depCutExclude,
                            int dependentVar2, float depCut2Low, float depCut2High, bool depCut2Exclude) {
  //
  // Add a cut with a function as a high cut
  //
  // check whether the variable identifiers are out of range
  // TODO: check the behaviour; throw warnings so the user is aware
  if(var<0 || var>=fNVars)   
    return;
  if(dependentVar<0 || dependentVar>=fNVars)
    return;
  if(dependentVar2 != -1 && (dependentVar2<0 || dependentVar2>=fNVars))
    return;
  if(!funcCutHigh)
    return;
  
  fCutVariables.push_back(var); 
  fCutLow.push_back(cutLow); 
  fCutHigh.push_back(-9999.); 
  fCutExclude.push_back(exclude);
  fFuncCutLow.push_back(nullptr);
  fFuncCutHigh.push_back(funcCutHigh);
  fgUsedVars[var] = true;
  
  fDependentVariable.push_back(dependentVar); 
  fDependentVariableCutLow.push_back(depCutLow); 
  fDependentVariableCutHigh.push_back(depCutHigh); 
  fDependentVariableExclude.push_back(depCutExclude);
  if(dependentVar!=-1) 
    fgUsedVars[dependentVar] = true;
  
  fDependentVariable2.push_back(dependentVar2); 
  fDependentVariable2CutLow.push_back(depCut2Low); 
  fDependentVariable2CutHigh.push_back(depCut2High); 
  fDependentVariable2Exclude.push_back(depCut2Exclude);
  if(dependentVar2!=-1) 
    fgUsedVars[dependentVar2] = true;
}


//____________________________________________________________________________
void AnalysisVarCut::AddCut(int var, TF1* funcCutLow, float cutHigh, bool exclude,
                            int dependentVar, float depCutLow, float depCutHigh, bool depCutExclude,
                            int dependentVar2, float depCut2Low, float depCut2High, bool depCut2Exclude) {
  //
  // Add a cut with a function as a low cut
  //
  // check whether the variable identifiers are out of range
  // TODO: check the behaviour; throw warnings so the user is aware
  if(var<0 || var>=fNVars)   
    return;
  if(dependentVar<0 || dependentVar>=fNVars)
    return;
  if(dependentVar2 != -1 && (dependentVar2<0 || dependentVar2>=fNVars))
    return;
  if(!funcCutLow)
    return;
  
  fCutVariables.push_back(var); 
  fCutLow.push_back(-9999.); 
  fCutHigh.push_back(cutHigh); 
  fCutExclude.push_back(exclude);
  fFuncCutLow.push_back(funcCutLow);
  fFuncCutHigh.push_back(nullptr);
  fgUsedVars[var] = true;
  
  fDependentVariable.push_back(dependentVar); 
  fDependentVariableCutLow.push_back(depCutLow); 
  fDependentVariableCutHigh.push_back(depCutHigh); 
  fDependentVariableExclude.push_back(depCutExclude);
  if(dependentVar!=-1) 
    fgUsedVars[dependentVar] = true;
  
  fDependentVariable2.push_back(dependentVar2); 
  fDependentVariable2CutLow.push_back(depCut2Low); 
  fDependentVariable2CutHigh.push_back(depCut2High); 
  fDependentVariable2Exclude.push_back(depCut2Exclude);
  if(dependentVar2!=-1) 
    fgUsedVars[dependentVar2] = true;
}

//____________________________________________________________________________
void AnalysisVarCut::AddCut(int var, TF1* funcCutLow, TF1* funcCutHigh, bool exclude,
                            int dependentVar, float depCutLow, float depCutHigh, bool depCutExclude,
                            int dependentVar2, float depCut2Low, float depCut2High, bool depCut2Exclude) {
  //
  // Add a cut with functions as low and high cuts
  //
  // check whether the variable identifiers are out of range
  // TODO: check the behaviour; throw warnings so the user is aware
  if(var<0 || var>=fNVars)   
    return;
  if(dependentVar<0 || dependentVar>=fNVars)
    return;
  if(dependentVar2 != -1 && (dependentVar2<0 || dependentVar2>=fNVars))
    return;
  if(!funcCutLow)
    return;
  if(!funcCutHigh)
    return;
  
  fCutVariables.push_back(var); 
  fCutLow.push_back(-9999.); 
  fCutHigh.push_back(-9999.); 
  fCutExclude.push_back(exclude);
  fFuncCutLow.push_back(funcCutLow);
  fFuncCutHigh.push_back(funcCutHigh);
  fgUsedVars[var] = true;
  
  fDependentVariable.push_back(dependentVar); 
  fDependentVariableCutLow.push_back(depCutLow); 
  fDependentVariableCutHigh.push_back(depCutHigh); 
  fDependentVariableExclude.push_back(depCutExclude);
  if(dependentVar!=-1) 
    fgUsedVars[dependentVar] = true;
  
  fDependentVariable2.push_back(dependentVar2); 
  fDependentVariable2CutLow.push_back(depCut2Low); 
  fDependentVariable2CutHigh.push_back(depCut2High); 
  fDependentVariable2Exclude.push_back(depCut2Exclude);
  if(dependentVar2!=-1) 
    fgUsedVars[dependentVar2] = true;
}

//____________________________________________________________________________
bool AnalysisVarCut::IsSelected(float* values) {
  //
  // apply the configured cuts
  //
  std::vector<short>::iterator itCutVar = fCutVariables.begin();
  std::vector<float>::iterator itCutLow = fCutLow.begin();
  std::vector<float>::iterator itCutHigh = fCutHigh.begin();
  std::vector<TF1*>::iterator itFuncLow = fFuncCutLow.begin();
  std::vector<TF1*>::iterator itFuncHigh = fFuncCutHigh.begin();
  std::vector<bool>::iterator itCutExclude = fCutExclude.begin();
  
  std::vector<short>::iterator itDepVar = fDependentVariable.begin();
  std::vector<float>::iterator itDepCutLow = fDependentVariableCutLow.begin();
  std::vector<float>::iterator itDepCutHigh = fDependentVariableCutHigh.begin();
  std::vector<bool>::iterator itDepExclude = fDependentVariableExclude.begin();
  
  std::vector<short>::iterator itDepVar2 = fDependentVariable2.begin();
  std::vector<float>::iterator itDepCut2Low = fDependentVariable2CutLow.begin();
  std::vector<float>::iterator itDepCut2High = fDependentVariable2CutHigh.begin();
  std::vector<bool>::iterator itDep2Exclude = fDependentVariable2Exclude.begin();
  
  
  for(; itCutVar != fCutVariables.end(); ++itCutVar, ++itDepVar, ++itDepCutLow, ++itDepCutHigh) {
    if(*itDepVar != -1) {
      bool inRange = ( values[*itDepVar] > *itDepCutLow && values[*itDepVar] <= *itDepCutHigh );
      if(!inRange && !(*itDepExclude)) continue;
      if(inRange && *itDepExclude) continue;
    }
    if(*itDepVar2 != -1) {
      bool inRange = ( values[*itDepVar2] > *itDepCut2Low && values[*itDepVar2] <= *itDepCut2High );
      if(!inRange && !(*itDep2Exclude)) continue;
      if(inRange && *itDep2Exclude) continue;
    }
    float cutLow, cutHigh;
    if(*itFuncLow)
      cutLow = (*itFuncLow)->Eval(values[*itDepVar]);
    else
      cutLow = (*itCutLow);
    if(*itFuncHigh)
      cutHigh = (*itFuncHigh)->Eval(values[*itDepVar]);
    else
      cutHigh = (*itCutHigh);
    bool inRange = (values[*itCutVar]>=cutLow && values[*itCutVar]<=cutHigh);
    if(!inRange && !(*itCutExclude)) 
      return false;
    if(inRange && (*itCutExclude))
      return false;
  }
  return true;
}
