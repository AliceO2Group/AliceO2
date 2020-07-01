// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Class for analysis cuts applied on the variables defined in the VarManager
//

#ifndef AnalysisVarCut_H
#define AnalysisVarCut_H

#include "Analysis/AnalysisCut.h"

#include <TF1.h>
#include <vector>

using std::vector;

//_________________________________________________________________________
class AnalysisVarCut : public AnalysisCut
{

 public:
  AnalysisVarCut(int nvars);
  AnalysisVarCut(const char* name, const char* title, int nvars);
  ~AnalysisVarCut() override;

  // NOTE: Apply a selection on variable "var" to be in the range [cutLow,cutHigh] or outside this range if "exclude" is set to true
  // NOTE: If a dependent variable is specified, then the selection is applied only if the dependent variable is in the range [depCutLow,depCutHigh]
  // NOTE:       or outside if "depCutExclude" is true
  void AddCut(int var, float cutLow, float cutHigh, bool exclude = false, 
              int dependentVar=-1, float depCutLow=0., float depCutHigh=0., bool depCutExclude=false,
              int dependentVar2=-1, float depCut2Low=0., float depCut2High=0., bool depCut2Exclude=false);
  // NOTE: Define cuts which use functions of a defined variable instead of a constant cut; the logic of the arguments is the same as for the above function
  void AddCut(int var, float cutLow, TF1* funcCutHigh, bool exclude = false,
              int dependentVar=-1, float depCutLow=0., float depCutHigh=0., bool depCutExclude=false,
              int dependentVar2=-1, float depCut2Low=0., float depCut2High=0., bool depCut2Exclude=false);
  void AddCut(int var, TF1* funcCutLow, float cutHigh, bool exclude = false,
              int dependentVar=-1, float depCutLow=0., float depCutHigh=0., bool depCutExclude=false,
              int dependentVar2=-1, float depCut2Low=0., float depCut2High=0., bool depCut2Exclude=false);
  void AddCut(int var, TF1* funcCutLow, TF1* funcCutHigh, bool exclude = false,
              int dependentVar=-1, float depCutLow=0., float depCutHigh=0., bool depCutExclude=false,
              int dependentVar2=-1, float depCut2Low=0., float depCut2High=0., bool depCut2Exclude=false);

  // TODO: implement also IsSelected() functions which take as argument the object to be selected
  //       But this would require to have access to the VarManager for extracting variables
  virtual bool IsSelected(float* values);
  
  static bool* fgUsedVars;   //! flags of used variables
  
 protected: 
   int fNVars;                           // number of variables handled (tipically from the Variable Manager)

   vector<short> fCutVariables;          // list of variables enabled to cut on
   vector<short> fDependentVariable;     // first (optional) variable on which the cut depends
   vector<short> fDependentVariable2;    // second (optional) variable on which the cut depends
   
   vector<float> fCutLow;                // lower cut limit
   vector<float> fCutHigh;               // upper cut limit
   vector<bool> fCutExclude;             // if true, then use the selection range for exclusion
   
   vector<float> fDependentVariableCutLow;   // lower limit for the first dependent variable
   vector<float> fDependentVariableCutHigh;  // upper limit for the first dependent variable
   vector<bool>  fDependentVariableExclude;  // if true, then use the dependent variable range as exclusion
   
   vector<float> fDependentVariable2CutLow;   // lower limit for the second dependent variable
   vector<float> fDependentVariable2CutHigh;  // upper limit for the second dependent variable
   vector<bool> fDependentVariable2Exclude;   // if true, then use the dependent variable range as exclusion
   
   vector<TF1*> fFuncCutLow;         // low cut functions
   vector<TF1*> fFuncCutHigh;        // upper cut functions
      
   AnalysisVarCut(const AnalysisVarCut &c);
   AnalysisVarCut& operator= (const AnalysisVarCut &c);
  
   ClassDef(AnalysisVarCut,1);
  
};
#endif
