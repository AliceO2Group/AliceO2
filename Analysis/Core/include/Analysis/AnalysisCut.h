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

#ifndef AnalysisCut_H
#define AnalysisCut_H

#include <TF1.h>
#include <vector>

using std::vector;

//_________________________________________________________________________
class AnalysisCut : public TNamed
{
 public:
  AnalysisCut();
  AnalysisCut(const char* name, const char* title);
  AnalysisCut(const AnalysisCut& c);
  AnalysisCut& operator=(const AnalysisCut& c);
  ~AnalysisCut() override;

  // NOTE: Apply a selection on variable "var" to be in the range [cutLow,cutHigh] or outside this range if "exclude" is set to true
  // NOTE: If a dependent variable is specified, then the selection is applied only if the dependent variable is in the range [depCutLow,depCutHigh]
  // NOTE:       or outside if "depCutExclude" is true
  template <typename T1, typename T2>
  void AddCut(int var, T1 cutLow, T2 cutHigh, bool exclude = false,
              int dependentVar = -1, float depCutLow = 0., float depCutHigh = 0., bool depCutExclude = false,
              int dependentVar2 = -1, float depCut2Low = 0., float depCut2High = 0., bool depCut2Exclude = false);

  // TODO: implement also IsSelected() functions which take as argument the object to be selected
  //       But this would require to have access to the VarManager for extracting variables
  virtual bool IsSelected(float* values);

  static vector<int> fgUsedVars; //! vector of used variables

  struct CutContainer {
    short fVar;    // variable to be cut upon
    float fLow;    // lower limit for the var
    float fHigh;   // upper limit for the var
    bool fExclude; // if true, use the selection range for exclusion

    short fDepVar;    // first (optional) variable on which the cut depends
    float fDepLow;    // lower limit for the first dependent var
    float fDepHigh;   // upper limit for the first dependent var
    bool fDepExclude; // if true, then use the dependent variable range as exclusion

    short fDepVar2;    // second (optional) variable on which the cut depends
    float fDep2Low;    // lower limit for the second dependent var
    float fDep2High;   // upper limit for the second dependent var
    bool fDep2Exclude; // if true, then use the dependent variable range as exclusion

    TF1* fFuncLow;  // function for the lower limit cut
    TF1* fFuncHigh; // function for the upper limit cut
  };

 protected:
  vector<CutContainer> fCuts;

  ClassDef(AnalysisCut, 1);
};

//____________________________________________________________________________
template <typename T1, typename T2>
void AnalysisCut::AddCut(int var, T1 cutLow, T2 cutHigh, bool exclude,
                         int dependentVar, float depCutLow, float depCutHigh, bool depCutExclude,
                         int dependentVar2, float depCut2Low, float depCut2High, bool depCut2Exclude)
{
  //
  // Add a cut
  //
  CutContainer cut = {};

  if constexpr (std::is_same_v<T1, TF1*>) {
    if (dependentVar < 0)
      return;
    cut.fFuncLow = cutLow;
    cut.fLow = -9999.0;
  } else {
    cut.fFuncLow = nullptr;
    cut.fLow = cutLow;
  }
  if constexpr (std::is_same_v<T2, TF1*>) {
    if (dependentVar < 0)
      return;
    cut.fFuncHigh = cutHigh;
    cut.fHigh = -9999.0;
  } else {
    cut.fFuncHigh = nullptr;
    cut.fHigh = cutHigh;
  }
  cut.fVar = var;
  cut.fExclude = exclude;
  fgUsedVars.push_back(var);

  cut.fDepVar = dependentVar;
  cut.fDepLow = depCutLow;
  cut.fDepHigh = depCutHigh;
  cut.fDepExclude = depCutExclude;
  if (dependentVar != -1)
    fgUsedVars.push_back(dependentVar);

  cut.fDepVar2 = dependentVar2;
  cut.fDep2Low = depCut2Low;
  cut.fDep2High = depCut2High;
  cut.fDep2Exclude = depCut2Exclude;
  if (dependentVar2 != -1)
    fgUsedVars.push_back(dependentVar2);

  fCuts.push_back(cut);
}

#endif
