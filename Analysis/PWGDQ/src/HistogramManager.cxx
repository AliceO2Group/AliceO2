// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PWGDQCore/HistogramManager.h"

#include <iostream>
#include <fstream>
using namespace std;

#include <TObject.h>
#include <TObjArray.h>
#include <THashList.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TProfile3D.h>
#include <THn.h>
#include <THnSparse.h>
#include <TIterator.h>
#include <TClass.h>

ClassImp(HistogramManager);

//_______________________________________________________________________________
HistogramManager::HistogramManager() : TNamed("", ""),
                                       fMainList(nullptr),
                                       fNVars(0),
                                       fUsedVars(nullptr),
                                       fVariablesMap(),
                                       fUseDefaultVariableNames(false),
                                       fBinsAllocated(0),
                                       fVariableNames(nullptr),
                                       fVariableUnits(nullptr)
{
  //
  // Constructor
  //
}

//_______________________________________________________________________________
HistogramManager::HistogramManager(const char* name, const char* title, const int maxNVars) : TNamed(name, title),
                                                                                              fMainList(),
                                                                                              fNVars(maxNVars),
                                                                                              fUsedVars(),
                                                                                              fVariablesMap(),
                                                                                              fUseDefaultVariableNames(kFALSE),
                                                                                              fBinsAllocated(0),
                                                                                              fVariableNames(),
                                                                                              fVariableUnits()
{
  //
  // Constructor
  //
  fMainList = new THashList;
  fMainList->SetOwner(kTRUE);
  fMainList->SetName(name);
  fUsedVars = new bool[maxNVars];
  fVariableNames = new TString[maxNVars];
  fVariableUnits = new TString[maxNVars];
}

//_______________________________________________________________________________
HistogramManager::~HistogramManager()
{
  //
  // De-constructor
  //
  delete fMainList;
  delete fUsedVars;
}

//_______________________________________________________________________________
void HistogramManager::SetDefaultVarNames(TString* vars, TString* units)
{
  //
  // Set default variable names
  //
  for (int i = 0; i < fNVars; ++i) {
    fVariableNames[i] = vars[i];
    fVariableUnits[i] = units[i];
  }
};

//__________________________________________________________________
void HistogramManager::AddHistClass(const char* histClass)
{
  //
  // Add a new histogram list
  //
  if (fMainList->FindObject(histClass)) {
    cout << "Warning in HistogramManager::AddHistClass(): Cannot add histogram class " << histClass
         << " because it already exists." << endl;
    return;
  }
  TList* hList = new TList;
  hList->SetOwner(kTRUE);
  hList->SetName(histClass);
  fMainList->Add(hList);
  std::list<std::vector<int>> varList;
  fVariablesMap[histClass] = varList;
  cout << "Adding histogram class " << histClass << endl;
  cout << "Variable map size :: " << fVariablesMap.size() << endl;
}

//_________________________________________________________________
void HistogramManager::AddHistogram(const char* histClass, const char* hname, const char* title, bool isProfile,
                                    int nXbins, double xmin, double xmax, int varX,
                                    int nYbins, double ymin, double ymax, int varY,
                                    int nZbins, double zmin, double zmax, int varZ,
                                    const char* xLabels, const char* yLabels, const char* zLabels,
                                    int varT, int varW)
{
  //
  // add a histogram  (this function can define TH1F,TH2F,TH3F,TProfile,TProfile2D, and TProfile3D)
  //
  // TODO: replace the cout warning messages with LOG (same for all the other functions)

  // get the list to which the histogram should be added
  TList* hList = (TList*)fMainList->FindObject(histClass);
  if (!hList) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram list " << histClass << " not found!" << endl;
    cout << "         Histogram not created" << endl;
    return;
  }
  // check whether this histogram name was used before
  if (hList->FindObject(hname)) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram " << hname << " already exists" << endl;
    return;
  }

  // deduce the dimension of the histogram from parameters
  // NOTE: in case of profile histograms, one extra variable is needed
  int dimension = 1;
  if (varY > kNothing) {
    dimension = 2;
  }
  if (varZ > kNothing) {
    dimension = 3;
  }

  // tokenize the title string; the user may include in it axis titles which will overwrite the defaults
  TString titleStr(title);
  std::unique_ptr<TObjArray> arr(titleStr.Tokenize(";"));
  // mark required variables as being used
  if (varX > kNothing) {
    fUsedVars[varX] = kTRUE;
  }
  if (varY > kNothing) {
    fUsedVars[varY] = kTRUE;
  }
  if (varZ > kNothing) {
    fUsedVars[varZ] = kTRUE;
  }
  if (varT > kNothing) {
    fUsedVars[varT] = kTRUE;
  }
  if (varW > kNothing) {
    fUsedVars[varW] = kTRUE;
  }

  // encode needed variable identifiers in a vector and push it to the std::list corresponding to the current histogram list
  std::vector<int> varVector;
  varVector.push_back(isProfile ? 1 : 0); // whether the histogram is a profile
  varVector.push_back(0);                 // whether it is a THn
  varVector.push_back(varW);              // variable used for weighting
  varVector.push_back(varX);              // variables on each axis
  varVector.push_back(varY);
  varVector.push_back(varZ);
  varVector.push_back(varT); // variable used for profiling in case of TProfile3D
  std::list varList = fVariablesMap[histClass];
  varList.push_back(varVector);
  cout << "Adding histogram " << hname << endl;
  cout << "size of array :: " << varList.size() << endl;
  fVariablesMap[histClass] = varList;

  // create and configure histograms according to required options
  TH1* h = nullptr;
  switch (dimension) {
    case 1: // TH1F
      h = new TH1F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax);
      fBinsAllocated += nXbins + 2;
      // TODO: possibly make the call of Sumw2() optional for all histograms
      h->Sumw2();
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s %s", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }
      hList->Add(h);
      h->SetDirectory(nullptr);
      break;

    case 2: // either TH2F or TProfile
      if (isProfile) {
        h = new TProfile(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax);
        fBinsAllocated += nXbins + 2;
        h->Sumw2();
        // if requested, build the profile using the profile widths instead of stat errors
        // TODO: make this option more transparent to the user ?
        if (titleStr.Contains("--s--")) {
          ((TProfile*)h)->BuildOptions(0., 0., "s");
        }
      } else {
        h = new TH2F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax, nYbins, ymin, ymax);
        fBinsAllocated += (nXbins + 2) * (nYbins + 2);
        h->Sumw2();
      }
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s %s", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }

      if (fVariableNames[varY][0]) {
        h->GetYaxis()->SetTitle(Form("%s %s", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }
      if (fVariableNames[varY][0] && isProfile) {
        h->GetYaxis()->SetTitle(Form("<%s> %s", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }
      if (arr->At(2)) {
        h->GetYaxis()->SetTitle(arr->At(2)->GetName());
      }
      if (yLabels[0] != '\0') {
        MakeAxisLabels(h->GetYaxis(), yLabels);
      }
      hList->Add(h);
      h->SetDirectory(nullptr);
      break;

    case 3: // TH3F, TProfile2D or TProfile3D
      if (isProfile) {
        if (varT > kNothing) { // TProfile3D
          h = new TProfile3D(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax, nYbins, ymin, ymax, nZbins, zmin, zmax);
          fBinsAllocated += (nXbins + 2) * (nYbins + 2) * (nZbins + 2);
          h->Sumw2();
          if (titleStr.Contains("--s--")) {
            ((TProfile3D*)h)->BuildOptions(0., 0., "s");
          }
        } else { // TProfile2D
          h = new TProfile2D(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax, nYbins, ymin, ymax);
          fBinsAllocated += (nXbins + 2) * (nYbins + 2);
          h->Sumw2();
          if (titleStr.Contains("--s--")) {
            ((TProfile2D*)h)->BuildOptions(0., 0., "s");
          }
        }
      } else { // TH3F
        h = new TH3F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xmin, xmax, nYbins, ymin, ymax, nZbins, zmin, zmax);
        fBinsAllocated += (nXbins + 2) * (nYbins + 2) * (nZbins + 2);
        h->Sumw2();
      }
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s %s", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }
      if (fVariableNames[varY][0]) {
        h->GetYaxis()->SetTitle(Form("%s %s", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }
      if (arr->At(2)) {
        h->GetYaxis()->SetTitle(arr->At(2)->GetName());
      }
      if (yLabels[0] != '\0') {
        MakeAxisLabels(h->GetYaxis(), yLabels);
      }
      if (fVariableNames[varZ][0]) {
        h->GetZaxis()->SetTitle(Form("%s %s", fVariableNames[varZ].Data(),
                                     (fVariableUnits[varZ][0] ? Form("(%s)", fVariableUnits[varZ].Data()) : "")));
      }
      if (fVariableNames[varZ][0] && isProfile && varT < 0) { // for TProfile2D
        h->GetZaxis()->SetTitle(Form("<%s> %s", fVariableNames[varZ].Data(),
                                     (fVariableUnits[varZ][0] ? Form("(%s)", fVariableUnits[varZ].Data()) : "")));
      }
      if (arr->At(3)) {
        h->GetZaxis()->SetTitle(arr->At(3)->GetName());
      }
      if (zLabels[0] != '\0') {
        MakeAxisLabels(h->GetZaxis(), zLabels);
      }
      h->SetDirectory(nullptr);
      hList->Add(h);
      break;
  } // end switch
}

//_________________________________________________________________
void HistogramManager::AddHistogram(const char* histClass, const char* hname, const char* title, bool isProfile,
                                    int nXbins, double* xbins, int varX,
                                    int nYbins, double* ybins, int varY,
                                    int nZbins, double* zbins, int varZ,
                                    const char* xLabels, const char* yLabels, const char* zLabels,
                                    int varT, int varW)
{
  //
  // add a histogram
  //

  // get the list to which the histogram should be added
  TList* hList = (TList*)fMainList->FindObject(histClass);
  if (!hList) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram list " << histClass << " not found!" << endl;
    cout << "         Histogram not created" << endl;
    return;
  }
  // check whether this histogram name was used before
  if (hList->FindObject(hname)) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram " << hname << " already exists" << endl;
    return;
  }

  // deduce the dimension of the histogram from parameters
  // NOTE: in case of profile histograms, one extra variable is needed
  int dimension = 1;
  if (varY > kNothing) {
    dimension = 2;
  }
  if (varZ > kNothing) {
    dimension = 3;
  }

  // mark required variables as being used
  if (varX > kNothing) {
    fUsedVars[varX] = kTRUE;
  }
  if (varY > kNothing) {
    fUsedVars[varY] = kTRUE;
  }
  if (varZ > kNothing) {
    fUsedVars[varZ] = kTRUE;
  }
  if (varT > kNothing) {
    fUsedVars[varT] = kTRUE;
  }
  if (varW > kNothing) {
    fUsedVars[varW] = kTRUE;
  }

  // tokenize the title string; the user may include in it axis titles which will overwrite the defaults
  TString titleStr(title);
  std::unique_ptr<TObjArray> arr(titleStr.Tokenize(";"));

  // encode needed variable identifiers in a vector and push it to the std::list corresponding to the current histogram list
  std::vector<int> varVector;
  varVector.push_back(isProfile ? 1 : 0); // whether the histogram is a profile
  varVector.push_back(0);                 // whether it is a THn
  varVector.push_back(varW);              // variable used for weighting
  varVector.push_back(varX);              // variables on each axis
  varVector.push_back(varY);
  varVector.push_back(varZ);
  varVector.push_back(varT); // variable used for profiling in case of TProfile3D
  std::list varList = fVariablesMap[histClass];
  varList.push_back(varVector);
  cout << "Adding histogram " << hname << endl;
  cout << "size of array :: " << varList.size() << endl;
  fVariablesMap[histClass] = varList;

  TH1* h = nullptr;
  switch (dimension) {
    case 1:
      h = new TH1F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins);
      fBinsAllocated += nXbins + 2;
      h->Sumw2();
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s %s", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }
      h->SetDirectory(nullptr);
      hList->Add(h);
      break;

    case 2:
      if (isProfile) {
        h = new TProfile(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins);
        fBinsAllocated += nXbins + 2;
        h->Sumw2();
        if (titleStr.Contains("--s--")) {
          ((TProfile*)h)->BuildOptions(0., 0., "s");
        }
      } else {
        h = new TH2F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins, nYbins, ybins);
        fBinsAllocated += (nXbins + 2) * (nYbins + 2);
        h->Sumw2();
      }
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s (%s)", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }
      if (fVariableNames[varY][0]) {
        h->GetYaxis()->SetTitle(Form("%s (%s)", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }
      if (fVariableNames[varY][0] && isProfile) {
        h->GetYaxis()->SetTitle(Form("<%s> (%s)", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }

      if (arr->At(2)) {
        h->GetYaxis()->SetTitle(arr->At(2)->GetName());
      }
      if (yLabels[0] != '\0') {
        MakeAxisLabels(h->GetYaxis(), yLabels);
      }
      h->SetDirectory(nullptr);
      hList->Add(h);
      break;

    case 3:
      if (isProfile) {
        if (varT > kNothing) {
          h = new TProfile3D(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins, nYbins, ybins, nZbins, zbins);
          fBinsAllocated += (nXbins + 2) * (nYbins + 2) * (nZbins + 2);
          h->Sumw2();
          if (titleStr.Contains("--s--")) {
            ((TProfile3D*)h)->BuildOptions(0., 0., "s");
          }
        } else {
          h = new TProfile2D(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins, nYbins, ybins);
          fBinsAllocated += (nXbins + 2) * (nYbins + 2);
          h->Sumw2();
          if (titleStr.Contains("--s--")) {
            ((TProfile2D*)h)->BuildOptions(0., 0., "s");
          }
        }
      } else {
        h = new TH3F(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nXbins, xbins, nYbins, ybins, nZbins, zbins);
        fBinsAllocated += (nXbins + 2) * (nYbins + 2) * (nZbins + 2);
        h->Sumw2();
      }
      if (fVariableNames[varX][0]) {
        h->GetXaxis()->SetTitle(Form("%s %s", fVariableNames[varX].Data(),
                                     (fVariableUnits[varX][0] ? Form("(%s)", fVariableUnits[varX].Data()) : "")));
      }
      if (arr->At(1)) {
        h->GetXaxis()->SetTitle(arr->At(1)->GetName());
      }
      if (xLabels[0] != '\0') {
        MakeAxisLabels(h->GetXaxis(), xLabels);
      }
      if (fVariableNames[varY][0]) {
        h->GetYaxis()->SetTitle(Form("%s %s", fVariableNames[varY].Data(),
                                     (fVariableUnits[varY][0] ? Form("(%s)", fVariableUnits[varY].Data()) : "")));
      }
      if (arr->At(2)) {
        h->GetYaxis()->SetTitle(arr->At(2)->GetName());
      }
      if (yLabels[0] != '\0') {
        MakeAxisLabels(h->GetYaxis(), yLabels);
      }
      if (fVariableNames[varZ][0]) {
        h->GetZaxis()->SetTitle(Form("%s %s", fVariableNames[varZ].Data(),
                                     (fVariableUnits[varZ][0] ? Form("(%s)", fVariableUnits[varZ].Data()) : "")));
      }
      if (fVariableNames[varZ][0] && isProfile && varT < 0) { // TProfile2D
        h->GetZaxis()->SetTitle(Form("<%s> %s", fVariableNames[varZ].Data(),
                                     (fVariableUnits[varZ][0] ? Form("(%s)", fVariableUnits[varZ].Data()) : "")));
      }

      if (arr->At(3)) {
        h->GetZaxis()->SetTitle(arr->At(3)->GetName());
      }
      if (zLabels[0] != '\0') {
        MakeAxisLabels(h->GetZaxis(), zLabels);
      }
      hList->Add(h);
      break;
  } // end switch(dimension)
}

//_________________________________________________________________
void HistogramManager::AddHistogram(const char* histClass, const char* hname, const char* title,
                                    int nDimensions, int* vars, int* nBins, double* xmin, double* xmax,
                                    TString* axLabels, int varW, bool useSparse)
{
  //
  // add a multi-dimensional histogram THnF or THnFSparseF
  //

  // get the list to which the histogram should be added
  TList* hList = (TList*)fMainList->FindObject(histClass);
  if (!hList) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram list " << histClass << " not found!" << endl;
    cout << "         Histogram not created" << endl;
    return;
  }
  // check whether this histogram name was used before
  if (hList->FindObject(hname)) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram " << hname << " already exists" << endl;
    return;
  }

  // tokenize the title string; the user may include in it axis titles which will overwrite the defaults
  TString titleStr(title);
  std::unique_ptr<TObjArray> arr(titleStr.Tokenize(";"));

  if (varW > kNothing) {
    fUsedVars[varW] = kTRUE;
  }

  // encode needed variable identifiers in a vector and push it to the std::list corresponding to the current histogram list
  std::vector<int> varVector;
  varVector.push_back(0);           // whether the histogram is a profile
  varVector.push_back(nDimensions); // number of dimensions
  varVector.push_back(varW);        // variable used for weighting
  for (int idim = 0; idim < nDimensions; ++idim) {
    varVector.push_back(vars[idim]); // axes variables
  }
  std::list varList = fVariablesMap[histClass];
  varList.push_back(varVector);
  cout << "Adding histogram " << hname << endl;
  cout << "size of array :: " << varList.size() << endl;
  fVariablesMap[histClass] = varList;

  unsigned long int nbins = 1;
  THnBase* h = nullptr;
  if (useSparse) {
    h = new THnSparseF(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nDimensions, nBins, xmin, xmax);
  } else {
    h = new THnF(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nDimensions, nBins, xmin, xmax);
  }
  h->Sumw2();

  // configure the THn histogram and count the allocated bins
  for (int idim = 0; idim < nDimensions; ++idim) {
    nbins *= (nBins[idim] + 2);
    TAxis* axis = h->GetAxis(idim);
    if (fVariableNames[vars[idim]][0]) {
      axis->SetTitle(Form("%s %s", fVariableNames[vars[idim]].Data(),
                          (fVariableUnits[vars[idim]][0] ? Form("(%s)", fVariableUnits[vars[idim]].Data()) : "")));
    }
    if (arr->At(1 + idim)) {
      axis->SetTitle(arr->At(1 + idim)->GetName());
    }
    if (axLabels && !axLabels[idim].IsNull()) {
      MakeAxisLabels(axis, axLabels[idim].Data());
    }

    fUsedVars[vars[idim]] = kTRUE;
  }
  if (useSparse) {
    hList->Add((THnSparseF*)h);
  } else {
    hList->Add((THnF*)h);
  }

  fBinsAllocated += nbins;
}

//_________________________________________________________________
void HistogramManager::AddHistogram(const char* histClass, const char* hname, const char* title,
                                    int nDimensions, int* vars, TArrayD* binLimits,
                                    TString* axLabels, int varW, bool useSparse)
{
  //
  // add a multi-dimensional histogram THnF or THnSparseF with equal or variable bin widths
  //

  // get the list to which the histogram should be added
  TList* hList = (TList*)fMainList->FindObject(histClass);
  if (!hList) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram list " << histClass << " not found!" << endl;
    cout << "         Histogram not created" << endl;
    return;
  }
  // check whether this histogram name was used before
  if (hList->FindObject(hname)) {
    cout << "Warning in HistogramManager::AddHistogram(): Histogram " << hname << " already exists" << endl;
    return;
  }

  // tokenize the title string; the user may include in it axis titles which will overwrite the defaults
  TString titleStr(title);
  std::unique_ptr<TObjArray> arr(titleStr.Tokenize(";"));

  if (varW > kNothing) {
    fUsedVars[varW] = kTRUE;
  }

  // encode needed variable identifiers in a vector and push it to the std::list corresponding to the current histogram list
  std::vector<int> varVector;
  varVector.push_back(0);           // whether the histogram is a profile
  varVector.push_back(nDimensions); // number of dimensions
  varVector.push_back(varW);        // variable used for weighting
  for (int idim = 0; idim < nDimensions; ++idim) {
    varVector.push_back(vars[idim]); // axes variables
  }
  std::list varList = fVariablesMap[histClass];
  varList.push_back(varVector);
  cout << "Adding histogram " << hname << endl;
  cout << "size of array :: " << varList.size() << endl;
  fVariablesMap[histClass] = varList;

  // get the min and max for each axis
  double* xmin = new double[nDimensions];
  double* xmax = new double[nDimensions];
  int* nBins = new int[nDimensions];
  for (int idim = 0; idim < nDimensions; ++idim) {
    nBins[idim] = binLimits[idim].GetSize() - 1;
    xmin[idim] = binLimits[idim][0];
    xmax[idim] = binLimits[idim][nBins[idim]];
  }

  // initialize the THn with equal spaced bins
  THnBase* h = nullptr;
  if (useSparse) {
    h = new THnSparseF(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nDimensions, nBins, xmin, xmax);
  } else {
    h = new THnF(hname, (arr->At(0) ? arr->At(0)->GetName() : ""), nDimensions, nBins, xmin, xmax);
  }
  // rebin the axes according to the user requested binning
  for (int idim = 0; idim < nDimensions; ++idim) {
    TAxis* axis = h->GetAxis(idim);
    axis->Set(nBins[idim], binLimits[idim].GetArray());
  }
  h->Sumw2();

  unsigned long int bins = 1;
  for (int idim = 0; idim < nDimensions; ++idim) {
    bins *= (nBins[idim] + 2);
    TAxis* axis = h->GetAxis(idim);
    if (fVariableNames[vars[idim]][0]) {
      axis->SetTitle(Form("%s %s", fVariableNames[vars[idim]].Data(),
                          (fVariableUnits[vars[idim]][0] ? Form("(%s)", fVariableUnits[vars[idim]].Data()) : "")));
    }
    if (arr->At(1 + idim)) {
      axis->SetTitle(arr->At(1 + idim)->GetName());
    }
    if (axLabels && !axLabels[idim].IsNull()) {
      MakeAxisLabels(axis, axLabels[idim].Data());
    }
    fUsedVars[vars[idim]] = kTRUE;
  }
  if (useSparse) {
    hList->Add((THnSparseF*)h);
  } else {
    hList->Add((THnF*)h);
  }
  fBinsAllocated += bins;
}

//__________________________________________________________________
void HistogramManager::FillHistClass(const char* className, Float_t* values)
{
  //
  //  fill a class of histograms
  //

  // get the needed histogram list
  TList* hList = (TList*)fMainList->FindObject(className);
  if (!hList) {
    // TODO: add some meaningfull error message
    /*cout << "Warning in HistogramManager::FillHistClass(): Histogram list " << className << " not found!" << endl;
    cout << "         Histogram list not filled" << endl; */
    return;
  }

  // get the corresponding std::list containng identifiers to the needed variables to be filled
  list varList = fVariablesMap[className];

  TIter next(hList);
  auto varIter = varList.begin();

  TObject* h = nullptr;
  bool isProfile;
  bool isTHn;
  int dimension = 0;
  bool isSparse = kFALSE;
  // TODO: At the moment, maximum 20 dimensions are foreseen for the THn histograms. We should make this more dynamic
  //       But maybe its better to have it like to avoid dynamically allocating this array in the histogram loop
  double fillValues[20] = {0.0};
  int varX = -1, varY = -1, varZ = -1, varT = -1, varW = -1;

  // loop over the histogram and std::list
  // NOTE: these two should contain the same number of elements and be synchronized, otherwise its a mess
  for (auto varIter = varList.begin(); varIter != varList.end(); varIter++) {
    h = next(); // get the histogram
    // decode information from the vector of indices
    isProfile = (varIter->at(0) == 1 ? true : false);
    isTHn = (varIter->at(1) > 0 ? true : false);
    if (isTHn) {
      dimension = varIter->at(1);
    } else {
      dimension = ((TH1*)h)->GetDimension();
    }

    // get the various variable indices
    varW = varIter->at(2);
    if (isTHn) {
      for (int i = 0; i < dimension; i++) {
        fillValues[i] = values[varIter->at(3 + i)];
      }
    } else {
      varX = varIter->at(3);
      varY = varIter->at(4);
      varZ = varIter->at(5);
      varT = varIter->at(6);
    }

    if (!isTHn) {
      switch (dimension) {
        case 1:
          if (isProfile) {
            if (varW > kNothing) {
              ((TProfile*)h)->Fill(values[varX], values[varY], values[varW]);
            } else {
              ((TProfile*)h)->Fill(values[varX], values[varY]);
            }
          } else {
            if (varW > kNothing) {
              ((TH1F*)h)->Fill(values[varX], values[varW]);
            } else {
              ((TH1F*)h)->Fill(values[varX]);
            }
          }
          break;
        case 2:
          if (isProfile) {
            if (varW > kNothing) {
              ((TProfile2D*)h)->Fill(values[varX], values[varY], values[varZ], values[varW]);
            } else {
              ((TProfile2D*)h)->Fill(values[varX], values[varY], values[varZ]);
            }
          } else {
            if (varW > kNothing) {
              ((TH2F*)h)->Fill(values[varX], values[varY], values[varW]);
            } else {
              ((TH2F*)h)->Fill(values[varX], values[varY]);
            }
          }
          break;
        case 3:
          if (isProfile) {
            if (varW > kNothing) {
              ((TProfile3D*)h)->Fill(values[varX], values[varY], values[varZ], values[varT], values[varW]);
            } else {
              ((TProfile3D*)h)->Fill(values[varX], values[varY], values[varZ], values[varT]);
            }
          } else {
            if (varW > kNothing) {
              ((TH3F*)h)->Fill(values[varX], values[varY], values[varZ], values[varW]);
            } else {
              ((TH3F*)h)->Fill(values[varX], values[varY], values[varZ]);
            }
          }
          break;

        default:
          break;
      } // end switch
    }   // end if(!isTHn)
    else {
      if (varW > kNothing) {
        if (isSparse) {
          ((THnSparseF*)h)->Fill(fillValues, values[varW]);
        } else {
          ((THnF*)h)->Fill(fillValues, values[varW]);
        }
      } else {
        if (isSparse) {
          ((THnSparseF*)h)->Fill(fillValues);
        } else {
          ((THnF*)h)->Fill(fillValues);
        }
      }
    } // end else
  }   // end loop over histograms
}

//____________________________________________________________________________________
void HistogramManager::MakeAxisLabels(TAxis* ax, const char* labels)
{
  //
  // add bin labels to an axis
  //
  TString labelsStr(labels);
  std::unique_ptr<TObjArray> arr(labelsStr.Tokenize(";"));
  for (int ib = 1; ib <= ax->GetNbins(); ++ib) {
    if (ib >= arr->GetEntries() + 1) {
      break;
    }
    ax->SetBinLabel(ib, arr->At(ib - 1)->GetName());
  }
}

//____________________________________________________________________________________
void HistogramManager::Print(Option_t*) const
{
  //
  // Print the defined histograms
  //
  cout << "###################################################################" << endl;
  cout << "HistogramManager:: " << fMainList->GetName() << endl;
  for (int i = 0; i < fMainList->GetEntries(); ++i) {
    TList* list = (TList*)fMainList->At(i);
    cout << "************** List " << list->GetName() << endl;
    for (int j = 0; j < list->GetEntries(); ++j) {
      TObject* obj = list->At(j);
      cout << obj->GetName() << ": " << obj->IsA()->GetName() << endl;
    }
  }
}
