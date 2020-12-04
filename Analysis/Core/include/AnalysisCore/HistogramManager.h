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
// Class to define and fill histograms
//

#ifndef HistogramManager_H
#define HistogramManager_H

#include <TString.h>
#include <TNamed.h>
#include <TList.h>
#include <THashList.h>
#include <TAxis.h>
#include <TArrayD.h>

#include <string>
#include <map>
#include <vector>
#include <list>

class HistogramManager : public TNamed
{

 public:
  HistogramManager();
  HistogramManager(const char* name, const char* title, const int maxNVars);
  ~HistogramManager() override;

  enum Constants {
    kNothing = -1
  };

  void SetMainHistogramList(THashList* list)
  {
    if (fMainList) {
      delete fMainList;
    }
    fMainList = list;
  }

  // Create a new histogram class
  void AddHistClass(const char* histClass);
  // Create a new histogram in the class <histClass> with name <name> and title <title>
  // The type of histogram is deduced from the parameters specified by the user
  // The binning for at least one dimension needs to be specified, namely: nXbins, xmin, xmax, varX which will result in a TH1F histogram
  // If the value for a variable is left as -1 then that is considered to not be used
  // Up to 3 dimensional histograms can be defined with this function
  // If isProfile = true, the last specified variable is the one being averaged
  //      For the case of TProfile3D, the user must use the varT to specify the averaged variable
  //  If specified, varW will be used as weight in the TH1::Fill() functions.
  // If specified, the xLabels, yLabels, zLabels will be used to set the labels of the x,y,z axes, respectively.
  // The axis titles will be set by default, if those were specified (e.g. taken from a Variable Manager)
  //   Otherwise these can be specified in the title string by separating them with semi-colons ";"
  void AddHistogram(const char* histClass, const char* name, const char* title, bool isProfile,
                    int nXbins, double xmin, double xmax, int varX,
                    int nYbins = 0, double ymin = 0, double ymax = 0, int varY = -1,
                    int nZbins = 0, double zmin = 0, double zmax = 0, int varZ = -1,
                    const char* xLabels = "", const char* yLabels = "", const char* zLabels = "",
                    int varT = -1, int varW = -1);
  // Similar to the above function, with the difference that the user can specify non-equidistant binning
  void AddHistogram(const char* histClass, const char* name, const char* title, bool isProfile,
                    int nXbins, double* xbins, int varX,
                    int nYbins = 0, double* ybins = nullptr, int varY = -1,
                    int nZbins = 0, double* zbins = nullptr, int varZ = -1,
                    const char* xLabels = "", const char* yLabels = "", const char* zLabels = "",
                    int varT = -1, int varW = -1);
  // Create a THn histogram (either THnF or THnSparse) with equidistant binning
  void AddHistogram(const char* histClass, const char* name, const char* title,
                    int nDimensions, int* vars, int* nBins, double* xmin, double* xmax,
                    TString* axLabels = nullptr, int varW = -1, bool useSparse = kFALSE);
  // Create a THn histogram (either THnF or THnSparse) with non-equidistant binning
  void AddHistogram(const char* histClass, const char* name, const char* title,
                    int nDimensions, int* vars, TArrayD* binLimits,
                    TString* axLabels = nullptr, int varW = -1, bool useSparse = kFALSE);

  void FillHistClass(const char* className, float* values);

  void SetUseDefaultVariableNames(bool flag) { fUseDefaultVariableNames = flag; };
  void SetDefaultVarNames(TString* vars, TString* units);
  const bool* GetUsedVars() const { return fUsedVars; }

  THashList* GetMainHistogramList() { return fMainList; } // get a histogram list

  unsigned long int GetAllocatedBins() const { return fBinsAllocated; }
  void Print(Option_t*) const override;

 private:
  THashList* fMainList; // master histogram list
  int fNVars;           // number of variables handled (tipically from the Variable Manager)

  bool* fUsedVars;                                                  //! flags of used variables
  std::map<std::string, std::list<std::vector<int>>> fVariablesMap; //!  map holding identifiers for all variables needed by histograms

  // various
  bool fUseDefaultVariableNames;    //! toggle the usage of default variable names and units
  unsigned long int fBinsAllocated; //! number of allocated bins
  TString* fVariableNames;          //! variable names
  TString* fVariableUnits;          //! variable units

  void MakeAxisLabels(TAxis* ax, const char* labels);

  HistogramManager& operator=(const HistogramManager& c);
  HistogramManager(const HistogramManager& c);

  ClassDef(HistogramManager, 1)
};

#endif
