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

using namespace std;

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
    if (fMainList)
      delete fMainList;
    fMainList = list;
  }

  void AddHistClass(const char* histClass);
  void AddHistogram(const char* histClass, const char* name, const char* title, bool isProfile,
                    int nXbins, double xmin, double xmax, int varX,
                    int nYbins = 0, double ymin = 0, double ymax = 0, int varY = -1,
                    int nZbins = 0, double zmin = 0, double zmax = 0, int varZ = -1,
                    const char* xLabels = "", const char* yLabels = "", const char* zLabels = "",
                    int varT = -1, int varW = -1);
  void AddHistogram(const char* histClass, const char* name, const char* title, bool isProfile,
                    int nXbins, double* xbins, int varX,
                    int nYbins = 0, double* ybins = nullptr, int varY = -1,
                    int nZbins = 0, double* zbins = nullptr, int varZ = -1,
                    const char* xLabels = "", const char* yLabels = "", const char* zLabels = "",
                    int varT = -1, int varW = -1);
  void AddHistogram(const char* histClass, const char* name, const char* title,
                    int nDimensions, int* vars, int* nBins, double* xmin, double* xmax,
                    TString* axLabels = nullptr, int varW = -1, bool useSparse = kFALSE);
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
  THashList* fMainList;                                             // master histogram list
  int fNVars;                                                       // number of variables handled (tipically fromt he Variable Manager)
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
