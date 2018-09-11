// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Base class for Analyses
 * All concrete analyses must inherit from this class in order to be handled by the MCAnalysisManager.
 * The constructor registeres an analysis automatically to the MCAnalysisManager
 */

#ifndef MCANALYSIS_H_
#define MCANALYSIS_H_

#include <string>
#include <vector>

// covers all histograms
#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"

#include "MCStepLogger/StepInfo.h"
#include "MCStepLogger/MCAnalysisManager.h"
#include "MCStepLogger/MCAnalysisFileWrapper.h"

namespace o2
{
namespace mcstepanalysis
{

// /class MCAnalysisManager;

class MCAnalysis
{
  /// Add as friend so it can access protected (private) members in order to use this class
  friend class MCAnalysisManager;

 public:
  /// only constructor allowed
  MCAnalysis(const std::string& name);
  virtual ~MCAnalysis() = default;

 protected:
  //
  // steering
  //
  /// this must be overwridden, otherwise it is not possible to register histograms
  virtual void initialize() = 0;
  /// this has to be overwridden
  virtual void analyze(const std::vector<StepInfo>* const steps,
                       const std::vector<MagCallInfo>* const magCalls) = 0;
  /// this can be overwridden
  virtual void finalize() { ; }
  //
  // internal histogram managing
  //
  /// get a 1D histogram and directly register it to this analysis
  template <typename T>
  T* getHistogram(const std::string& name, int nBins, double lower, double upper)
  {
    return &mAnalysisFile->getHistogram<T>(name, nBins, lower, upper);
  }
  /// get a 2D histogram and directly register it to this analysis
  template <typename T>
  T* getHistogram(const std::string& name, int nBinsX, double lowerX, double upperX,
                  int nBinsY, double lowerY, double upperY)
  {
    return &mAnalysisFile->getHistogram<T>(name, nBinsX, lowerX, upperX, nBinsY, lowerY, upperY);
  }
  /// get a 2D histogram and directly register it to this analysis
  template <typename T>
  T* getHistogram(const std::string& name, int nBinsX, double lowerX, double upperX,
                  int nBinsY, double lowerY, double upperY,
                  int nBinsZ, double lowerZ, double upperZ)
  {
    return &mAnalysisFile->getHistogram<T>(name, nBinsX, lowerX, upperX, nBinsY, lowerY, upperY, nBinsZ, lowerZ, upperZ);
  }
  //
  // setting
  //
  /// set analysis file to write histograms to
  void setAnalysisFile(MCAnalysisFileWrapper& analysisFile)
  {
    mAnalysisFile = &analysisFile;
  }
  //
  // getting
  //
  /// get and set some status of the analysis. All of these methods are only used by the MCAnalysisManager
  bool isInitialized() const
  {
    return mIsInitialized;
  }
  /// retrieve name of the analysis
  const std::string& name() const
  {
    return mName;
  }

 private:
  /// don't allow default construction operations
  MCAnalysis() = delete;
  MCAnalysis& operator=(const MCAnalysis&) = delete;
  MCAnalysis(const MCAnalysis&) = delete;
  /// only the AnalysisManager is allowed to set this
  void isInitialized(bool val)
  {
    mIsInitialized = val;
  }

 protected:
  /// pointer to MCAnalysisManager
  // \note \todo given that derived classes have non acces to private members like void isInitialized(bool val)
  // making the MCAnalysisManager available to derived class is somehow breaking this logic since as a friend this
  // actually can access these methods. So a derived analysis can recursively access the private member functions
  // via the MCAnalysisManager pointer.
  MCAnalysisManager* mAnalysisManager;

 private:
  /// the analysis name
  std::string mName;
  /// initialisation flag
  bool mIsInitialized;
  /// save all histograms used in this analysis
  MCAnalysisFileWrapper* mAnalysisFile;

  ClassDefNV(MCAnalysis, 1);
};
} // end namespace mcstepanalysis
} // end namespace o2
#endif /* MCANALYSIS_H_ */