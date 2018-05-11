// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MCANALYSIS_FILE_WRAPPER_H_
#define MCANALYSIS_FILE_WRAPPER_H_

#include <string>
#include <vector>
#include <memory> // std::unique_ptr

#include "FairLogger.h"

#include "MCStepLogger/MetaInfo.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"

namespace o2
{
namespace mcstepanalysis
{

/*
 * Interface to files produced with the analysis framework
 * Write files, read them form disk and provide access to histograms and meta information
 */
class MCAnalysisFileWrapper
{
 public:
  MCAnalysisFileWrapper();
  ~MCAnalysisFileWrapper() = default;
  // \todo make a deep copy?!
  MCAnalysisFileWrapper(const MCAnalysisFileWrapper&) = default;
  MCAnalysisFileWrapper& operator=(const MCAnalysisFileWrapper&) = default;
  //
  // monitoring, control
  //
  bool isSane() const;
  //
  // file system methods
  //
  /// read analysis obects from file
  bool read(const std::string& filepath);
  /// write to a directory, analysis name and file name are derived here
  void write(const std::string& directory) const;
  //
  // retrieve and modify histogram content
  //
  /// check if some histogram is present in general
  bool hasHistogram(const std::string& name);
  /// get one histogram by name, exit if not present
  template <typename T = TH1>
  T& getHistogram(const std::string& name)
  {
    return *castHistogram<T>(findHistogram(name), false);
  }
  /// get a 1D histogram and directly register it to this this wrapper
  template <typename T>
  T& getHistogram(const std::string& name, int nBins, double lower, double upper)
  {
    static_assert(std::is_base_of<TH1, T>::value, "the requested object type does not derive from TH1");
    T* histogramSearch = castHistogram<T>(findHistogram(name));
    if (histogramSearch) {
      return *histogramSearch;
    }
    mHistograms.push_back(std::make_shared<T>(name.c_str(), "", nBins, lower, upper));
    mAnalysisMetaInfo.nHistograms++;
    mHasChanged = true;
    return *(dynamic_cast<T*>(mHistograms.back().get()));
  }
  /// get a 2D histogram and directly register it to this this wrapper
  template <typename T>
  T& getHistogram(const std::string& name, int nBinsX, double lowerX, double upperX,
                  int nBinsY, double lowerY, double upperY)
  {
    static_assert(std::is_base_of<TH2, T>::value, "the requested object type does not derive from TH2");
    T* histogramSearch = castHistogram<T>(findHistogram(name));
    if (histogramSearch) {
      return *histogramSearch;
    }
    mHistograms.push_back(std::make_shared<T>(name.c_str(), "", nBinsX, lowerX, upperX, nBinsY, lowerY, upperY));
    mAnalysisMetaInfo.nHistograms++;
    mHasChanged = true;
    return *(dynamic_cast<T*>(mHistograms.back().get()));
  }
  /// get a 3D histogram and directly register it to this this wrapper
  template <typename T>
  T& getHistogram(const std::string& name, int nBinsX, double lowerX, double upperX,
                  int nBinsY, double lowerY, double upperY,
                  int nBinsZ, double lowerZ, double upperZ)
  {
    T* histogramSearch = castHistogram<T>(findHistogram(name));
    if (histogramSearch) {
      return *histogramSearch;
    }
    static_assert(std::is_base_of<TH3, T>::value, "the requested object type does not derive from TH3");
    mHistograms.push_back(std::make_shared<T>(name.c_str(), "", nBinsX, lowerX, upperX, nBinsY, lowerY, upperY, nBinsZ, lowerZ, upperZ));
    mAnalysisMetaInfo.nHistograms++;
    mHasChanged = true;
    return *(dynamic_cast<T*>(mHistograms.back().get()));
  }
  //
  // retrieving meta information
  //
  /// getting the meta info of the analysis run
  MCAnalysisMetaInfo& getAnalysisMetaInfo();
  //
  // verbosity
  //
  /// print separate the meta info
  void printAnalysisMetaInfo() const;
  // print histogram names, available options are "base", "range", "all". For more information see documenation of TH1::Print()
  void printHistogramInfo(const std::string& option = "") const;
  /// check for and create a directory
  static bool createDirectory(const std::string& dir);

 private:
  /// convert between histogram types, accept/don't accept a nullptr as argument
  template <typename T>
  T* castHistogram(TH1* histogram, bool acceptNull = true)
  {
    if (!histogram && acceptNull) {
      return nullptr;
    }
    if (!histogram) {
      LOG(FATAL) << "Not casting nullptr.";
      exit(1);
    }
    T* histoCasted = dynamic_cast<T*>(histogram);
    if (!histoCasted) {
      LOG(FATAL) << histogram->GetName() << " cannot be casted to " << typeid(T).name();
      exit(1);
    }
    return histoCasted;
  }
  /// find a histogram, return nullptr if not present
  TH1* findHistogram(const std::string& name);

 private:
  /// input file path
  std::string mInputFilepath;
  /// meta info of the analysis run
  MCAnalysisMetaInfo mAnalysisMetaInfo;
  /// histograms
  std::vector<std::shared_ptr<TH1>> mHistograms;
  /// flag to check whether object has been changed (since reading from file)
  bool mHasChanged;

  ClassDefNV(MCAnalysisFileWrapper, 1);
};

} // end namespace mcstepanalysis
} // end namespace o2
#endif /* MCANALYSIS_FILE_WRAPPER_H_ */
