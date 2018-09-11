// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>

#include "TSystem.h" // to check for and create directories

#include "FairLogger.h"

#include "MCStepLogger/MCAnalysisFileWrapper.h"
#include "MCStepLogger/ROOTIOUtilities.h"

ClassImp(o2::mcstepanalysis::MCAnalysisFileWrapper);

using namespace o2::mcstepanalysis;

MCAnalysisFileWrapper::MCAnalysisFileWrapper()
  : mInputFilepath(""), mAnalysisMetaInfo(MCAnalysisMetaInfo()), mHasChanged(false)
{
  mHistograms.clear();
}

bool MCAnalysisFileWrapper::isSane() const
{
  bool sane = true;
  // so far only check number of histograms vs. expected number of histograms
  if (mAnalysisMetaInfo.nHistograms != mHistograms.size()) {
    LOG(WARNING) << "Histograms are corrupted: found " << mHistograms.size() << " but " << mAnalysisMetaInfo.nHistograms << " expected.";
    sane = false;
  }
  return sane;
}

bool MCAnalysisFileWrapper::read(const std::string& filepath)
{
  ROOTIOUtilities rootutil(filepath);
  // look for and read meta info of step logger and analysis run
  if (!rootutil.hasObject(defaults::mcAnalysisMetaInfoName)) {
    rootutil.close();
    return false;
  }
  rootutil.readObject(mAnalysisMetaInfo, defaults::mcAnalysisMetaInfoName);

  // try to recover histograms
  TH1* histoRecover = nullptr;

  if (!rootutil.changeToTDirectory(defaults::mcAnalysisObjectsDirName)) {
    rootutil.close();
    return false;
  }
  while (true) {
    rootutil.readObject(histoRecover);
    // assuming that all objects have been read
    if (!histoRecover) {
      break;
    }
    TH1* histo = dynamic_cast<TH1*>(histoRecover->Clone());
    histo->SetDirectory(0);
    mHistograms.push_back(std::shared_ptr<TH1>(histo));
  }
  rootutil.close();
  isSane();
  return true;
}

void MCAnalysisFileWrapper::write(const std::string& filedir) const
{
  if (!isSane()) {
    LOG(ERROR) << "Analysis file cannot be written, see above.";
    return;
  }
  const std::string outputDir = filedir + "/" + mAnalysisMetaInfo.analysisName;
  if (!createDirectory(outputDir)) {
    LOG(ERROR) << "Directory " << outputDir << " could not be created for analysis " << mAnalysisMetaInfo.analysisName << ". Skip...\n";
    return;
  }
  ROOTIOUtilities rootutil(outputDir + "/Analysis.root", ETFileMode::kRECREATE);

  LOG(INFO) << "Save histograms of analysis " << mAnalysisMetaInfo.analysisName << "\n\tat " << filedir << "\n";
  rootutil.writeObject(&mAnalysisMetaInfo, defaults::mcAnalysisMetaInfoName);
  rootutil.changeToTDirectory(defaults::mcAnalysisObjectsDirName);
  for (const auto& h : mHistograms) {
    rootutil.writeObject(h.get());
  }
  rootutil.close();
}

TH1* MCAnalysisFileWrapper::findHistogram(const std::string& name)
{
  for (auto& h : mHistograms) {
    if (name.compare(h->GetName()) == 0) {
      return h.get();
    }
  }
  return nullptr;
}

bool MCAnalysisFileWrapper::hasHistogram(const std::string& name)
{
  return (findHistogram(name) != nullptr);
}

bool MCAnalysisFileWrapper::createDirectory(const std::string& dir)
{
  gSystem->mkdir(dir.c_str(), true);
  // according to documentation returns false if possible to access
  return (gSystem->AccessPathName(dir.c_str()) == 0);
}

/// getting the meta info of the analysis run
MCAnalysisMetaInfo& MCAnalysisFileWrapper::getAnalysisMetaInfo()
{
  return mAnalysisMetaInfo;
}

void MCAnalysisFileWrapper::printAnalysisMetaInfo() const
{
  LOG(INFO) << "Meta info of analysis file";
  mAnalysisMetaInfo.print();
}

void MCAnalysisFileWrapper::printHistogramInfo(const std::string& option) const
{
  if (mHistograms.empty()) {
    return;
  }
  LOG(INFO) << "Histograms of analysis file";
  for (auto& h : mHistograms) {
    std::cout << "Histogram of class " << h->ClassName() << std::endl;
    h->Print(option.c_str());
  }
}
