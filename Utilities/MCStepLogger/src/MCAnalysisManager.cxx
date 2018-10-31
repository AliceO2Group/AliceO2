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

#include "FairLogger.h"

#include "MCStepLogger/MCAnalysisManager.h"
#include "MCStepLogger/MCAnalysis.h"
#include "MCStepLogger/MCAnalysisFileWrapper.h"
#include "MCStepLogger/ROOTIOUtilities.h"

ClassImp(o2::mcstepanalysis::MCAnalysisManager);

using namespace o2::mcstepanalysis;

void MCAnalysisManager::setInputFilepath(const std::string& filepath)
{
  mInputFilepath = filepath;
}

void MCAnalysisManager::registerAnalysis(MCAnalysis* analysis)
{
  if (!mIsInitialized) {
    for (auto& a : mAnalyses) {
      if (a->name().compare(analysis->name()) == 0) {
        LOG(ERROR) << "Analysis " << analysis->name() << " already present...skip";
        mAnalysesToDump.push_back(analysis);
      }
    }
    mAnalyses.push_back(analysis);
  }
}

bool MCAnalysisManager::checkReadiness() const
{
  std::string errorMessage;
  if (mInputFilepath.empty()) {
    errorMessage += "Input file required...\n";
  }
  if (mLabel.empty()) {
    errorMessage += "Label required...\n";
  }
  if (errorMessage.empty()) {
    return true;
  }
  LOG(ERROR) << errorMessage;
  return false;
}

void MCAnalysisManager::run(int nEvents)
{
  if (!checkReadiness()) {
    LOG(ERROR) << "MCAnalysisManager not ready to run, errors occured";
    exit(1);
  }
  initialize();
  analyze(nEvents);
  finalize();
}

bool MCAnalysisManager::dryrun()
{
  if (mInputFilepath.empty()) {
    LOG(FATAL) << "Input file required...\n";
    exit(1);
  }
  return analyze(-1, true);
}

void MCAnalysisManager::initialize()
{
  if (mIsInitialized) {
    LOG(ERROR) << "Already initialized ==> not initialized again...";
    return;
  }
  // get rid of all analyses not needed anymore
  for (auto& a : mAnalysesToDump) {
    delete a;
  }
  mAnalysesToDump.clear();

  for (auto& a : mAnalyses) {
    // prepare an analysis file and set first analysis meta info
    mAnalysisFiles.emplace_back();
    mAnalysisFiles.back().getAnalysisMetaInfo().label = mLabel;
    mAnalysisFiles.back().getAnalysisMetaInfo().analysisName = a->name();
    a->setAnalysisFile(mAnalysisFiles.back());
    a->initialize();
    a->isInitialized(true);
  }
  mIsInitialized = true;
}

bool MCAnalysisManager::analyze(int nEvents, bool isDryrun)
{
  if (!mIsInitialized && !isDryrun) {
    LOG(ERROR) << "Not yet initialized ==> nothing to analyze...";
    return false;
  }
  // taking only the first entry is a hack! \todo change this by enabling also for TChains
  ROOTIOUtilities rootutil(mInputFilepath);

  // this is by default opening the file in READ mode
  if (!rootutil.changeToTTree(mAnalysisTreename)) {
    rootutil.close();
    if (isDryrun) {
      return false;
    }
    LOG(FATAL) << "Tree " << mAnalysisTreename << " could not be found in file " << mInputFilepath;
    exit(1);
  }
  // print warning if desired number of entries is bigger than number of present entries
  if (nEvents > rootutil.nEntries()) {
    LOG(WARNING) << "You want to process " << nEvents << ", however only " << rootutil.nEntries() << " are present.";
  }

  // prepare variables and connect to branches
  // \todo align branch names with MCStepLogger and get rid of hard coded names
  if (!rootutil.setBranch("Steps", &mCurrentStepInfo) || !rootutil.setBranch("Calls", &mCurrentMagCallInfo) || !rootutil.setBranch("Lookups", &mCurrentLookups)) {
    rootutil.close();
    if (isDryrun) {
      return false;
    }
    LOG(FATAL) << "Cannot find required branches in TTree " << mAnalysisTreename;
    exit(1);
  }
  // process tree and analyze
  while (rootutil.processTTree()) {
    if (nEvents <= mCurrentEventNumber && nEvents > 0) {
      break;
    }
    // check whether all pointers to MCStepLogger branches are set...
    if (mCurrentStepInfo == nullptr || mCurrentMagCallInfo == nullptr || mCurrentLookups == nullptr) {
      rootutil.close();
      if (isDryrun) {
        return false;
      }
      LOG(FATAL) << "Obtained nullptrs while processing TTree " << mAnalysisTreename;
      exit(1);
    }
    // ... if so, next event
    mCurrentEventNumber++;
    mNSteps += mCurrentStepInfo->size();

    std::cout << "---> Event " << mCurrentEventNumber << " <---\n";
    std::cout << "#steps: " << mCurrentStepInfo->size() << "\n";
    std::cout << "#mag field calls: " << mCurrentMagCallInfo->size() << "\n";
    if (!isDryrun) {
      std::cout << "\nStart..." << std::endl;
      for (auto& a : mAnalyses) {
        std::cout << "\t\tCall analysis " << a->name() << std::endl;
        a->analyze(mCurrentStepInfo, mCurrentMagCallInfo);
      }
      std::cout << "Done\n";
    }
  }
  rootutil.close();
  if (!isDryrun) {
    LOG(INFO) << "Analysis run on file " << mInputFilepath << " done.";
    mIsAnalyzed = true;
  } else {
    mCurrentEventNumber = 0;
    mNSteps = 0;
  }
  return true;
}

void MCAnalysisManager::finalize()
{
  if (!mIsAnalyzed) {
    LOG(ERROR) << "Not yet analyzed ==> nothing to finalize...";
    return;
  }
  for (auto& a : mAnalyses) {
    a->finalize();
  }
}

void MCAnalysisManager::write(const std::string& directory) const
{
  for (auto& af : mAnalysisFiles) {
    af.write(directory);
  }
}

void MCAnalysisManager::terminate()
{
  LOG(DEBUG) << "Terminate MCAnalysisManager...";
  mIsInitialized = false;
  mIsAnalyzed = false;
  mCurrentEventNumber = 0;
  mNSteps = 0;
  mCurrentStepInfo = nullptr;
  mCurrentMagCallInfo = nullptr;
  mCurrentLookups = nullptr;
  // tell each analysis to terminat/reset
  for (auto& a : mAnalyses) {
    a->isInitialized(false);
  }
}

void MCAnalysisManager::setLabel(const std::string& label)
{
  mLabel = label;
}

void MCAnalysisManager::setStepLoggerTreename(const std::string& treename)
{
  mAnalysisTreename = treename;
}

void MCAnalysisManager::printAnalyses() const
{
  LOG(INFO) << "Analyses registered with MCAnalysisManager are:";
  for (auto& a : mAnalyses) {
    std::cout << "\t" << a->name() << "\n";
  }
}

int MCAnalysisManager::getEventNumber() const
{
  return mCurrentEventNumber;
}

void MCAnalysisManager::getLookupVolName(int volId, std::string& name) const
{
  if (volId > -1 && volId < mCurrentLookups->volidtovolname.size()) {
    if (mCurrentLookups->volidtovolname[volId] != nullptr ||
        mCurrentLookups->volidtovolname[volId]->size() != 0) {
      name = *(mCurrentLookups->volidtovolname[volId]);
      return;
    }
  }
  name = "UNKNOWNVOLNAME";
}

void MCAnalysisManager::getLookupModName(int volId, std::string& name) const
{
  if (volId > -1 && volId < mCurrentLookups->volidtomodule.size()) {
    if (mCurrentLookups->volidtomodule[volId] != nullptr ||
        mCurrentLookups->volidtomodule[volId]->size() != 0) {
      name = *(mCurrentLookups->volidtomodule[volId]);
      return;
    }
  }
  name = "UNKNOWNMODNAME";
}

void MCAnalysisManager::getLookupMedName(int volId, std::string& name) const
{
  if (volId > -1 && volId < mCurrentLookups->volidtomedium.size()) {
    if (mCurrentLookups->volidtomedium[volId] != nullptr ||
        mCurrentLookups->volidtomedium[volId]->size() != 0) {
      name = *(mCurrentLookups->volidtomedium[volId]);
      return;
    }
  }
  name = "UNKNOWNMEDNAME";
}

void MCAnalysisManager::getLookupPDG(int trackId, int& id) const
{
  if (trackId > -1 && trackId < mCurrentLookups->tracktopdg.size()) {
    id = mCurrentLookups->tracktopdg[trackId];
    return;
  }
  id = -2;
}

void MCAnalysisManager::getLookupParent(int trackId, int& parentId) const
{
  parentId = -2;
  if (trackId > -1 && trackId < mCurrentLookups->tracktoparent.size()) {
    parentId = mCurrentLookups->tracktoparent[trackId];
    return;
  }
}
