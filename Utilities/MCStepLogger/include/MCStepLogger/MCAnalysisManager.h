// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Singleton class handling/steering single analyses and their lifecycle
 * 1. registration
 *    -> each analysis is automatically registered
 *    -> the Analysis base class provides registration during construction so there is no
 *       stand-alone analysis
 * 2. bookkeeping of analyses objects (histograms and meta info)
 *    -> an AnalysisFileHandler does the Bookkeeping of all objects produced during the analyses
 * 3. initialize
 *    -> steering all registered analyses to initialize their analysis objects
 * 4. analyze
 *    -> forwarding the steps and magnetic field calls per event to registered analyses
 * 5. finalize
 *    -> steering all analyses to finalize its objects
 */
#ifndef MCANALYSIS_MANAGER_H_
#define MCANALYSIS_MANAGER_H_

#include <string>
#include <vector>

#include "MCStepLogger/StepInfo.h"
#include "MCStepLogger/MetaInfo.h"

namespace o2
{
namespace mcstepanalysis
{

class MCAnalysis;
class MCAnalysisFileWrapper;

class MCAnalysisManager
{
 public:
  // get singleton instance
  static MCAnalysisManager& Instance()
  {
    static MCAnalysisManager inst;
    return inst;
  }
  //
  // steering from the outside world
  //
  /// check whether everything is fine before the run
  bool checkReadiness() const;
  /// run tha chain depending on the mode
  void run(int nEvents = -1);
  /// do a dryrun just to see what's in the MCStepLogger ROOT file
  bool dryrun();
  /// write produced analysis data to disk
  void write(const std::string& directory) const;
  /// terminate, reset everything
  void terminate();
  //
  // setting
  //
  /// set the path to the MCStepLogger input file path
  void setInputFilepath(const std::string& filepath);
  // register analysis to manager, done implicitly in the base Analysis class during construction
  void registerAnalysis(MCAnalysis* analysis);
  /// label for an analysis run (e.g. 'GEANT4_allModules')
  void setLabel(const std::string& label);
  /// name of the TTree of the MCStepLogger output
  void setStepLoggerTreename(const std::string& treename);
  //
  // getting
  //
  /// get current event number
  int getEventNumber() const;
  // volume name by id
  void getLookupVolName(int volId, std::string& name) const;
  /// module name by volume ID
  void getLookupModName(int volId, std::string& name) const;
  /// medium name by volume ID
  void getLookupMedName(int volId, std::string& name) const;
  /// PDG ID by track ID
  void getLookupPDG(int trackId, int& id) const;
  /// parent track ID by track ID
  void getLookupParent(int trackId, int& parentId) const;
  //
  // verbosity
  //
  /// print registered analyses
  void printAnalyses() const;

 private:
  // don't allow uncontrolled construction of AnalysisManager objects
  MCAnalysisManager() = default;
  MCAnalysisManager operator=(const MCAnalysisManager&) = delete;
  MCAnalysisManager(const MCAnalysisManager&) = delete;
  //
  // running the machinery
  //
  /// initialize AnalysisManager and registered analyses
  void initialize();
  /// analyse events and forward vectors of step and magnetic field info to single analyses
  bool analyze(int nEvents = -1, bool isDryrun = false);
  /// finalize all analyses
  void finalize();

 private:
  /// holding the pointers to registered analyses
  std::vector<MCAnalysis*> mAnalyses;
  /// collect pointers which will be safely deleted which happens e.g. when the same analysis is registered twice
  std::vector<MCAnalysis*> mAnalysesToDump;
  /// keep track of status of MCAnalysisManager
  bool mIsInitialized = false;
  bool mIsAnalyzed = false;
  /// the input file the analysis is conducted on
  std::string mInputFilepath = "";
  /// treename of step log data
  std::string mAnalysisTreename = defaults::defaultStepLoggerTTreeName;
  /// label for analyses, this is the same for all analyses since it depends on the simulation run and not on a specific analysis
  std::string mLabel = defaults::defaultLabel;
  /// keep track of which event is currently analysed
  int mCurrentEventNumber = 0;
  /// count overall number of steps
  long mNSteps = 0;
  // holding current step and magnetic field information of current event
  /// information of single steps
  std::vector<o2::StepInfo>* mCurrentStepInfo = nullptr;
  /// information of magnetic field calls
  std::vector<o2::MagCallInfo>* mCurrentMagCallInfo = nullptr;
  /// some lookups to map IDs to names
  o2::StepLookups* mCurrentLookups = nullptr;
  /// analysis files histograms are written to
  std::vector<MCAnalysisFileWrapper> mAnalysisFiles;

  ClassDefNV(MCAnalysisManager, 1);
};
} // end namespace mcstepanalysis
} // end namespace o2
#endif /* MCANALYSISMANAGER_H_ */
