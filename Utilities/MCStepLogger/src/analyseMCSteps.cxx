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
#include <string>
#include <vector>
#include <functional>

#include <boost/program_options.hpp>

#include "TROOT.h"
#include "TInterpreter.h"
#include "TSystemDirectory.h"

#include "FairLogger.h"

#include "MCStepLogger/MCAnalysisManager.h"
#include "MCStepLogger/MCAnalysisFileWrapper.h"
#include "MCStepLogger/BasicMCAnalysis.h"

using namespace o2::mcstepanalysis;

namespace bpo = boost::program_options;

std::vector<std::string> availableCommands = { "analyze", "checkFile" };

// print help message
void helpMessage(const bpo::options_description& desc)
{
  std::cout << desc << std::endl;
}
/// pass macros in analysisDir to ROOT interpreter
void registerAnalyses(const std::string& analysisDir)
{
  // try to read analyses from analyis directory
  // \todo try-catch to only read analysis macros and skip other files or non-conforming
  //       analysis classes ==> print warnings/errors
  TSystemDirectory macroDir(analysisDir.c_str(), analysisDir.c_str());
  if (macroDir.IsDirectory()) {
    TList* macros = macroDir.GetListOfFiles();
    TIter next(macros);
    TSystemFile* file = nullptr;
    while ((file = (TSystemFile*)next())) {
      if (!file->IsDirectory()) {
        const std::string& filepath = analysisDir + "/" + file->GetName();
        LOG(DEBUG) << "Try to load analysis from " << filepath << "\n";
        gROOT->LoadMacro(filepath.c_str());
        gInterpreter->ProcessLine("declareAnalysis()");
      }
    }
  }
}
/// analyze function
int analyze(const bpo::variables_map& vm, std::string& errorMessage)
{
  //////////////////////////////////////////////////////////////////////////////////////////////
  // if external analyses are desired but no analysis directory is passed, fail
  if (vm.count("analyses") && !vm.count("analysis-dir") && !vm.count("list-analyses")) {
    errorMessage += "Analysis names but no analysis directory passed.\n";
  }
  //////////////////////////////////////////////////////////////////////////////////////////////
  // now the check for ROOT files
  if (!vm.count("root-file") && !vm.count("list-analyses")) {
    errorMessage += "Need ROOT file from MCStepLogger.\n";
  }
  //////////////////////////////////////////////////////////////////////////////////////////////
  // is there an output directory?
  if (!vm.count("output-dir") && !vm.count("list-analyses")) {
    errorMessage += "Need an output directory.\n";
  }
  //////////////////////////////////////////////////////////////////////////////////////////////
  // a label needs to be given to identify this analysis run
  if (!vm.count("label") && !vm.count("list-analyses")) {
    errorMessage += "Need a label for this analysis.\n";
  }
  if (!errorMessage.empty()) {
    return 1;
  }
  //////////////////////////////////////////////////////////////////////////////////////////////
  // try to read analyses from analyis directory
  // \todo try-catch to only read analysis macros and skip other files or non-conforming
  //       analysis classes ==> print warnings/errors
  if (vm.count("analysis-dir") && vm.count("analyses")) {
    const std::string analysisDir = vm["analysis-dir"].as<std::string>().c_str();
    registerAnalyses(analysisDir);
  }
  // create basic analysis by default, is registered automaticallyt to AnalysisManager
  new BasicMCAnalysis();
  //////////////////////////////////////////////////////////////////////////////////////////////
  // List analyses and quit. This has to be done here after analyses are read from analysis-dir
  // the first time the AnalysisManager is needed
  auto& anamgr = MCAnalysisManager::Instance();
  if (vm.count("list-analyses")) {
    anamgr.printAnalyses();
    return 0;
  }
  //////////////////////////////////////////////////////////////////////////////////////////////
  // set a label and the input file from MCStepLogger
  anamgr.setLabel(vm["label"].as<std::string>());
  anamgr.setInputFilepath(vm["root-file"].as<std::string>());
  // if ready, run
  if (!anamgr.checkReadiness()) {
    return 1;
  }
  anamgr.run(vm["number-events"].as<int>());
  // save what the AnalysisFileHandler has gotten during the analysis run
  anamgr.write(vm["output-dir"].as<std::string>());
  return 0;
}

// check type and sanity of a file
// \note this is a beta version which will always complain unless an analysis file was passed.
//       In that case, the meta info is printed
int checkFile(const bpo::variables_map& vm, std::string& errorMessage)
{
  if (!vm.count("root-file")) {
    errorMessage += "ROOT file required.\n";
  }
  if (!errorMessage.empty()) {
    return 1;
  }
  LOG(INFO) << "Check type and sanity of the input file " << vm["root-file"].as<std::string>();
  MCAnalysisFileWrapper fileWrapper;
  if (fileWrapper.read(vm["root-file"].as<std::string>())) {
    fileWrapper.printAnalysisMetaInfo();
    fileWrapper.printHistogramInfo();
    return 0;
  }
  auto& anamgr = MCAnalysisManager::Instance();
  anamgr.setInputFilepath(vm["root-file"].as<std::string>());
  if (anamgr.dryrun()) {
    return 0;
  }
  LOG(INFO) << "This ROOT file is neither an MCStepLogger nor an MCAnalysis file.";
  return 1;
}

// Initialize everything for the final run depending on the command
void initializeForRun(const std::string& cmd, bpo::options_description& cmdOptionsDescriptions, std::function<int(const bpo::variables_map&, std::string&)>& cmdFunction)
{
  if (cmd == "analyze") {
    cmdOptionsDescriptions.add_options()("help,h", "show this help message and exit")("analyses,a", bpo::value<std::vector<std::string>>()->multitoken(), "analyses to be run")("analysis-dir,d", bpo::value<std::string>(), "directory containing analysis macros (required, if --analyses is used)")("list-analyses,s", "list available analyses and exit")("root-file,f", bpo::value<std::string>(), "ROOT file from MCStepLogger to be analysed (required)")("label,l", bpo::value<std::string>(), "custom label for the analysis (required)")("output-dir,o", bpo::value<std::string>(), "output directory for analyses (required)")("number-events,n", bpo::value<int>()->default_value(-1), "only analyse a certain number of events");
    cmdFunction = analyze;
  } else if (cmd == "checkFile") {
    cmdOptionsDescriptions.add_options()("help,h", "show this help message and exit")("root-file,f", bpo::value<std::string>(), "ROOT file to be checked");
    cmdFunction = checkFile;
  }
}

int main(int argc, char* argv[])
{
  // Global variables map mapping the command line options to their values
  bpo::variables_map vm;
  // Description of the available top-level commands/options
  bpo::options_description desc("Available commands/options");
  desc.add_options()("help,h", "show this help message and exit")("command", bpo::value<std::string>(), "command to be executed (\"analyze\", \"checkFile\"")("positional", bpo::value<std::vector<std::string>>(), "positional arguments");
  // Dedicated description for positional arguments
  bpo::positional_options_description pos;
  // First positional argument is actually the command, all others are real positional arguments "( "positional", -1 )"
  pos.add("command", 1).add("positional", -1);
  // Parse and store in variables map, allow unregistered since these are not yet specified but will be so as soon as
  // the command name is checked
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv).options(desc).positional(pos).allow_unregistered().run();
  bpo::store(parsed, vm);

  // check whether 'command' is missing or if there is even no option at all; show global help message in that case
  // and return
  if (!vm.count("command") || vm.empty()) {
    if (!vm.count("help")) {
      LOG(FATAL) << "Need command to execute.";
    }
    // Print global help message in any case...
    helpMessage(desc);
    // ...but check whther it was explicitly asked for it and return accordingly
    return (!vm.count("help"));
  }

  // Extract command
  const std::string& cmd = vm["command"].as<std::string>();
  if (std::find(availableCommands.begin(), availableCommands.end(), cmd) == availableCommands.end()) {
    LOG(ERROR) << "Command \"" << cmd << "\" unknown.";
    helpMessage(desc);
    return 1;
  }

  // extract all positional arguments passed which are unrecognized by desc
  std::vector<std::string> opts = bpo::collect_unrecognized(parsed.options, bpo::include_positional);
  // since 'command' as positional argument is unknown, it needs to be erased explicitly
  opts.erase(opts.begin());

  // Preprocessing done.
  // Track return value
  int returnValue = 0;
  // function to be run
  std::function<int(const bpo::variables_map&, std::string&)> cmdFunction;
  // corresponding options with description
  bpo::options_description cmdOptionsDescriptions;
  // initialize command function and corresponding options description
  initializeForRun(cmd, cmdOptionsDescriptions, cmdFunction);
  // Parse again to get positional arguments correctly labelled for analysis run
  bpo::store(bpo::command_line_parser(opts).options(cmdOptionsDescriptions).run(), vm);
  // Notify and fix variables map
  bpo::notify(vm);
  // To collect error messsages
  std::string errorMessage("");
  if (vm.size() == 1) {
    errorMessage += "Options required...\n";
    returnValue = 1;
  }
  // check if specific help message is desired
  else if (!vm.count("help")) {
    returnValue = cmdFunction(vm, errorMessage);
  }
  // check return value
  if (returnValue > 0) {
    LOG(ERROR) << "Errors occured:";
    std::cerr << errorMessage << "\n";
  }
  if (returnValue > 0 || vm.count("help")) {
    helpMessage(cmdOptionsDescriptions);
  }
  return returnValue;
}
