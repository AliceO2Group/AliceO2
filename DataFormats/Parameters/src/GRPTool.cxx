// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>
#include <string>
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "CommonUtils/FileSystemUtils.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <CommonUtils/NameConf.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <CCDB/BasicCCDBManager.h>
#include <SimConfig/SimConfig.h>
#include <unordered_map>
#include <filesystem>

//
// Created by Sandro Wenzel on 20.06.22.
//

// A utility to create/edit GRP objects for MC

enum class GRPCommand {
  kNONE,
  kCREATE,
  kANCHOR,
  kSETROMODE,
  kPRINTECS,
  kPRINTLHC,
  kPRINTMAG
};

// options struct filled from command line
struct Options {
  std::vector<std::string> readout;
  std::vector<std::string> skipreadout;
  int run;               // run number
  int orbitsPerTF = 256; // number of orbits per timeframe --> used to calculate start orbit for collisions
  GRPCommand command = GRPCommand::kNONE;
  std::string grpfilename = ""; // generic filename placeholder used by various commands
  std::vector<std::string> continuous = {};
  std::vector<std::string> triggered = {};
  bool clearRO = false;
  std::string outprefix = "";
  std::string fieldstring = "";
  std::string bcPatternFile = "";
  bool print = false; // whether to print outcome of GRP operation
  bool lhciffromccdb = false; // whether only to take GRPLHCIF from CCDB
  std::string publishto = "";
  std::string ccdbhost = "https://alice-ccdb.cern.ch";
  bool isRun5 = false; // whether or not this is supposed to be a Run5 detector configuration
};

void print_globalHelp(int argc, char* argv[])
{
  std::cout << "** A GRP utility **\n\n";
  std::cout << "Usage: " << argv[0] << " subcommand [sub-command-options]\n";
  std::cout << "\n";
  std::cout << "The following subcommands are available:\n";
  std::cout << "\t createGRPs : Create baseline GRP objects/file\n";
  std::cout << "\t anchorGRPs : Fetch GRP objects from CCDB based on run number\n";
  std::cout << "\t print_GRPECS : print a GRPECS object/file\n";
  std::cout << "\t print_GRPMAG : print a GRPMagField object/file\n";
  std::cout << "\t print_GRPLHC : print a GRPLHCIF object/file\n";
  std::cout << "\t setROMode : modify/set readoutMode in a GRPECS file\n";
  std::cout << "\n";
  std::cout << "Sub-command options can be seen with subcommand --help\n";
}

namespace
{
template <typename T>
void printGRP(std::string const& filename, std::string const& objtype)
{
  std::cout << "\nPrinting " << objtype << " from file " << filename << "\n\n";
  auto grp = T::loadFrom(filename);
  if (grp) {
    grp->print();
    delete grp;
  } else {
    std::cerr << "Error loading " << objtype << " objects from file " << filename << "\n";
  }
}
} // namespace

void printGRPECS(std::string const& filename)
{
  printGRP<o2::parameters::GRPECSObject>(filename, "GRPECS");
}

void printGRPMAG(std::string const& filename)
{
  printGRP<o2::parameters::GRPMagField>(filename, "GRPMAG");
}

void printGRPLHC(std::string const& filename)
{
  printGRP<o2::parameters::GRPLHCIFData>(filename, "GRPLHCIF");
}

void setROMode(std::string const& filename, std::vector<std::string> const& continuous,
               std::vector<std::string> const& triggered, bool clear = false)
{
  using o2::detectors::DetID;

  if (filename.size() == 0) {
    std::cout << "no filename given\n";
    return;
  }
  using GRPECSObject = o2::parameters::GRPECSObject;
  const std::string grpName{o2::base::NameConf::CCDBOBJECT};
  TFile flGRP(filename.c_str(), "update");
  if (flGRP.IsZombie()) {
    LOG(error) << "Failed to open GRPECS file " << filename << " in update mode ";
    return;
  }
  std::unique_ptr<GRPECSObject> grp(static_cast<GRPECSObject*>(flGRP.GetObjectChecked(grpName.c_str(), GRPECSObject::Class())));
  if (grp.get()) {
    // clear complete state (continuous state) first of all when asked
    if (clear) {
      for (auto id = DetID::First; id <= DetID::Last; ++id) {
        if (grp->isDetReadOut(id)) {
          grp->remDetContinuousReadOut(id);
        }
      }
    }

    //
    for (auto& detstr : continuous) {
      // convert to detID
      o2::detectors::DetID id(detstr.c_str());
      if (grp->isDetReadOut(id)) {
        grp->addDetContinuousReadOut(id);
        LOG(info) << "Setting det " << detstr << " to continuous RO mode";
      }
    }
    //
    for (auto& detstr : triggered) {
      // convert to detID
      o2::detectors::DetID id(detstr.c_str());
      if (grp->isDetReadOut(id)) {
        grp->addDetTrigger(id);
        LOG(info) << "Setting det " << detstr << " to trigger CTP";
      }
    }
    grp->print();
    flGRP.WriteObjectAny(grp.get(), grp->Class(), grpName.c_str());
  }
  flGRP.Close();
}

// copies a file idendified by filename to a CCDB snapshot starting under path
// and with the CCDBpath hierarchy
bool publish(std::string const& filename, std::string const& path, std::string CCDBpath)
{
  if (!std::filesystem::exists(filename)) {
    LOG(error) << "Input file " << filename << "does not exist\n";
    return false;
  }

  std::string targetdir = path + CCDBpath;
  try {
    o2::utils::createDirectoriesIfAbsent(targetdir);
  } catch (std::exception e) {
    LOGP(error, fmt::format("Could not create local snapshot cache directory {}, reason: {}", targetdir, e.what()));
    return false;
  }

  auto targetfile = std::filesystem::path(targetdir + "/snapshot.root");
  auto opts = std::filesystem::copy_options::overwrite_existing;
  std::filesystem::copy_file(filename, targetfile, opts);
  if (std::filesystem::exists(targetfile)) {
    LOG(info) << "file " << filename << " copied/published to " << targetfile;
  }
  return true;
}

// download a set of basic GRP files based on run number/time
bool anchor_GRPs(Options const& opts, std::vector<std::string> const& paths = {"GLO/Config/GRPECS", "GLO/Config/GRPMagField", "GLO/Config/GRPLHCIF"})
{
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  auto soreor = ccdbmgr.getRunDuration(opts.run);
  // fix the timestamp early
  uint64_t runStart = soreor.first;

  o2::ccdb::CcdbApi api;
  api.init(opts.ccdbhost);

  const bool preserve_path = true;
  const std::string filename("snapshot.root");
  std::map<std::string, std::string> filter;
  bool success = true;
  for (auto& p : paths) {
    LOG(info) << "Fetching " << p << " from CCDB";
    success &= api.retrieveBlob(p, opts.publishto, filter, runStart, preserve_path, filename);
  }
  return success;
}

// creates a set of basic GRP files (for simulation)
bool create_GRPs(Options const& opts)
{
  // some code duplication from o2-sim --> remove it

  uint64_t runStart = -1; // used in multiple GRPs

  // GRPECS
  {
    LOG(info) << " --- creating GRP ECS -----";
    o2::parameters::GRPECSObject grp;
    grp.setRun(opts.run);
    // if
    auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
    auto soreor = ccdbmgr.getRunDuration(opts.run);
    runStart = soreor.first;
    grp.setTimeStart(runStart);
    grp.setTimeEnd(runStart + 3600000);
    grp.setNHBFPerTF(opts.orbitsPerTF);
    std::vector<std::string> modules{};
    o2::conf::SimConfig::determineActiveModules(opts.readout, std::vector<std::string>(), modules, opts.isRun5);
    std::vector<std::string> readout{};
    o2::conf::SimConfig::determineReadoutDetectors(modules, std::vector<std::string>(), opts.skipreadout, readout);
    for (auto& detstr : readout) {
      o2::detectors::DetID id(detstr.c_str());
      grp.addDetReadOut(id);
      // set default RO modes
      if (!o2::parameters::GRPECSObject::alwaysTriggeredRO(id)) {
        grp.addDetContinuousReadOut(id);
      }
    }
    grp.setIsMC(true);
    grp.setRunType(o2::parameters::GRPECSObject::RunType::PHYSICS);
    // grp.setDataPeriod("mc"); // decide what to put here
    std::string grpfilename = o2::base::NameConf::getGRPECSFileName(opts.outprefix);
    TFile grpF(grpfilename.c_str(), "recreate");
    grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
    grpF.Close();
    if (opts.print) {
      grp.print();
    }
    if (opts.publishto.size() > 0) {
      publish(grpfilename, opts.publishto, "/GLO/Config/GRPECS");
    }
  }

  // GRPMagField
  {
    LOG(info) << " --- creating magfield GRP -----";
    o2::parameters::GRPMagField grp;
    // parse the wanted field value
    int fieldvalue = 0;
    o2::conf::SimFieldMode fieldmode;
    auto ok = o2::conf::SimConfig::parseFieldString(opts.fieldstring, fieldvalue, fieldmode);
    if (!ok) {
      LOG(error) << "Error parsing field string " << opts.fieldstring;
      return false;
    }

    if (fieldmode == o2::conf::SimFieldMode::kCCDB) {
      // we download the object from CCDB
      LOG(info) << "Downloading mag field directly from CCDB";
      anchor_GRPs(opts, {"GLO/Config/GRPMagField"});
    } else {
      // let's not create an actual mag field object for this
      // we only need to lookup the currents from the possible
      // values of mag field
      // +-2,+-5,0 and uniform

      const std::unordered_map<int, std::pair<int, int>> field_to_current = {{2, {12000, 6000}},
                                                                             {5, {30000, 6000}},
                                                                             {-2, {-12000, -6000}},
                                                                             {-5, {-30000, -6000}},
                                                                             {0, {0, 0}}};

      auto currents_iter = field_to_current.find(fieldvalue);
      if (currents_iter == field_to_current.end()) {
        LOG(error) << " Could not lookup currents for fieldvalue " << fieldvalue;
        return false;
      }

      o2::units::Current_t currDip = (*currents_iter).second.second;
      o2::units::Current_t currL3 = (*currents_iter).second.first;
      grp.setL3Current(currL3);
      grp.setDipoleCurrent(currDip);
      grp.setFieldUniformity(fieldmode == o2::conf::SimFieldMode::kUniform);
      if (opts.print) {
        grp.print();
      }
      std::string grpfilename = o2::base::NameConf::getGRPMagFieldFileName(opts.outprefix);
      TFile grpF(grpfilename.c_str(), "recreate");
      grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
      grpF.Close();
      if (opts.publishto.size() > 0) {
        publish(grpfilename, opts.publishto, "/GLO/Config/GRPMagField");
      }
    }
  }

  // GRPLHCIF --> complete it later
  {
    LOG(info) << " --- creating GRP LHCIF -----";
    if (opts.lhciffromccdb) { // if we take the whole object it directly from CCDB, we can just download it
      LOG(info) << "Downloading complete GRPLHCIF object directly from CCDB";
      anchor_GRPs(opts, {"GLO/Config/GRPLHCIF"});
    } else {
      o2::parameters::GRPLHCIFData grp;
      // eventually we need to set the beam info from the generator, at the moment put some plausible values
      grp.setFillNumberWithTime(runStart, 0);         // RS FIXME
      grp.setInjectionSchemeWithTime(runStart, "");   // RS FIXME
      grp.setBeamEnergyPerZWithTime(runStart, 6.8e3); // RS FIXME
      grp.setAtomicNumberB1WithTime(runStart, 1.);    // RS FIXME
      grp.setAtomicNumberB2WithTime(runStart, 1.);    // RS FIXME
      grp.setCrossingAngleWithTime(runStart, 0.);     // RS FIXME
      grp.setBeamAZ();

      // set the BC pattern if necessary
      if (opts.bcPatternFile.size() > 0) {
        // load bunch filling from the file (with standard CCDB convention)
        auto* bc = o2::BunchFilling::loadFrom(opts.bcPatternFile, "ccdb_object");
        if (!bc) {
          // if it failed, retry with default naming
          bc = o2::BunchFilling::loadFrom(opts.bcPatternFile);
        }
        if (!bc) {
          LOG(fatal) << "Failed to load bunch filling from " << opts.bcPatternFile;
        }
        grp.setBunchFillingWithTime(grp.getBeamEnergyPerZTime(), *bc); // borrow the time from the existing entry
        delete bc;
      } else {
        // we initialize with a default bunch filling scheme;
        LOG(info) << "Initializing with default bunch filling";
        o2::BunchFilling bc;
        bc.setDefault();
        grp.setBunchFillingWithTime(grp.getBeamEnergyPerZTime(), bc);
      }

      std::string grpfilename = o2::base::NameConf::getGRPLHCIFFileName(opts.outprefix);
      if (opts.print) {
        grp.print();
      }
      TFile grpF(grpfilename.c_str(), "recreate");
      grpF.WriteObjectAny(&grp, grp.Class(), o2::base::NameConf::CCDBOBJECT.data());
      grpF.Close();
      if (opts.publishto.size() > 0) {
        publish(grpfilename, opts.publishto, "/GLO/Config/GRPLHCIF");
      }
    }
  }

  return true;
}

void perform_Command(Options const& opts)
{
  switch (opts.command) {
    case GRPCommand::kCREATE: {
      create_GRPs(opts);
      break;
    }
    case GRPCommand::kANCHOR: {
      anchor_GRPs(opts);
      break;
    }
    case GRPCommand::kPRINTECS: {
      printGRPECS(opts.grpfilename);
      break;
    }
    case GRPCommand::kPRINTMAG: {
      printGRPMAG(opts.grpfilename);
      break;
    }
    case GRPCommand::kPRINTLHC: {
      printGRPLHC(opts.grpfilename);
      break;
    }
    case GRPCommand::kSETROMODE: {
      setROMode(opts.grpfilename, opts.continuous, opts.triggered, opts.clearRO);
      break;
    }
    default: {
    }
  }
}

bool parseOptions(int argc, char* argv[], Options& optvalues)
{
  namespace bpo = boost::program_options;
  bpo::options_description global("Global options");
  global.add_options()("command", bpo::value<std::string>(), "command to execute")("subargs", bpo::value<std::vector<std::string>>(), "Arguments for command");
  global.add_options()("help,h", "Produce help message.");

  bpo::positional_options_description pos;
  pos.add("command", 1).add("subargs", -1);

  bpo::variables_map vm;
  bpo::parsed_options parsed{nullptr};
  try {
    parsed = bpo::command_line_parser(argc, argv).options(global).positional(pos).allow_unregistered().run();

    bpo::store(parsed, vm);

    // help
    if (vm.count("help") > 0 && vm.count("command") == 0) {
      print_globalHelp(argc, argv);
      return false;
    }
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing global options; Available options:\n";
    std::cerr << global << std::endl;
    return false;
  }

  auto subparse = [&parsed](auto& desc, auto& vm, std::string const& command_name) {
    try {
      // Collect all the unrecognized options from the first pass. This will include the
      // (positional) command name, so we need to erase that.
      std::vector<std::string> opts = bpo::collect_unrecognized(parsed.options, bpo::include_positional);
      // opts.erase(opts.begin());

      // Parse again... and store to vm
      bpo::store(bpo::command_line_parser(opts).options(desc).run(), vm);
      bpo::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
      }
    } catch (const bpo::error& e) {
      std::cerr << e.what() << "\n\n";
      std::cerr << "Error parsing options for " << command_name << " Available options:\n";
      std::cerr << desc << std::endl;
      return false;
    }
    return true;
  };

  std::string cmd = vm["command"].as<std::string>();

  if (cmd == "anchorGRPs") {
    optvalues.command = GRPCommand::kANCHOR;
    // ls command has the following options:
    bpo::options_description desc("anchor GRP options");

    // ls command has the following options:
    desc.add_options()("run", bpo::value<int>(&optvalues.run)->default_value(-1), "Run number");
    desc.add_options()("print", "print resulting GRPs");
    desc.add_options()("publishto", bpo::value<std::string>(&optvalues.publishto)->default_value("GRP"), "Base path under which GRP objects should be published on disc. This path can serve as lookup for CCDB queries of the GRP objects.");
    if (!subparse(desc, vm, "anchorGRPs")) {
      return false;
    }
    if (vm.count("print") > 0) {
      optvalues.print = true;
    }
  } else if (cmd == "createGRPs") {
    optvalues.command = GRPCommand::kCREATE;

    // ls command has the following options:
    bpo::options_description desc("create options");
    desc.add_options()("readoutDets", bpo::value<std::vector<std::string>>(&optvalues.readout)->multitoken()->default_value(std::vector<std::string>({"all"}), "all Run3 detectors"), "Detector list to be readout/active");
    desc.add_options()("skipReadout", bpo::value<std::vector<std::string>>(&optvalues.skipreadout)->multitoken()->default_value(std::vector<std::string>(), "nothing skipped"), "list of inactive detectors (precendence over --readout)");
    desc.add_options()("run", bpo::value<int>(&optvalues.run)->default_value(-1), "Run number");
    desc.add_options()("hbfpertf", bpo::value<int>(&optvalues.orbitsPerTF)->default_value(128), "heart beat frames per timeframe (timeframelength)");
    desc.add_options()("field", bpo::value<std::string>(&optvalues.fieldstring)->default_value("-5"), "L3 field rounded to kGauss, allowed values +-2,+-5 and 0; +-<intKGaus>U for uniform field");
    desc.add_options()("outprefix,o", bpo::value<std::string>(&optvalues.outprefix)->default_value("o2sim"), "Prefix for GRP output files");
    desc.add_options()("bcPatternFile", bpo::value<std::string>(&optvalues.bcPatternFile)->default_value(""), "Interacting BC pattern file (e.g. from CreateBCPattern.C)");
    desc.add_options()("lhcif-CCDB", "take GRPLHCIF directly from CCDB");
    desc.add_options()("print", "print resulting GRPs");
    desc.add_options()("publishto", bpo::value<std::string>(&optvalues.publishto)->default_value(""), "Base path under which GRP objects should be published on disc. This path can serve as lookup for CCDB queries of the GRP objects.");
    desc.add_options()("isRun5", bpo::bool_switch(&optvalues.isRun5), "Whether or not to expect a Run5 detector configuration.");
    if (!subparse(desc, vm, "createGRPs")) {
      return false;
    }
    if (vm.count("print") > 0) {
      optvalues.print = true;
    }
    if (vm.count("lhcif-CCDB") > 0) {
      optvalues.lhciffromccdb = true;
    }
  } else if (cmd == "setROMode") {
    // set/modify the ROMode
    optvalues.command = GRPCommand::kSETROMODE;
    bpo::options_description desc("setting detector readout modes");
    desc.add_options()("file,f", bpo::value<std::string>(&optvalues.grpfilename)->default_value("o2sim_grpecs.root"), "Path to GRPECS file");
    desc.add_options()("continuousRO", bpo::value<std::vector<std::string>>(&optvalues.continuous)->multitoken()->default_value(std::vector<std::string>({"all"}), "all active detectors"), "List of detectors to set to continuous mode");
    desc.add_options()("triggerCTP", bpo::value<std::vector<std::string>>(&optvalues.triggered)->multitoken()->default_value(std::vector<std::string>({""}), "none"), "List of detectors to trigger CTP");
    desc.add_options()("clear", "clears all RO modes (prio to applying other options)");
    if (!subparse(desc, vm, "setROMode")) {
      return false;
    }
    if (vm.count("clear") > 0) {
      optvalues.clearRO = true;
    }
  } else if (cmd == "print_GRPECS") {
    optvalues.command = GRPCommand::kPRINTECS;
    // print the GRP
    bpo::options_description desc("print options");
    desc.add_options()("file,f", bpo::value<std::string>(&optvalues.grpfilename), "Path to GRP file");
    if (!subparse(desc, vm, "print_GRPECS")) {
      return false;
    }
  } else if (cmd == "print_GRPLHC") {
    optvalues.command = GRPCommand::kPRINTLHC;
    // print the GRP
    bpo::options_description desc("print options");
    desc.add_options()("file,f", bpo::value<std::string>(&optvalues.grpfilename), "Path to GRP file");
    if (!subparse(desc, vm, "print_GRPECS")) {
      return false;
    }
  } else if (cmd == "print_GRPMAG") {
    optvalues.command = GRPCommand::kPRINTMAG;
    // print the GRP
    bpo::options_description desc("print options");
    desc.add_options()("file,f", bpo::value<std::string>(&optvalues.grpfilename), "Path to GRP file");
    if (!subparse(desc, vm, "print_GRPECS")) {
      return false;
    }
  } else {
    std::cerr << "Error: Unknown command " << cmd << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char* argv[])
{
  Options options;
  if (parseOptions(argc, argv, options)) {
    perform_Command(options);
  } else {
    std::cout << "Parse options failed\n";
    return 1;
  }
  return 0;
}
