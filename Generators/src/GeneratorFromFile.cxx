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

#include "Generators/GeneratorFromFile.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <fairlogger/Logger.h>
#include <FairPrimaryGenerator.h>
#include <TBranch.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TMCProcess.h>
#include <TParticle.h>
#include <TTree.h>
#include <sstream>
#include <filesystem>
#include <TGrid.h>
#include <TSystem.h>

namespace o2
{
namespace eventgen
{
GeneratorFromFile::GeneratorFromFile(const char* name)
{
  mEventFile = TFile::Open(name);
  if (mEventFile == nullptr) {
    LOG(fatal) << "EventFile " << name << " not found \n";
    return;
  }
  // the kinematics will be stored inside a Tree "TreeK" with branch "Particles"
  // different events are stored inside TDirectories

  // we need to probe for the number of events
  TObject* object = nullptr;
  do {
    std::stringstream eventstringstr;
    eventstringstr << "Event" << mEventsAvailable;
    // std::cout << "probing for " << eventstring << "\n";
    object = mEventFile->Get(eventstringstr.str().c_str());
    // std::cout << "got " << object << "\n";
    if (object != nullptr) {
      mEventsAvailable++;
    }
  } while (object != nullptr);
  LOG(info) << "Found " << mEventsAvailable << " events in this file \n";
}

void GeneratorFromFile::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    LOG(error) << "start event bigger than available events\n";
  }
}

bool GeneratorFromFile::rejectOrFixKinematics(TParticle& p)
{
  // avoid compute if the particle is not known in the PDG database
  if (!p.GetPDG()) {
    LOG(warn) << "Particle with pdg " << p.GetPdgCode() << " not known in DB (not fixing mass)";
    // still returning true here ... primary will be flagged as non-trackable by primary event generator
    return true;
  }

  const auto nominalmass = p.GetMass();
  auto mom2 = p.Px() * p.Px() + p.Py() * p.Py() + p.Pz() * p.Pz();
  auto calculatedmass = p.Energy() * p.Energy() - mom2;
  calculatedmass = (calculatedmass >= 0.) ? std::sqrt(calculatedmass) : -std::sqrt(-calculatedmass);
  const double tol = 1.E-4;
  auto difference = std::abs(nominalmass - calculatedmass);
  if (std::abs(nominalmass - calculatedmass) > tol) {
    const auto asgmass = p.GetCalcMass();
    bool fix = mFixOffShell && std::abs(nominalmass - asgmass) < tol;
    LOG(warn) << "Particle " << p.GetPdgCode() << " has off-shell mass: M_PDG= " << nominalmass << " (assigned= " << asgmass
              << ") calculated= " << calculatedmass << " -> diff= " << difference << " | " << (fix ? "fixing" : "skipping");
    if (fix) {
      double e = std::sqrt(nominalmass * nominalmass + mom2);
      p.SetMomentum(p.Px(), p.Py(), p.Pz(), e);
      p.SetCalcMass(nominalmass);
    } else {
      return false;
    }
  }
  return true;
}

Bool_t GeneratorFromFile::ReadEvent(FairPrimaryGenerator* primGen)
{
  if (mEventCounter < mEventsAvailable) {
    int particlecounter = 0;

    // get the tree and the branch
    std::stringstream treestringstr;
    treestringstr << "Event" << mEventCounter << "/TreeK";
    TTree* tree = (TTree*)mEventFile->Get(treestringstr.str().c_str());
    if (tree == nullptr) {
      return kFALSE;
    }

    auto branch = tree->GetBranch("Particles");
    TParticle* particle = nullptr;
    branch->SetAddress(&particle);
    LOG(info) << "Reading " << branch->GetEntries() << " particles from Kinematics file";

    // read the whole kinematics initially
    std::vector<TParticle> particles;
    for (int i = 0; i < branch->GetEntries(); ++i) {
      branch->GetEntry(i);
      particles.push_back(*particle);
    }

    // filter the particles from Kinematics.root originally put by a generator
    // and which are trackable
    auto isFirstTrackableDescendant = [](TParticle const& p) {
      const int kTransportBit = BIT(14);
      // The particle should have not set kDone bit and its status should not exceed 1
      if ((p.GetUniqueID() > 0 && p.GetUniqueID() != kPNoProcess) || !p.TestBit(kTransportBit)) {
        return false;
      }
      return true;
    };

    for (int i = 0; i < branch->GetEntries(); ++i) {
      auto& p = particles[i];
      if (!isFirstTrackableDescendant(p)) {
        continue;
      }

      bool wanttracking = true; // RS as far as I understand, if it reached this point, it is trackable
      if (wanttracking || !mSkipNonTrackable) {
        if (!rejectOrFixKinematics(p)) {
          continue;
        }
        auto pdgid = p.GetPdgCode();
        auto px = p.Px();
        auto py = p.Py();
        auto pz = p.Pz();
        auto vx = p.Vx();
        auto vy = p.Vy();
        auto vz = p.Vz();
        auto parent = -1;
        auto e = p.Energy();
        auto tof = p.T();
        auto weight = p.GetWeight();
        LOG(debug) << "Putting primary " << pdgid << " " << p.GetStatusCode() << " " << p.GetUniqueID();
        primGen->AddTrack(pdgid, px, py, pz, vx, vy, vz, parent, wanttracking, e, tof, weight);
        particlecounter++;
      }
    }
    mEventCounter++;

    LOG(info) << "Event generator put " << particlecounter << " on stack";
    return kTRUE;
  } else {
    LOG(error) << "GeneratorFromFile: Ran out of events\n";
  }
  return kFALSE;
}

// based on O2 kinematics

GeneratorFromO2Kine::GeneratorFromO2Kine(const char* name)
{
  // this generator should leave all dimensions the same as in the incoming kinematics file
  setMomentumUnit(1.);
  setEnergyUnit(1.);
  setPositionUnit(1.);
  setTimeUnit(1.);

  if (strncmp(name, "alien:/", 7) == 0 && !gGrid) {
    TGrid::Connect("alien:");
    if (!gGrid) {
      LOG(fatal) << "Could not connect to alien, did you check the alien token?";
      return;
    }
  }
  mEventFile = TFile::Open(name);
  if (mEventFile == nullptr) {
    LOG(fatal) << "EventFile " << name << " not found";
    return;
  }
  // the kinematics will be stored inside a branch MCTrack
  // different events are stored inside different entries
  auto tree = (TTree*)mEventFile->Get("o2sim");
  if (tree) {
    mEventBranch = tree->GetBranch("MCTrack");
    if (mEventBranch) {
      mEventsAvailable = mEventBranch->GetEntries();
      LOG(info) << "Found " << mEventsAvailable << " events in this file";
    }
    mMCHeaderBranch = tree->GetBranch("MCEventHeader.");
    if (mMCHeaderBranch) {
      LOG(info) << "Found " << mMCHeaderBranch->GetEntries() << " event-headers";
    } else {
      LOG(warn) << "No MCEventHeader branch found in kinematics input file";
    }
    return;
  }
  LOG(error) << "Problem reading events from file " << name;
}

GeneratorFromO2Kine::GeneratorFromO2Kine(O2KineGenConfig const& pars) : GeneratorFromO2Kine(pars.fileName.c_str())
{
  mConfig = std::make_unique<O2KineGenConfig>(pars);
}

bool GeneratorFromO2Kine::Init()
{

  // read and set params

  LOG(info) << "Init \'FromO2Kine\' generator";
  mSkipNonTrackable = mConfig->skipNonTrackable;
  mContinueMode = mConfig->continueMode;
  mRoundRobin = mConfig->roundRobin;
  mRandomize = mConfig->randomize;
  mRngSeed = mConfig->rngseed;
  mRandomPhi = mConfig->randomphi;
  if (mRandomize) {
    gRandom->SetSeed(mRngSeed);
  }

  return true;
}

void GeneratorFromO2Kine::SetStartEvent(int start)
{
  if (start < mEventsAvailable) {
    mEventCounter = start;
  } else {
    LOG(error) << "start event bigger than available events\n";
  }
}

bool GeneratorFromO2Kine::importParticles()
{
  // NOTE: This should be usable with kinematics files without secondaries
  // It might need some adjustment to make it work with secondaries or to continue
  // from a kinematics snapshot

  // Randomize the order of events in the input file
  if (mRandomize) {
    mEventCounter = gRandom->Integer(mEventsAvailable);
    LOG(info) << "GeneratorFromO2Kine - Picking event " << mEventCounter;
  }

  double dPhi = 0.;
  // Phi rotation
  if (mRandomPhi) {
    dPhi = gRandom->Uniform(2 * TMath::Pi());
    LOG(info) << "Rotating phi by " << dPhi;
  }

  if (mEventCounter < mEventsAvailable) {
    int particlecounter = 0;

    std::vector<o2::MCTrack>* tracks = nullptr;
    mEventBranch->SetAddress(&tracks);
    mEventBranch->GetEntry(mEventCounter);

    if (mMCHeaderBranch) {
      o2::dataformats::MCEventHeader* mcheader = nullptr;
      mMCHeaderBranch->SetAddress(&mcheader);
      mMCHeaderBranch->GetEntry(mEventCounter);
      mOrigMCEventHeader.reset(mcheader);
    }

    for (auto& t : *tracks) {

      // in case we do not want to continue, take only primaries
      if (!mContinueMode && !t.isPrimary()) {
        continue;
      }

      auto pdg = t.GetPdgCode();
      auto px = t.Px();
      auto py = t.Py();
      if (mRandomPhi) {
        // transformation applied through rotation matrix
        auto cos = TMath::Cos(dPhi);
        auto sin = TMath::Sin(dPhi);
        auto newPx = px * cos - py * sin;
        auto newPy = px * sin + py * cos;
        px = newPx;
        py = newPy;
      }
      auto pz = t.Pz();
      auto vx = t.Vx();
      auto vy = t.Vy();
      auto vz = t.Vz();
      auto m1 = t.getMotherTrackId();
      auto m2 = t.getSecondMotherTrackId();
      auto d1 = t.getFirstDaughterTrackId();
      auto d2 = t.getLastDaughterTrackId();
      auto e = t.GetEnergy();
      auto vt = t.T() * 1e-9; // MCTrack stores in ns ... generators and engines use seconds
      auto weight = t.getWeight();
      auto wanttracking = t.getToBeDone();

      if (mContinueMode) { // in case we want to continue, do only inhibited tracks
        wanttracking &= t.getInhibited();
      }

      LOG(debug) << "Putting primary " << pdg;

      mParticles.push_back(TParticle(pdg, t.getStatusCode().fullEncoding, m1, m2, d1, d2, px, py, pz, e, vx, vy, vz, vt));
      mParticles.back().SetUniqueID((unsigned int)t.getProcess()); // we should propagate the process ID
      mParticles.back().SetBit(ParticleStatus::kToBeDone, wanttracking);
      mParticles.back().SetWeight(weight);

      particlecounter++;
    }
    mEventCounter++;
    if (mRoundRobin) {
      LOG(info) << "Resetting event counter to 0; Reusing events from file";
      mEventCounter = mEventCounter % mEventsAvailable;
    }

    if (tracks) {
      delete tracks;
    }

    LOG(info) << "Event generator put " << particlecounter << " on stack";
    return true;
  } else {
    LOG(error) << "GeneratorFromO2Kine: Ran out of events\n";
  }
  return false;
}

void GeneratorFromO2Kine::updateHeader(o2::dataformats::MCEventHeader* eventHeader)
{
  /** update header **/

  // we forward the original header information if any
  if (mOrigMCEventHeader.get()) {
    eventHeader->copyInfoFrom(*mOrigMCEventHeader.get());
  }
  // we forward also the original basic vertex information contained in FairMCEventHeader
  static_cast<FairMCEventHeader&>(*eventHeader) = static_cast<FairMCEventHeader&>(*mOrigMCEventHeader.get());

  // put additional information about input file and event number of the current event
  eventHeader->putInfo<std::string>("forwarding-generator", "generatorFromO2Kine");
  eventHeader->putInfo<std::string>("forwarding-generator_inputFile", mEventFile->GetName());
  eventHeader->putInfo<int>("forwarding-generator_inputEventNumber", mEventCounter - 1);
}

namespace
{
// some helper to execute a command and capture it's output in a vector
std::vector<std::string> executeCommand(const std::string& command)
{
  std::vector<std::string> result;
  std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("Failed to open pipe");
  }

  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
    std::string line(buffer);
    // Remove trailing newline character, if any
    if (!line.empty() && line.back() == '\n') {
      line.pop_back();
    }
    result.push_back(line);
  }
  return result;
}
} // namespace

GeneratorFromEventPool::GeneratorFromEventPool(EventPoolGenConfig const& pars) : mConfig{pars}
{
}

bool GeneratorFromEventPool::Init()
{
  // this simply passes tracks trough. Leave units intact.
  setTimeUnit(1.);
  setPositionUnit(1.);
  setEnergyUnit(1.);

  // initialize the event pool
  if (mConfig.rngseed > 0) {
    mRandomEngine.seed(mConfig.rngseed);
  } else {
    std::random_device rd;
    mRandomEngine.seed(rd());
  }
  TString expPath(mConfig.eventPoolPath);
  gSystem->ExpandPathName(expPath);
  mPoolFilesAvailable = setupFileUniverse(expPath.Data());

  if (mPoolFilesAvailable.size() == 0) {
    LOG(error) << "No file found that can be used with EventPool generator";
    return false;
  }
  LOG(info) << "Found " << mPoolFilesAvailable.size() << " available event pool files";

  // now choose the actual file
  std::uniform_int_distribution<int> distribution(0, mPoolFilesAvailable.size() - 1);
  auto chosenIndex = distribution(mRandomEngine);
  mFileChosen = mPoolFilesAvailable[chosenIndex];
  LOG(info) << "EventPool is using file " << mFileChosen;

  // we bring up the internal mO2KineGenerator
  auto kine_config = O2KineGenConfig{
    .skipNonTrackable = mConfig.skipNonTrackable,
    .continueMode = false,
    .roundRobin = false,
    .randomize = mConfig.randomize,
    .rngseed = mConfig.rngseed,
    .randomphi = mConfig.randomphi,
    .fileName = mFileChosen};
  mO2KineGenerator.reset(new GeneratorFromO2Kine(kine_config));
  return mO2KineGenerator->Init();
}

namespace
{
namespace fs = std::filesystem;
// checks a single file name
bool checkFileName(std::string const& pathStr)
{
  // LOG(info) << "Checking filename " << pathStr;
  try {
    // Remove optional protocol prefix "alien://"
    const std::string protocol = "alien://";
    std::string finalPathStr(pathStr);
    if (pathStr.starts_with(protocol)) {
      finalPathStr = pathStr.substr(protocol.size());
    }
    fs::path path(finalPathStr);

    // Check if the filename is "evtpool.root"
    return path.filename() == GeneratorFromEventPool::eventpool_filename;
  } catch (const fs::filesystem_error& e) {
    // Invalid path syntax will throw an exception
    std::cerr << "Filesystem error: " << e.what() << '\n';
    return false;
  } catch (...) {
    // Catch-all for other potential exceptions
    std::cerr << "An unknown error occurred while checking the path.\n";
    return false;
  }
}

// checks a whole universe of file names
bool checkFileUniverse(std::vector<std::string> const& universe)
{
  if (universe.size() == 0) {
    return false;
  }
  for (auto& fn : universe) {
    if (!checkFileName(fn)) {
      return false;
    }
  }
  // TODO: also check for a common path structure with maximally 00X as only difference

  return true;
}

std::vector<std::string> readLines(const std::string& filePath)
{
  std::vector<std::string> lines;

  // Check if the file is a valid text file
  fs::path path(filePath);

  // Open the file
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::ios_base::failure("Failed to open the file.");
  }

  // Read up to n lines
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  return lines;
}

// Function to find all files named eventpool_filename under a given path
std::vector<std::string> getLocalFileList(const fs::path& rootPath)
{
  std::vector<std::string> result;

  // Ensure the root path exists and is a directory
  if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) {
    throw std::invalid_argument("The provided path is not a valid directory.");
  }

  // Iterate over the directory and subdirectories
  for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
    if (entry.is_regular_file() && entry.path().filename() == GeneratorFromEventPool::eventpool_filename) {
      result.push_back(entry.path().string());
    }
  }
  return result;
}

} // end anonymous namespace

/// A function determining the universe of event pool files, as determined by the path string
/// returns empty vector if it fails
std::vector<std::string> GeneratorFromEventPool::setupFileUniverse(std::string const& path) const
{
  // the path could refer to a local or alien filesystem; find out first
  bool onAliEn = strncmp(path.c_str(), std::string(alien_protocol_prefix).c_str(), alien_protocol_prefix.size()) == 0;
  std::vector<std::string> result;

  if (onAliEn) {
    // AliEn case
    // we support: (a) an actual evtgen file and (b) a path containing multiple eventfiles

    auto alienStatTypeCommand = std::string("alien.py stat ") + mConfig.eventPoolPath + std::string(" 2>/dev/null | grep Type ");
    auto typeString = executeCommand(alienStatTypeCommand);
    if (typeString.size() == 0) {
      return result;
    } else if (typeString.size() == 1 && typeString.front() == std::string("Type: f")) {
      // this is a file ... simply use it
      result.push_back(mConfig.eventPoolPath);
      return result;
    } else if (typeString.size() == 1 && typeString.front() == std::string("Type: d")) {
      // this is a directory
      // construct command to find actual event files
      std::string alienSearchCommand = std::string("alien.py find ") +
                                       mConfig.eventPoolPath + "/ " + std::string(eventpool_filename);

      auto universe_vector = executeCommand(alienSearchCommand);
      // check vector
      if (!checkFileUniverse(universe_vector)) {
        return result;
      }
      for (auto& f : universe_vector) {
        f = std::string(alien_protocol_prefix) + f;
      }

      return universe_vector;
    } else {
      LOG(error) << "Unsupported file type";
      return result;
    }
  } else {
    // local file case
    // check if the path is a regular file
    auto is_actual_file = std::filesystem::is_regular_file(path);
    if (is_actual_file) {
      // The files must match a criteria of being canonical paths ending with evtpool.root
      if (checkFileName(path)) {
        TFile rootfile(path.c_str(), "OPEN");
        if (!rootfile.IsZombie()) {
          result.push_back(path);
          return result;
        }
      } else {
        // otherwise assume it is a text file containing a list of files themselves
        auto files = readLines(path);
        if (checkFileUniverse(files)) {
          result = files;
          return result;
        }
      }
    } else {
      // check if the path is just a path
      // In this case we need to search something and check
      auto is_dir = std::filesystem::is_directory(path);
      if (!is_dir) {
        return result;
      }
      auto files = getLocalFileList(path);
      if (checkFileUniverse(files)) {
        result = files;
        return result;
      }
    }
  }
  return result;
}

} // namespace eventgen
} // end namespace o2

ClassImp(o2::eventgen::GeneratorFromEventPool);
ClassImp(o2::eventgen::GeneratorFromFile);
ClassImp(o2::eventgen::GeneratorFromO2Kine);
