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
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <FairPrimaryGenerator.h>
#include <TBranch.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TMCProcess.h>
#include <TParticle.h>
#include <TTree.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <numeric>
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

namespace
{
// connects to AliEn in case at least one of the given file names lives there
void connectToAlienIfNeeded(std::vector<std::string> const& filenames)
{
  if (gGrid) {
    return;
  }
  for (auto const& name : filenames) {
    if (name.starts_with("alien:/")) {
      TGrid::Connect("alien:");
      if (!gGrid) {
        LOG(fatal) << "Could not connect to alien, did you check the alien token?";
      }
      return;
    }
  }
}
} // namespace

std::vector<std::string> GeneratorFromO2Kine::splitFileNames(std::string const& filenames)
{
  // splits a comma-separated list of file names, trimming white space and dropping empty tokens
  return o2::utils::Str::tokenize(filenames, ',');
}

std::string GeneratorFromO2Kine::getCurrentFileName() const
{
  if (mCurrentFileIndex < 0 || mCurrentFileIndex >= (int)mFileNames.size()) {
    return std::string();
  }
  return mFileNames[mCurrentFileIndex];
}

void GeneratorFromO2Kine::closeCurrentFile()
{
  // the branches belong to the tree of the file, so they die with it
  mEventBranch = nullptr;
  mMCHeaderBranch = nullptr;
  mEventOrder.clear();
  mEventsAvailable = 0;
  mEventCounter = 0;
  if (mCurrentFile) {
    mCurrentFile->Close();
    delete mCurrentFile;
    mCurrentFile = nullptr;
  }
}

void GeneratorFromO2Kine::establishEventOrder()
{
  // The order in which the events of the current file are served is decided here
  mEventOrder.resize(mEventsAvailable);
  std::iota(mEventOrder.begin(), mEventOrder.end(), 0);
  if (mRandomize) {
    // Fisher-Yates shuffle based on the  ROOT random generator
    for (int i = mEventsAvailable - 1; i > 0; --i) {
      auto j = (int)gRandom->Integer(i + 1);
      std::swap(mEventOrder[i], mEventOrder[j]);
    }
  }
}

bool GeneratorFromO2Kine::openFile(int index)
{
  // opens one file of the list, connects the branches to it and fixes the
  // order in which its events are going to be served;
  // any previously open file is closed first, so that we never keep more than
  // one input file open at a time
  closeCurrentFile();
  if (index < 0 || index >= (int)mFileNames.size()) {
    return false;
  }
  auto const& name = mFileNames[index];

  mCurrentFile = TFile::Open(name.c_str());
  if (mCurrentFile == nullptr || mCurrentFile->IsZombie()) {
    LOG(error) << "EventFile " << name << " could not be opened";
    closeCurrentFile();
    return false;
  }
  // the kinematics will be stored inside a branch MCTrack
  // different events are stored inside different entries
  auto tree = (TTree*)mCurrentFile->Get("o2sim");
  if (!tree) {
    LOG(error) << "EventFile " << name << " does not contain an 'o2sim' tree";
    closeCurrentFile();
    return false;
  }
  mEventBranch = tree->GetBranch("MCTrack");
  if (!mEventBranch) {
    LOG(error) << "No MCTrack branch found in " << name;
    closeCurrentFile();
    return false;
  }
  mEventsAvailable = mEventBranch->GetEntries();
  if (mEventsAvailable <= 0) {
    LOG(warn) << "EventFile " << name << " does not contain any event";
    closeCurrentFile();
    return false;
  }
  mMCHeaderBranch = tree->GetBranch("MCEventHeader.");
  if (!mMCHeaderBranch) {
    LOG(warn) << "No MCEventHeader branch found in kinematics input file";
  }
  establishEventOrder();
  mCurrentFileIndex = index;
  mEventCounter = 0;
  mFilesUsed++;
  LOG(info) << "Reading events from kinematics file " << name << " (" << mEventsAvailable
            << " events, " << (mRandomize ? "randomized" : "sequential") << " order)";
  return true;
}

bool GeneratorFromO2Kine::openNextFile(bool wrapAround)
{
  // advances to the next usable file of the list; unusable files are skipped.
  auto numFiles = (int)mFileNames.size();
  if (numFiles == 0) {
    return false;
  }
  // we try each of the remaining files at most once
  for (int trial = 0; trial < numFiles; ++trial) {
    auto next = mCurrentFileIndex + 1 + trial;
    if (next >= numFiles) {
      if (!wrapAround) {
        return false;
      }
      if (trial == 0) {
        LOG(info) << "Reached the end of the input file list; reusing events from the beginning";
      }
      next = next % numFiles;
    }
    if (next == mCurrentFileIndex && mCurrentFile) {
      // this is the only usable file and it is already open - restart from it,
      // drawing a fresh event order
      establishEventOrder();
      mEventCounter = 0;
      return true;
    }
    if (openFile(next)) {
      return true;
    }
  }
  LOG(error) << "GeneratorFromO2Kine: no further usable input file";
  return false;
}

GeneratorFromO2Kine::GeneratorFromO2Kine(std::vector<std::string> const& filenames)
{
  // this generator should leave all dimensions the same as in the incoming kinematics file
  setMomentumUnit(1.);
  setEnergyUnit(1.);
  setPositionUnit(1.);
  setTimeUnit(1.);

  mFileNames = filenames;
  if (mFileNames.empty()) {
    LOG(error) << "GeneratorFromO2Kine: no input file given";
    return;
  }
  LOG(info) << "GeneratorFromO2Kine will read from " << mFileNames.size()
            << " file(s), one after the other";
  connectToAlienIfNeeded(mFileNames);
}

GeneratorFromO2Kine::GeneratorFromO2Kine(const char* name) : GeneratorFromO2Kine(splitFileNames(name ? name : ""))
{
}

GeneratorFromO2Kine::GeneratorFromO2Kine(O2KineGenConfig const& pars) : GeneratorFromO2Kine(splitFileNames(pars.fileName))
{
  mConfig = std::make_unique<O2KineGenConfig>(pars);
}

GeneratorFromO2Kine::GeneratorFromO2Kine(O2KineGenConfig const& pars, std::vector<std::string> const& filenames) : GeneratorFromO2Kine(filenames)
{
  mConfig = std::make_unique<O2KineGenConfig>(pars);
}

GeneratorFromO2Kine::~GeneratorFromO2Kine()
{
  closeCurrentFile();
}

bool GeneratorFromO2Kine::Init()
{

  // read and set params

  LOG(info) << "Init \'FromO2Kine\' generator";
  if (mConfig) {
    mSkipNonTrackable = mConfig->skipNonTrackable;
    mContinueMode = mConfig->continueMode;
    mRoundRobin = mConfig->roundRobin;
    mRandomize = mConfig->randomize;
    mRngSeed = mConfig->rngseed;
    mRandomPhi = mConfig->randomphi;
  }
  if (mRandomize && mRngSeed > 0) {
    // with a zero the seed given to the driver (o2-sim / o2-sim-dpl-eventgen --seed) stays in control
    gRandom->SetSeed(mRngSeed);
  }
  mCurrentFileIndex = -1;
  if (!openNextFile(false)) {
    LOG(error) << "Problem reading events from the given kinematics input";
    return false;
  }
  if (mStartEvent > 0) {
    if (mStartEvent < mEventsAvailable) {
      mEventCounter = mStartEvent;
    } else {
      LOG(error) << "start event bigger than available events";
    }
  }
  // Simple estimate of events without checking all the files.
  // To be discussed if we want instead to do this, or provide an additional file with the pools
  auto requested = getTotalNEvents();
  if (requested > 0 && !mRoundRobin && mEventsAvailable > 0) {
    auto estimate = (size_t)mEventsAvailable * mFileNames.size();
    if (estimate < requested) {
      LOG(warn) << "This job will request " << requested << " events, but the input ("
                << mFileNames.size() << " file(s), " << mEventsAvailable
                << " events in the first one) holds only about " << estimate << ". Unless the "
                << "remaining files are larger, the job will stop with 'ran out of events' - "
                << "provide more files/events or enable roundRobin";
    }
  }

  return true;
}

void GeneratorFromO2Kine::SetStartEvent(int start)
{
  // this refers to the first file and is applied once that file has been opened
  mStartEvent = start;
}

bool GeneratorFromO2Kine::importParticles()
{
  // NOTE: This should be usable with kinematics files without secondaries
  // It might need some adjustment to make it work with secondaries or to continue
  // from a kinematics snapshot

  // Next file in the list opened when the events of the current one are used up
  if (mEventCounter >= mEventsAvailable) {
    if (!openNextFile(mRoundRobin)) {
      auto requested = getTotalNEvents();
      LOG(fatal) << "GeneratorFromO2Kine: ran out of events after " << mEventsServed
                 << " event(s) from " << mFilesUsed << " input file(s)"
                 << (requested > 0 ? " (" + std::to_string(requested) + " were requested)" : "")
                 << ". Provide more input files/events or allow reusing them via roundRobin";
      return false;
    }
  }
  if (mCurrentFile == nullptr || mEventBranch == nullptr || mEventCounter >= (int)mEventOrder.size()) {
    LOG(fatal) << "GeneratorFromO2Kine: no input file available";
    return false;
  }
  // the entry to be read from the file which is currently open; the order was fixed
  // when the file was opened, so every event of it is used exactly once
  auto entry = mEventOrder[mEventCounter];
  if (mRandomize) {
    LOG(info) << "GeneratorFromO2Kine - Picking event " << entry;
  }

  double dPhi = 0.;
  // Phi rotation
  if (mRandomPhi) {
    dPhi = gRandom->Uniform(2 * TMath::Pi());
    LOG(info) << "Rotating phi by " << dPhi;
  }

  int particlecounter = 0;

  std::vector<o2::MCTrack>* tracks = nullptr;
  mEventBranch->SetAddress(&tracks);
  mEventBranch->GetEntry(entry);
  mLastEntryRead = entry;

  if (mMCHeaderBranch) {
    o2::dataformats::MCEventHeader* mcheader = nullptr;
    mMCHeaderBranch->SetAddress(&mcheader);
    mMCHeaderBranch->GetEntry(entry);
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
  mEventsServed++;

  if (tracks) {
    delete tracks;
  }

  LOG(info) << "Event generator put " << particlecounter << " on stack";
  return true;
}

void GeneratorFromO2Kine::updateHeader(o2::dataformats::MCEventHeader* eventHeader)
{
  /** update header **/

  // we forward the original header information if any
  if (mOrigMCEventHeader.get()) {
    eventHeader->copyInfoFrom(*mOrigMCEventHeader.get());
    // we forward also the original basic vertex information contained in FairMCEventHeader
    static_cast<FairMCEventHeader&>(*eventHeader) = static_cast<FairMCEventHeader&>(*mOrigMCEventHeader.get());
  }

  // put additional information about input file and event number of the current event
  eventHeader->putInfo<std::string>("forwarding-generator", "generatorFromO2Kine");
  eventHeader->putInfo<std::string>("forwarding-generator_inputFile", getCurrentFileName());
  eventHeader->putInfo<int>("forwarding-generator_inputEventNumber", mLastEntryRead);
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

  // initialize the event pool.
  // When zero is provided as seed, the global ROOT random sequence is followed
  // so that the seed given to the o2-sim or o2-sim-dpl-eventgen
  // also determines which files of the pool the job will pick
  if (mConfig.rngseed > 0) {
    mRandomEngine.seed(mConfig.rngseed);
  } else {
    mRandomEngine.seed(gRandom->Integer(std::numeric_limits<int>::max()));
  }
  TString expPath(mConfig.eventPoolPath);
  gSystem->ExpandPathName(expPath);
  mPoolFilesAvailable = setupFileUniverse(expPath.Data());

  if (mPoolFilesAvailable.size() == 0) {
    LOG(error) << "No file found that can be used with EventPool generator";
    return false;
  }
  LOG(info) << "Found " << mPoolFilesAvailable.size() << " available event pool files";

  // shuffle the pool so that different jobs go through it in a different order
  mFilesChosen = selectFiles(mPoolFilesAvailable);
  LOG(info) << "EventPool will go through all " << mFilesChosen.size() << " pool files";

  // we bring up the internal mO2KineGenerator with the shuffled file list
  auto kine_config = O2KineGenConfig{
    .skipNonTrackable = mConfig.skipNonTrackable,
    .continueMode = false,
    .roundRobin = mConfig.roundRobin,
    .randomize = mConfig.randomize,
    .rngseed = mConfig.rngseed,
    .randomphi = mConfig.randomphi};
  mO2KineGenerator.reset(new GeneratorFromO2Kine(kine_config, mFilesChosen));
  return mO2KineGenerator->Init();
}

std::vector<std::string> GeneratorFromEventPool::selectFiles(std::vector<std::string> const& universe)
{
  // shuffles the whole pool universe so that different jobs go through it in a
  // different order
  auto result = universe;
  std::shuffle(result.begin(), result.end(), mRandomEngine);
  return result;
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
      // this is a file:
      // 1) list of files ==> select one of the lines and use it
      // 2) evtpool.root  ==> use as it is
      if (!checkFileName(path)) {
        // Assume it is a text file containing a list of pools
        auto tmpPath = (std::filesystem::temp_directory_path() / ("list_" + std::to_string(getpid()) + ".txt")).string();
        auto res = TFile::Cp(Form("%s?filetype=raw", path.c_str()), tmpPath.c_str());
        if (!res) {
          LOG(fatal) << "Failed to copy file from AliEn: " << path;
        } else {
          auto files = readLines(tmpPath);
          if (checkFileUniverse(files)) {
            result = files;
          } else {
            LOG(fatal) << "The list of files in " << path << " is not valid";
          }
          std::filesystem::remove(tmpPath);
        }
        return result;
      }
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
