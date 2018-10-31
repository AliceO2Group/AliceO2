// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   MCStepLoggerImpl.cxx
//  @author Sandro Wenzel
//  @since  2017-06-29
//  @brief  A logging service for MCSteps (hooking into Stepping of TVirtualMCApplication's)

#include "MCStepLogger/StepInfo.h"
#include "MCStepLogger/MetaInfo.h"
#include <TBranch.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TTree.h>
#include <TVirtualMC.h>
#include <TVirtualMCApplication.h>
#include <TVirtualMagField.h>
#include <sstream>

#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace o2
{
const char* getLogFileName()
{
  if (const char* f = std::getenv("MCSTEPLOG_OUTFILE")) {
    return f;
  } else {
    return "MCStepLoggerOutput.root";
  }
}

const char* getVolMapFile()
{
  if (const char* f = std::getenv("MCSTEPLOG_VOLMAPFILE")) {
    return f;
  } else {
    return "MCStepLoggerVolMap.dat";
  }
}

// initializes a mapping from volumename to detector
// used for step resolution to detectors
void initVolumeMap()
{
  auto volmap = new std::map<std::string, std::string>;
  // open for reading or fail
  std::ifstream ifs;
  auto f = getVolMapFile();
  std::cerr << "[MCLOGGER:] TRYING TO READ VOLUMEMAPS FROM " << f << "\n";
  ifs.open(f);
  if (ifs.is_open()) {
    std::string line;
    while (std::getline(ifs, line)) {
      std::istringstream ss(line);
      std::string token;
      // split the line into key + value
      int counter = 0;
      std::string keyvalue[2] = { "NULL", "NULL" };
      while (counter < 2 && std::getline(ss, token, ' ')) {
        if (!token.empty()) {
          keyvalue[counter] = token;
          counter++;
        }
      }
      // put into map
      volmap->insert({ keyvalue[0], keyvalue[1] });
    }
    ifs.close();
    StepInfo::volnametomodulemap = volmap;
  } else {
    std::cerr << "[MCLOGGER:] VOLUMEMAPSFILE NOT FILE\n";
    StepInfo::volnametomodulemap = nullptr;
  }
}

template <typename T>
void flushToTTree(const char* branchname, T* address)
{
  TFile* f = new TFile(getLogFileName(), "UPDATE");
  const char* treename = "StepLoggerTree";
  auto tree = (TTree*)f->Get(treename);
  if (!tree) {
    // create tree
    tree = new TTree(treename, "Tree container information from MC step logger");
  }
  auto branch = tree->GetBranch(branchname);
  if (!branch) {
    branch = tree->Branch(branchname, &address);
  }
  branch->SetAddress(&address);
  branch->Fill();
  tree->SetEntries(branch->GetEntries());
  // To avoid large number of cycles since whenever the file is opened and things are written, this is done as a new cycle
  //f->Write();
  tree->Write("", TObject::kOverwrite);
  f->Close();
  delete f;
}

void initTFile()
{
  if (!std::getenv("MCSTEPLOG_TTREE")) {
    return;
  }
  TFile* f = new TFile(getLogFileName(), "RECREATE");
  f->Close();
  delete f;
}

// a class collecting field access per volume
class FieldLogger
{
  int counter = 0;
  std::map<int, int> volumetosteps;
  std::map<int, std::string> idtovolname;
  bool mTTreeIO = false;
  std::vector<MagCallInfo> callcontainer;

 public:
  FieldLogger()
  {
    // check if streaming or interactive
    // configuration done via env variable
    if (std::getenv("MCSTEPLOG_TTREE")) {
      mTTreeIO = true;
    }
  }

  void addStep(TVirtualMC* mc, const double* x, const double* b)
  {
    if (mTTreeIO) {
      callcontainer.emplace_back(mc, x[0], x[1], x[2], b[0], b[1], b[2]);
      return;
    }
    counter++;
    int copyNo;
    auto id = mc->CurrentVolID(copyNo);
    if (volumetosteps.find(id) == volumetosteps.end()) {
      volumetosteps.insert(std::pair<int, int>(id, 0));
    } else {
      volumetosteps[id]++;
    }
    if (idtovolname.find(id) == idtovolname.end()) {
      idtovolname.insert(std::pair<int, std::string>(id, std::string(mc->CurrentVolName())));
    }
  }

  void clear()
  {
    counter = 0;
    volumetosteps.clear();
    idtovolname.clear();
    if (mTTreeIO) {
      callcontainer.clear();
    }
  }

  void flush()
  {
    if (mTTreeIO) {
      flushToTTree("Calls", &callcontainer);
    } else {
      std::cerr << "[FIELDLOGGER]: did " << counter << " steps \n";
      // summarize steps per volume
      for (auto& p : volumetosteps) {
        std::cerr << "[FIELDLOGGER]: VolName " << idtovolname[p.first] << " COUNT " << p.second;
        std::cerr << "\n";
      }
      std::cerr << "[FIELDLOGGER]: ----- END OF EVENT ------\n";
    }
    clear();
  }
};

class StepLogger
{
  int stepcounter = 0;

  std::set<int> trackset;
  std::set<int> pdgset;
  std::map<int, int> volumetosteps;
  std::map<int, std::string> idtovolname;
  std::map<int, int> volumetoNSecondaries;            // number of secondaries created in this volume
  std::map<std::pair<int, int>, int> volumetoProcess; // mapping of volumeid x processID to secondaries produced

  std::vector<StepInfo> container;
  bool mTTreeIO = false;

 public:
  StepLogger()
  {
    // check if streaming or interactive
    // configuration done via env variable
    if (std::getenv("MCSTEPLOG_TTREE")) {
      mTTreeIO = true;
    }
    // try to load the volumename -> modulename mapping
    initVolumeMap();
  }

  void addStep(TVirtualMC* mc)
  {
    if (mTTreeIO) {
      container.emplace_back(mc);
    } else {
      assert(mc);
      stepcounter++;

      auto stack = mc->GetStack();
      assert(stack);
      trackset.insert(stack->GetCurrentTrackNumber());
      pdgset.insert(mc->TrackPid());
      int copyNo;
      auto id = mc->CurrentVolID(copyNo);

      TArrayI procs;
      mc->StepProcesses(procs);

      if (volumetosteps.find(id) == volumetosteps.end()) {
        volumetosteps.insert(std::pair<int, int>(id, 0));
      } else {
        volumetosteps[id]++;
      }
      if (idtovolname.find(id) == idtovolname.end()) {
        idtovolname.insert(std::pair<int, std::string>(id, std::string(mc->CurrentVolName())));
      }

      // for the secondaries
      if (volumetoNSecondaries.find(id) == volumetoNSecondaries.end()) {
        volumetoNSecondaries.insert(std::pair<int, int>(id, mc->NSecondaries()));
      } else {
        volumetoNSecondaries[id] += mc->NSecondaries();
      }

      // for the processes
      for (int i = 0; i < mc->NSecondaries(); ++i) {
        auto process = mc->ProdProcess(i);
        auto p = std::pair<int, int>(id, process);
        if (volumetoProcess.find(p) == volumetoProcess.end()) {
          volumetoProcess.insert(std::pair<std::pair<int, int>, int>(p, 1));
        } else {
          volumetoProcess[p]++;
        }
      }
    }
  }

  void clear()
  {
    stepcounter = 0;
    trackset.clear();
    pdgset.clear();
    volumetosteps.clear();
    idtovolname.clear();
    volumetoNSecondaries.clear();
    volumetoProcess.clear();
    if (mTTreeIO) {
      container.clear();
    }
    StepInfo::resetCounter();
  }

  // prints list of processes for volumeID
  void printProcesses(int volid)
  {
    for (auto& p : volumetoProcess) {
      if (p.first.first == volid) {
        std::cerr << "P[" << TMCProcessName[p.first.second] << "]:" << p.second << "\t";
      }
    }
  }

  void flush()
  {
    if (!mTTreeIO) {
      std::cerr << "[STEPLOGGER]: did " << stepcounter << " steps \n";
      std::cerr << "[STEPLOGGER]: transported " << trackset.size() << " different tracks \n";
      std::cerr << "[STEPLOGGER]: transported " << pdgset.size() << " different types \n";
      // summarize steps per volume
      for (auto& p : volumetosteps) {
        std::cerr << "[STEPLOGGER]: VolName " << idtovolname[p.first] << " COUNT " << p.second << " SECONDARIES "
                  << volumetoNSecondaries[p.first] << " ";
        // loop over processes
        printProcesses(p.first);
        std::cerr << "\n";
      }
      std::cerr << "[STEPLOGGER]: ----- END OF EVENT ------\n";
    } else {
      flushToTTree("Steps", &container);
      flushToTTree("Lookups", &StepInfo::lookupstructures);
      // we need to reset some parts of the lookupstructures for the next event
      StepInfo::lookupstructures.tracktoparent.clear();
      StepInfo::lookupstructures.tracktopdg.clear();
    }
    clear();
  }
};

// the global logging instances (in anonymous namespace)
// pointers to dissallow construction at each library load
StepLogger* logger;
FieldLogger* fieldlogger;
} // end namespace

// a helper template kernel describing generically the redispatching prodecure
template <typename Object /* the original object type */, typename MethodType /* member function type */,
          typename... Args /* original arguments to function */>
void dispatchOriginalKernel(Object* obj, char const* libname, char const* origFunctionName, Args... args)
{
  // Object, MethodType, and Args are of course related so we could do some static_assert checks or automatic deduction

  // static map to avoid having to lookup the right symbols in the shared lib at each call
  // (We could do this outside of course)
  static std::map<const char*, MethodType> functionNameToSymbolMap;
  MethodType origMethod = nullptr;

  auto iter = functionNameToSymbolMap.find(origFunctionName);
  if (iter == functionNameToSymbolMap.end()) {
    auto libHandle = dlopen(libname, RTLD_NOW);
    // try to make the library loading a bit more portable:
    if (!libHandle) {
      // try appending *.so
      std::stringstream stream;
      stream << libname << ".so";
      libHandle = dlopen(stream.str().c_str(), RTLD_NOW);
    }
    if (!libHandle) {
      // try appending *.dylib
      std::stringstream stream;
      stream << libname << ".dylib";
      libHandle = dlopen(stream.str().c_str(), RTLD_NOW);
    }
    assert(libHandle);
    void* symbolAddress = dlsym(libHandle, origFunctionName);
    assert(symbolAddress);
// Purposely ignore compiler warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsizeof-pointer-memaccess"
    // hack since C++ does not allow casting to C++ member function pointers
    // thanks to gist.github.com/mooware/1174572
    memcpy(&origMethod, &symbolAddress, sizeof(&symbolAddress));
#pragma GCC diagnostic pop
    functionNameToSymbolMap[origFunctionName] = origMethod;
  } else {
    origMethod = iter->second;
  }
  // the final C++ member function call redispatch
  (obj->*origMethod)(args...);
}

// a generic function that can dispatch to the original method of a TVirtualMCApplication
extern "C" void dispatchOriginal(TVirtualMCApplication* app, char const* libname, char const* origFunctionName)
{
  typedef void (TVirtualMCApplication::*StepMethodType)();
  dispatchOriginalKernel<TVirtualMCApplication, StepMethodType>(app, libname, origFunctionName);
}

// a generic function that can dispatch to the original method of a TVirtualMagField
extern "C" void dispatchOriginalField(TVirtualMagField* field, char const* libname, char const* origFunctionName,
                                      const double x[3], double* B)
{
  typedef void (TVirtualMagField::*MethodType)(const double[3], double*);
  dispatchOriginalKernel<TVirtualMagField, MethodType>(field, libname, origFunctionName, x, B);
}

extern "C" void performLogging(TVirtualMCApplication* app)
{
  static TVirtualMC* mc = TVirtualMC::GetMC();
  o2::logger->addStep(mc);
}

extern "C" void logField(double const* p, double const* b)
{
  static TVirtualMC* mc = TVirtualMC::GetMC();
  o2::fieldlogger->addStep(mc, p, b);
}

extern "C" void initLogger()
{
  // init TFile for logging output
  o2::initTFile();
  // initializes the logging instances
  o2::logger = new o2::StepLogger();
  o2::fieldlogger = new o2::FieldLogger();
}

extern "C" void flushLog()
{
  std::cerr << "[MCLOGGER:] START FLUSHING ----\n";
  o2::logger->flush();
  o2::fieldlogger->flush();
  std::cerr << "[MCLOGGER:] END FLUSHING ----\n";
}
