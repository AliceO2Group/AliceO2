// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Sandro Wenzel <sandro.wenzel@cern.ch>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   MCStepLoggerImpl.cxx
//  @author Sandro Wenzel
//  @since  2017-06-29
//  @brief  A logging service for MCSteps (hooking into Stepping of TVirtualMCApplication's)

#include <TVirtualMCApplication.h>
#include <TVirtualMC.h>
#include <sstream>
#include <TTree.h>

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <set>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace o2 {

class StepLogger
{
  int stepcounter = 0;

  std::set<int> trackset;
  std::set<int> pdgset;
  std::map<int, int> volumetosteps;
  std::map<int, char const*> idtovolname;

  // TODO: consider writing to a TTree/TFile
 public:
  void addStep(TVirtualMC* mc)
  {
    assert(mc);
    stepcounter++;
    auto stack=mc->GetStack();
    assert(stack);
    trackset.insert(stack->GetCurrentTrackNumber());
    pdgset.insert(mc->TrackPid());
    int copyNo;
    auto id = mc->CurrentVolID(copyNo);
    if (volumetosteps.find(id) == volumetosteps.end()){
      volumetosteps.insert(std::pair<int, int>(id, 0));
    }
    else {
      volumetosteps[id]++;
    }
    if (idtovolname.find(id) == idtovolname.end()) {
      idtovolname.insert(std::pair<int, char const*>(id, mc->CurrentVolName()));
    }
  }

  void clear() {
    stepcounter = 0;
    trackset.clear();
    pdgset.clear();
    volumetosteps.clear();
    idtovolname.clear();
  }

  void flush() {
    std::cerr << "did " << stepcounter << " steps \n";
    std::cerr << "transported " << trackset.size() << " different tracks \n";
    std::cerr << "transported " << pdgset.size() << " different types \n";
    // summarize steps per volume
    for (auto& p : volumetosteps) {
      std::cout << " VolName " << idtovolname[p.first] << " COUNT " << p.second << "\n";
    }
    clear();
  }
};

StepLogger logger;

} // end namespace

// a generic function that can dispatch to the original method of a TVirtualMCApplication
// (for functions of type void TVirtualMCApplication::Method(void); )
extern "C" void dispatchOriginal(TVirtualMCApplication* app, char const* libname, char const* origFunctionName)
{
  typedef void (TVirtualMCApplication::*StepMethodType)();
  // static map to avoid having to lookup the right symbols in the shared lib at each call
  // (We could do this outside of course)
  static std::map<const char *, StepMethodType> functionNameToSymbolMap;
  StepMethodType origMethod = nullptr;

  auto iter = functionNameToSymbolMap.find(origFunctionName);
  if(iter == functionNameToSymbolMap.end()){
    auto libHandle = dlopen(libname, RTLD_NOW);
    // try to make the library loading a bit more portable:
    if (!libHandle){
      // try appending *.so
      std::stringstream stream;
      stream << libname << ".so";
      libHandle = dlopen(stream.str().c_str(), RTLD_NOW);
    } 
    if (!libHandle){
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
    functionNameToSymbolMap[origFunctionName]=origMethod;
  } else {
    origMethod = iter->second;
  }
  (app->*origMethod)();
}

extern "C" void performLogging(TVirtualMCApplication* app)
{
  static TVirtualMC* mc = TVirtualMC::GetMC();
  o2::logger.addStep(mc);
}

extern "C" void flushLog(){
  o2::logger.flush();
}
