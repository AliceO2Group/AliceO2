// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#ifndef O2_STEPINFO
#define O2_STEPINFO

#include <Rtypes.h>
#include <chrono>
#include <iostream>
#include <map>
#include <vector>

class TVirtualMC;
class TGeoVolume;
class TGeoMedium;

namespace o2
{
// class collecting info about one MC step done

struct VolInfoContainer {
  VolInfoContainer() = default;

  // keeps info about volumes (might exist somewhere else already)
  // essentially a mapping from volumeID x copyNo to TGeoVolumes
  std::vector<std::vector<TGeoVolume const*>*> volumes; // sparse container

  void insert(int id, int copyNo, TGeoVolume const* vol)
  {
    if (volumes.size() <= id) {
      volumes.resize(id + 1, nullptr);
    }
    if (volumes[id] == nullptr) {
      volumes[id] = new std::vector<TGeoVolume const*>;
    }
    if (volumes[id]->size() <= copyNo) {
      volumes[id]->resize(copyNo + 1, nullptr);
    }
    (*volumes[id])[copyNo] = vol;
  }

  TGeoVolume const* get(int id, int copy) const { return (*volumes[id])[copy]; }
  ClassDefNV(VolInfoContainer, 1);
};

// LookupStructures to translate some step information such
// as volumeid, to readable names
struct StepLookups {
  // using pointers to allow "nullptr==unknown" and faster query
  std::vector<std::string*> volidtovolname;
  std::vector<std::string*> volidtomodule;
  std::vector<std::string*> volidtomedium;
  std::vector<int> tracktopdg;
  std::vector<int> tracktoparent; // when parent is -1 we mean primary

  void insertVolName(int index, std::string const& s) { insertValueAt(index, s, volidtovolname); }
  void insertModuleName(int index, std::string const& s) { insertValueAt(index, s, volidtomodule); }
  std::string* getModuleAt(int index) const
  {
    if (index >= volidtomodule.size())
      return nullptr;
    return volidtomodule[index];
  }

  void insertPDG(int trackindex, int pdg)
  {
    constexpr int INVALIDPDG = 0;
    if (trackindex >= tracktopdg.size()) {
      tracktopdg.resize(trackindex + 1, INVALIDPDG);
    }
    auto prev = tracktopdg[trackindex];
    if (prev != INVALIDPDG && prev != pdg) {
      std::cerr << "Warning: Seeing more than one pdg for same trackID\n";
    }
    tracktopdg[trackindex] = pdg;
  }

  void insertParent(int trackindex, int parent)
  {
    constexpr int PRIMARY = -1;
    if (trackindex >= tracktoparent.size()) {
      tracktoparent.resize(trackindex + 1, PRIMARY);
    }
    auto prev = tracktoparent[trackindex];
    if (prev != PRIMARY && prev != parent) {
      std::cerr << "Warning: Seeing more than one parent for same trackID\n";
    }
    tracktoparent[trackindex] = parent;
  }

 private:
  void insertValueAt(int index, std::string const& s, std::vector<std::string*>& container)
  {
    if (index >= container.size()) {
      container.resize(index + 1, nullptr);
    }
    //#ifdef CHECKMODE
    // check that if a value exists at some index it is the same that we want to write
    if (container[index] != nullptr) {
      auto previous = *(container[index]);
      if (s.compare(previous) != 0) {
        std::cerr << "trying to override " << previous << " with " << s << "\n";
      }
    }
    //#endif
    // can we use unique pointers??
    container[index] = new std::string(s);
  }

  ClassDefNV(StepLookups, 1);
};

struct StepInfo {
  StepInfo() = default;
  // construct directly using virtual mc
  StepInfo(TVirtualMC* mc);

  // long cputimestamp;
  int stepid = -1; // serves as primary key
  int volId = -1;  // keep another branch somewhere mapping this to name, medium, etc.
  int copyNo = -1;
  int trackID = -1;
  float x = 0.;
  float y = 0.;
  float z = 0.;
  float E = 0.;
  float step = 0.;
  float maxstep = 0.;
  int nsecondaries = 0;
  int* secondaryprocesses = nullptr; //[nsecondaries]
  int nprocessesactive = 0;          // number of active processes
  bool stopped = false;              //

  static int stepcounter;           //!
  static StepInfo* currentinstance; //!
  static std::chrono::time_point<std::chrono::high_resolution_clock> starttime;
  static void resetCounter() { stepcounter = -1; }
  static std::map<std::string, std::string>* volnametomodulemap;
  static std::vector<std::string*> volidtomodulevector;

  static StepLookups lookupstructures;
  ClassDefNV(StepInfo, 2);
};

struct MagCallInfo {
  MagCallInfo() = default;
  MagCallInfo(TVirtualMC* mc, float x, float y, float z, float Bx, float By, float Bz);

  long id = -1;
  long stepid = -1; // cross-reference to current MC stepid (if any??)
  float x = 0.;
  float y = 0.;
  float z = 0.;
  float B = 0.; // absolute value of the B field

  static int stepcounter;
  ClassDefNV(MagCallInfo, 1);
};
}
#endif
