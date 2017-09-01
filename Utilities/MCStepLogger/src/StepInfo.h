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

#ifndef O2_STEPINFO
#define O2_STEPINFO

#include <Rtypes.h>
#include <chrono>
#include <map>

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

struct StepInfo {
  StepInfo() = default;
  // construct directly using virtual mc
  StepInfo(TVirtualMC* mc);

  long cputimestamp;
  int stepid = -1; // serves as primary key
  int eventid = -1;
  int volId = -1; // keep another branch somewhere mapping this to name, medium, etc.
  int copyNo = -1;
  int trackID = -1;
  int pdg = 0;
  std::string pname; // particle name
  std::string modulename;
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
  bool primary = false;

  std::string volname;
  std::string mediumname;

  static int stepcounter;           //!
  static StepInfo* currentinstance; //!
  static std::chrono::time_point<std::chrono::high_resolution_clock> starttime;
  static void resetCounter() { stepcounter = -1; }
  static std::map<std::string, std::string>* volnametomodulemap;
  static std::vector<std::string*> volidtomodulevector;

  ClassDefNV(StepInfo, 2);
};

struct MagCallInfo {
  MagCallInfo() = default;
  MagCallInfo(TVirtualMC* mc, float x, float y, float z, float Bx, float By, float Bz);

  long id = -1;
  long stepid = -1; // cross-reference to current MC stepid (if any??)
  //  StepInfo stepinfo; // cross-reference to step info via pointer
  float x = 0.;
  float y = 0.;
  float z = 0.;
  float Bx = 0.;
  float By = 0.;
  float Bz = 0.;

  static int stepcounter;
  ClassDefNV(MagCallInfo, 1);
};
}
#endif
