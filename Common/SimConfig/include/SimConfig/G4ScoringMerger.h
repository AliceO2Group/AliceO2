// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMCONFIG_G4SCORINGMERGER_H
#define O2_SIMCONFIG_G4SCORINGMERGER_H

#include <string>

namespace o2::conf
{

/// Name of the Geant4 scoring dump written by one simulation worker
std::string g4ScoringWorkerFileName(const std::string& meshName, int pid);

/// Sum the per-worker Geant4 scoring dumps <mesh>.worker<pid>.txt in a directory into <mesh>.txt.
/// If expectedWorkers > 0, each mesh must have exactly that many dumps.
/// Returns the number of merged meshes, or -1 if the worker files are inconsistent.
int mergeG4ScoringDumps(const std::string& directory, int expectedWorkers = 0);

} // namespace o2::conf

#endif
