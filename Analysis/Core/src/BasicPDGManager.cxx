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

#include "AnalysisCore/BasicPDGManager.h"

namespace o2::pdg
{
PDGManagerInstance::PDGManagerInstance(const char* path)
  : dbPath{path}
{
  db = new TDatabasePDG();
  db->ReadPDGTable(path);
}

PDGManagerInstance::~PDGManagerInstance()
{
  db->Delete();
}

void PDGManagerInstance::setDatabasePath(const char* path)
{
  dbPath = path;
}

void PDGManagerInstance::reReadDatabase()
{
  if (db != nullptr) {
    db->ReadPDGTable(dbPath.c_str());
  }
}

TParticlePDG* PDGManagerInstance::GetParticle(int pdgcode)
{
  return db->GetParticle(pdgcode);
}
} // namespace o2::pdg
