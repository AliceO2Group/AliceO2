// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PDG_H_
#define ALICEO2_PDG_H_

#include <Rtypes.h>

//< Class to encapsulate the ALICE updates to TDatabasePDG.h
//< Can be used by TGeant3 and TGeant4
//< Author: andreas.morsch@cern.ch
//< Ported from AliRoot to O2 by ruben.shahoyan@cern.ch

namespace o2
{

class PDG
{
 public:
  static void addParticlesToPdgDataBase(int verbose = 0);

 private:
  static void addParticle(const char* name, const char* title, double mass, bool stable, double width, double charge,
                          const char* particleClass, int pdgCode, int verbose = 0);
  static void addAntiParticle(const char* name, Int_t pdgCode, int verbose = 0);

  ClassDefNV(PDG, 1); // PDG database related information
};
} // namespace o2

#endif
