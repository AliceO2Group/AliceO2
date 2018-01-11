// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Geometry.h"

using namespace o2::phos;

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Int_t Geometry::RelToAbsId(Int_t moduleNumber, Int_t strip, Int_t cell)
{
  // calculates absolute cell Id from moduleNumber, strip (number) and cell (number)
  // PHOS layout parameters:
  const Int_t nStrpZ = 28;                 // Number of strips along z-axis
  const Int_t nCrystalsInModule = 56 * 64; // Total number of crystals in module
  const Int_t nCellsXInStrip = 8;          // Number of crystals in strip unit along x-axis
  const Int_t nZ = 56;                     // nStripZ * nCellsZInStrip

  Int_t row = nStrpZ - (strip - 1) % nStrpZ;
  Int_t col = (Int_t)TMath::Ceil((Double_t)strip / (nStrpZ)) - 1;

  return (moduleNumber - 1) * nCrystalsInModule + row * 2 + (col * nCellsXInStrip + (cell - 1) / 2) * nZ -
         (cell & 1 ? 1 : 0);
}
