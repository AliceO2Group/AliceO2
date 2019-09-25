// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                  M. Al-Turany   June 2014                     -----
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -----             Pythia6Generator source file                      -----
// -----          Created 08/08/08  by S. Spataro                      -----
// -------------------------------------------------------------------------
#include "Generators/Pythia6Generator.h"

#include "FairPrimaryGenerator.h"

#include <iostream>
#include <cstdio>

using std::cout;
using std::endl;
using std::max;
namespace o2
{
namespace eventgen
{

// -----   Default constructor   ------------------------------------------
Pythia6Generator::Pythia6Generator() = default;
// ------------------------------------------------------------------------

// -----   Standard constructor   -----------------------------------------
Pythia6Generator::Pythia6Generator(const char* fileName)
  : mFileName(fileName), mInputFile(nullptr), mVerbose(0)
{
  cout << "-I Pythia6Generator: Opening input file " << mFileName << endl;
  if ((mInputFile = fopen(mFileName, "r")) == nullptr) {
    Fatal("Pythia6Generator", "Cannot open input file.");
  }

  // mPDG=TDatabasePDG::Instance();
}
// ------------------------------------------------------------------------

// -----   Destructor   ---------------------------------------------------
Pythia6Generator::~Pythia6Generator()
{
  CloseInput();
}
// ------------------------------------------------------------------------

// -----   Public method ReadEvent   --------------------------------------
Bool_t Pythia6Generator::ReadEvent(FairPrimaryGenerator* primGen)
{

  // Check for input file
  if (!mInputFile) {
    // if ( ! mInputFile->is_open() ) {
    cout << "-E Pythia6Generator: Input file not open!" << endl;
    return kFALSE;
  }

  // Define event variable to be read from file
  Int_t ntracks = 0, eventID = 0, ncols = 0;

  // Define track variables to be read from file
  Int_t nLev = 0, pdgID = 0, nM1 = -1, nM2 = -1, nDF = -1, nDL = -1;
  Float_t fPx = 0., fPy = 0., fPz = 0., fM = 0., fE = 0.;
  Float_t fVx = 0., fVy = 0., fVz = 0., fT = 0.;

  // Read event header line from input file

  Int_t max_nr = 0;

  Text_t buffer[200];
  ncols = fscanf(mInputFile, "%d\t%d", &eventID, &ntracks);

  if (ncols && ntracks > 0) {

    if (mVerbose > 0)
      cout << "Event number: " << eventID << "\tNtracks: " << ntracks << endl;

    for (Int_t ll = 0; ll < ntracks; ll++) {
      ncols = fscanf(mInputFile, "%d %d %d %d %d %d %f %f %f %f %f %f %f %f %f", &nLev, &pdgID, &nM1, &nM2, &nDF, &nDL, &fPx, &fPy, &fPz, &fE, &fM, &fVx, &fVy, &fVz, &fT);
      if (mVerbose > 0)
        cout << nLev << "\t" << pdgID << "\t" << nM1 << "\t" << nM2 << "\t" << nDF << "\t" << nDL << "\t" << fPx << "\t" << fPy << "\t" << fPz << "\t" << fE << "\t" << fM << "\t" << fVx << "\t" << fVy << "\t" << fVz << "\t" << fT << endl;
      if (nLev == 1)
        primGen->AddTrack(pdgID, fPx, fPy, fPz, fVx, fVy, fVz);
    }
  } else {
    cout << "-I Pythia6Generator: End of input file reached " << endl;
    CloseInput();
    return kFALSE;
  }

  // If end of input file is reached : close it and abort run
  if (feof(mInputFile)) {
    cout << "-I Pythia6Generator: End of input file reached " << endl;
    CloseInput();
    return kFALSE;
  }

  /*
    cout << "-I Pythia6Generator: Event " << eventID << ",  vertex = ("
    << vx << "," << vy << "," << vz << ") cm,  multiplicity "
    << ntracks << endl;
  */

  return kTRUE;
}
// ------------------------------------------------------------------------

// -----   Private method CloseInput   ------------------------------------
void Pythia6Generator::CloseInput()
{
  if (mInputFile) {
    //if ( mInputFile->is_open() ) {
    {
      cout << "-I Pythia6Generator: Closing input file "
           << mFileName << endl;
      //  mInputFile->close();

      fclose(mInputFile);
    }
    delete mInputFile;
    mInputFile = nullptr;
  }
}
// ------------------------------------------------------------------------

} // namespace eventgen
} // namespace o2

ClassImp(o2::eventgen::Pythia6Generator);
