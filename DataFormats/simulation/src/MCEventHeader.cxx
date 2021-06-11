// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#include "SimulationDataFormat/MCEventHeader.h"
#include "FairRootManager.h"
#include <TFile.h>
#include <TTree.h>

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

void MCEventHeader::Reset()
{
  /** reset **/

  FairMCEventHeader::Reset();

  clearInfo();
  mEmbeddingFileName.clear();
  mEmbeddingEventIndex = 0;
}

void MCEventHeader::extractFileFromKinematics(std::string_view kinefilename, std::string_view targetfilename)
{
  auto oldfile = TFile::Open(kinefilename.data());
  auto kinetree = (TTree*)oldfile->Get("o2sim");
  // deactivate all branches
  kinetree->SetBranchStatus("*", 0);
  // activate the header branch
  kinetree->SetBranchStatus("MCEventHeader*", 1);
  // create a new file + a clone of old tree header. Do not copy events
  auto newfile = TFile::Open(targetfilename.data(), "RECREATE");
  auto newtree = kinetree->CloneTree(0);
  // here we copy the branches
  newtree->CopyEntries(kinetree, kinetree->GetEntries());
  newtree->SetEntries(kinetree->GetEntries());
  // flush to disk
  newtree->Write();
  newfile->Close();
  delete newfile;

  // clean
  if (oldfile) {
    oldfile->Close();
    delete oldfile;
  }
}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::MCEventHeader);
