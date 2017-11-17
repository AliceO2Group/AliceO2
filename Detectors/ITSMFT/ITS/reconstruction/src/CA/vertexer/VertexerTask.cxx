// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Tracker.cxx
/// \brief 
///


#include "ITSReconstruction/CA/vertexer/VertexerTask.h"
#include <iostream>
#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::ITS::CA::VertexerTask)

namespace o2
{
namespace ITS
{
namespace CA
{
VertexerTask::VertexerTask()
{
  std::cout<<"Monte Carlo666!!"<<std::endl;
};
VertexerTask::~VertexerTask()
{
  std::cout<<"Distruggere!"<<std::endl;
};

InitStatus VertexerTask::Init()
{
  std::cout<<"Init Function!!"<<std::endl;
  return kSUCCESS;
}

void VertexerTask::Exec(Option_t* option)
{
  std::cout<<"Exec Function!!"<<std::endl;
}

}
}
}