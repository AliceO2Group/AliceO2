// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CA/TrackerTask.h
/// \brief Definition of the ITS CA tracker task

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_VERTEXERTASK_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_VERTEXERTASK_H_

#include "FairTask.h"

#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/vertexer/Vertexer.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class VertexerTask : public FairTask
{
  public:
    VertexerTask(/*bool useMCTruth=true*/);
    ~VertexerTask() override;
    InitStatus Init() override;
    void Exec(Option_t* option) override;
  private:
    // Event mEvent;            ///< CA event
    // Vertexer<false> mVertexer
    ClassDefOverride(VertexerTask, 1)
};

}
}
}
#endif /* O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_VERTEXERTASK_H__ */
