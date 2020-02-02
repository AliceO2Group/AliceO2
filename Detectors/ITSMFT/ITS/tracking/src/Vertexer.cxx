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
/// \file Vertexer.cxx
/// \brief
///

#include "ITStracking/Vertexer.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/VertexerTraits.h"

namespace o2
{
namespace its
{

Vertexer::Vertexer(VertexerTraits* traits)
{
  mTraits = traits;
}

float Vertexer::clustersToVertices(ROframe& event, const bool useMc, std::ostream& timeBenchmarkOutputStream)
{
  ROframe* eventptr = &event;
  float total{0.f};
  total += evaluateTask(&Vertexer::initialiseVertexer, "Vertexer initialisation", timeBenchmarkOutputStream, eventptr);
  total += evaluateTask(&Vertexer::findTracklets, "Tracklet finding", timeBenchmarkOutputStream);
#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (useMc) {
    total += evaluateTask(&Vertexer::filterMCTracklets, "MC tracklets filtering", timeBenchmarkOutputStream);
  }
#endif
  total += evaluateTask(&Vertexer::validateTracklets, "Adjacent tracklets validation", timeBenchmarkOutputStream);
  total += evaluateTask(&Vertexer::findVertices, "Vertex finding", timeBenchmarkOutputStream);

  return total;
}

void Vertexer::findVertices()
{
  mTraits->computeVertices();
}

void Vertexer::findHistVertices()
{
  mTraits->computeHistVertices();
}

} // namespace its
} // namespace o2
