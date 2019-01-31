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

namespace o2
{
namespace ITS
{

Vertexer::Vertexer(VertexerTraits* traits)
{
  mTraits = traits;
}

void Vertexer::clustersToVertices(ROframe& event, std::ostream& timeBenchmarkOutputStream)
{
  ROframe* eventptr = &event;
  float total{ 0.f };
  total += evaluateTask(&Vertexer::initialiseVertexer, "Vertexer initialisation", timeBenchmarkOutputStream, eventptr);
  total += evaluateTask(&Vertexer::findTracklets, "Tracklet finding", timeBenchmarkOutputStream, false);
  total += evaluateTask(&Vertexer::findVertices, "Vertex finding", timeBenchmarkOutputStream);
}

void Vertexer::findTracklets(const bool useMCLabels)
{
  mTraits->computeTracklets(useMCLabels);
}

void Vertexer::findVertices()
{
  mTraits->computeVertices();
}

} // namespace ITS
} // namespace o2
