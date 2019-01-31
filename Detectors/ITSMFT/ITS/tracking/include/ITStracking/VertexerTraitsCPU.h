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
/// \file VertexerTraitsCPU.h
/// \brief
///

#ifndef VERTEXERTRAITSCPU_H_
#define VERTEXERTRAITSCPU_H_

#include <vector>
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ClusterLines.h"

namespace o2
{
namespace ITS
{

class VertexerTraitsCPU : public VertexerTraits
{
 public:
  VertexerTraitsCPU();
  // ~VertexerTraitsCPU();

  void findLayerTracklets(const bool useMCLabels) final;
  void findLayerVertices() final;

 protected:
  std::vector<Line> mTracklets;
};
}
}

#endif /* VERTEXERTRAITCPUS_H_ */