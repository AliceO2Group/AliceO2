// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrivialVertexer.h
/// \brief Definition of the ITS trivial vertex finder
#ifndef ALICEO2_ITS_TRIVIALVERTEXER_H
#define ALICEO2_ITS_TRIVIALVERTEXER_H

#include <array>

#include "Rtypes.h" // for TrivialVertexer::Class, Double_t, ClassDef, etc

class TFile;
class TTree;
class FairMCEventHeader;

namespace o2
{
namespace ITSMFT
{
class Cluster;
}
} // namespace o2

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}
namespace ITS
{
class TrivialVertexer
{
  using Cluster = o2::ITSMFT::Cluster;
  using Label = o2::MCCompLabel;

 public:
  TrivialVertexer();
  ~TrivialVertexer();

  TrivialVertexer(const TrivialVertexer&) = delete;
  TrivialVertexer& operator=(const TrivialVertexer&) = delete;

  Bool_t openInputFile(const Char_t*);

  void process(const std::vector<Cluster>& clusters, std::vector<std::array<Double_t, 3>>& vertices);
  void setMCTruthContainer(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truth) { mClsLabels = truth; }

 private:
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; // Cluster MC labels

  TFile* mFile = nullptr;
  TTree* mTree = nullptr;
  FairMCEventHeader* mHeader = nullptr;
};
} // namespace ITS
} // namespace o2

#endif /* ALICEO2_ITS_TRIVIALVERTEXER_H */
