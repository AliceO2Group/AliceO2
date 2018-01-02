// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Vertexer.h
/// \brief
/// \author matteo.concas@cern.ch
/// \author maximiliano.puccio@cern.ch

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_

#include <vector>
#include "ITSReconstruction/CA/IndexTable.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class Event;
class Vertexer final
{
public:
  explicit Vertexer(const Event&);
  virtual ~Vertexer();
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

protected:
  void computeVertex(); // dummy, atm.
  const Event& mEvent;
  std::vector<int> mUsedClustersTable;
  std::array<std::vector<Cluster>, 3> mClusters;
  std::array<IndexTable, Constants::ITS::TrackletsPerRoad> mIndexTables;
};
}
}
}

#endif /* O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_ */
