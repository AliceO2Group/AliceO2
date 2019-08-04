// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the PHOS cluster finder
#ifndef ALICEO2_PHOS_CLUSTERER_H
#define ALICEO2_PHOS_CLUSTERER_H

#include "Rtypes.h" // for Clusterer::Class, Double_t, ClassDef, etc

namespace o2
{
namespace phos
{
class Digit;
class Cluster;
class Geometry;

class Clusterer
{
 public:
  Clusterer() = default;
  ~Clusterer() = default;

  void process(const std::vector<Digit>* digits, std::vector<Cluster>* clusters);
  void MakeClusters(const std::vector<Digit>* digits, std::vector<Cluster>* clusters);
  void EvalCluProperties(const std::vector<Digit>* digits, std::vector<Cluster>* clusters);

 protected:
  Geometry* mPHOSGeom = nullptr; ///< PHOS geometry
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_ITS_TRIVIALCLUSTERER_H */
