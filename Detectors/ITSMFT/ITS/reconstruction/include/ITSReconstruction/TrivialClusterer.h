// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrivialClusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_TRIVIALCLUSTERER_H
#define ALICEO2_ITS_TRIVIALCLUSTERER_H

#include "Rtypes.h"  // for TrivialClusterer::Class, Double_t, ClassDef, etc
#include "ITSMFTBase/GeometryTGeo.h"

namespace o2 {
  namespace ITSMFT {
    class Digit;
    class Cluster;
  }
}

namespace o2
{
class MCCompLabel;
namespace dataformats
{
  template<typename T>
  class MCTruthContainer;
}
namespace ITS
{
  class TrivialClusterer
{
  using Digit = o2::ITSMFT::Digit;
  using Cluster = o2::ITSMFT::Cluster;
  using Label = o2::MCCompLabel;
 public:
  TrivialClusterer();
  ~TrivialClusterer();

  TrivialClusterer(const TrivialClusterer&) = delete;
  TrivialClusterer& operator=(const TrivialClusterer&) = delete;

  /// Steer conversion of points to digits
  /// @param points Container with ITS points
  /// @return digits container
  void process(const std::vector<Digit>* digits, std::vector<Cluster>* clusters);
  // provide the common ITSMFT::GeometryTGeo to access matrices
  void setGeometry(const o2::ITSMFT::GeometryTGeo* gm) { mGeometry = gm;}
  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel> *truth) {
    mClsLabels = truth;
  }

 protected:
  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr;    ///< ITS OR MFT upgrade geometry
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mClsLabels = nullptr; // Cluster MC labels

};
}
}

#endif /* ALICEO2_ITS_TRIVIALCLUSTERER_H */
