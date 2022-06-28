// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <gsl/gsl>
#ifndef ALICEO2_CPV_FULLCLUSTER_H_
#define ALICEO2_CPV_FULLCLUSTER_H_

#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace cpv
{
/// \class FullCluster
/// \brief CPV cluster implementation

class FullCluster : public Cluster
{

  using Label = o2::MCCompLabel;

 public:
  struct CluElement {
    short absId;
    float energy;
    float localX;
    float localZ;
    int label;
    CluElement(short a, float e, float x, float z, int l) : absId(a), energy(e), localX(x), localZ(z), label(l) {}
  };

  FullCluster() = default;
  FullCluster(short digitAbsId, float energy, int label);

  ~FullCluster() = default;

  /// \brief Method to add digit to a cluster
  /// \param digit being added, energy of this digit, may be smaller than full due to everlap
  void addDigit(short digitAbsId, float energy, int label);

  void evalAll();

  // Get index of a digit with i
  short getDigitAbsId(Int_t i) const { return mElementList.at(i).absId; }

  const std::vector<CluElement>* getElementList() const { return &mElementList; }

  // Counts local maxima and returns their positions
  char getNumberOfLocalMax(gsl::span<int> maxAt) const;

  void purify(); // Removes digits below threshold

 protected:
  void evalLocalPosition(); // computes the position in the CPV module

 private:
  std::vector<CluElement> mElementList; //!  Transient Array of digits

  ClassDefNV(FullCluster, 1);
};
} // namespace cpv
} // namespace o2

#endif
