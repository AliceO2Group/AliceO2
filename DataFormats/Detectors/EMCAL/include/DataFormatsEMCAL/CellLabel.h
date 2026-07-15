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

#ifndef ALICEO2_EMCAL_CELLLABEL_H_
#define ALICEO2_EMCAL_CELLLABEL_H_


#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace o2::emcal
{

/// \class CellLabel
/// \brief cell class for MC particle IDs and their respective amplitude fraction
/// \ingroup EMCALDataFormat
/// \author Marvin Hemmer <marvin.hemmer@cern.ch>, Goethe university Frankfurt
/// \since December 13, 2023
///

class CellLabel
{
 public:
  // CellLabel() = default;

  /// \brief Constructor using std::vector by moving NOT copying
  /// \param labels list of mc labels
  /// \param amplitudeFractions list of amplitude fractions
  CellLabel(std::vector<int> labels, std::vector<float> amplitudeFractions);

  /// \brief Constructor using std::span
  /// \param labels list of mc labels
  /// \param amplitudeFractions list of amplitude fractions
  CellLabel(std::span<const int> labels, std::span<const float> amplitudeFractions);

  // ~CellLabel() = default;
  // CellLabel(const CellLabel& clus) = default;
  // CellLabel& operator=(const CellLabel& source) = default;

  /// \brief Getter of label size
  /// \param index index which label to get
  [[nodiscard]] size_t GetLabelSize() const { return mLabels.size(); }

  /// \brief Getter for label
  /// \param index index which label to get
  [[nodiscard]] int32_t GetLabel(size_t index) const { return mLabels[index]; }

  /// \brief Getter for labels
  [[nodiscard]] std::vector<int32_t> GetLabels() const { return mLabels; }

  /// \brief Getter for amplitude fraction
  /// \param index index which amplitude fraction to get
  [[nodiscard]] float GetAmplitudeFraction(size_t index) const { return mAmplitudeFraction[index]; }

  /// \brief Getter for amplitude fractions
  [[nodiscard]] std::vector<float> GetAmplitudeFractions() const { return mAmplitudeFraction; }

  /// \brief Getter for label with leading amplitude fraction
  [[nodiscard]] int32_t GetLeadingMCLabel() const;

 protected:
  std::vector<int32_t> mLabels;          ///< List of MC particles that generated the cluster, ordered in deposited energy.
  std::vector<float> mAmplitudeFraction; ///< List of the fraction of the cell energy coming from a MC particle. Index aligns with mLabels!
};

} // namespace o2::emcal
#endif // ALICEO2_EMCAL_CELLLABEL_H_
