// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_DIGIT_H_
#define ALICEO2_CPV_DIGIT_H_

#include "CommonDataFormat/TimeStamp.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CPVBase/Hit.h"

namespace o2
{

namespace cpv
{
/// \class CPVDigit
/// \brief CPV digit implementation

using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{

  using Label = o2::MCCompLabel;

 public:
  static constexpr int kTimeGate = 25; // Time in ns between digits to be added as one signal.
                                       // Should it be readout time (6000 ns???): to be tested

  Digit() = default;

  /// \brief Main Digit constructor
  /// \param cell absId of a cell, amplitude energy deposited in a cell, time time measured in cell, label label of a
  /// particle in case of MC \return constructed Digit
  Digit(int cell, float amplitude, float time, int label);

  /// \brief Digit constructor from Hit
  /// \param CPV Hit
  /// \return constructed Digit
  Digit(Hit hit, int label);

  ~Digit() = default; // override

  /// \brief Replace content of this digit with new one, from hit
  /// \param CPV Hit
  /// \return
  void FillFromHit(Hit hit);

  /// \brief Comparison oparator, based on time and absId
  /// \param another CPV Digit
  /// \return result of comparison: first time, if time same, then absId
  bool operator<(const Digit& other) const;
  /// \brief Comparison oparator, based on time and absId
  /// \param another CPV Digit
  /// \return result of comparison: first time, if time same, then absId
  bool operator>(const Digit& other) const;
  /// \brief Check, if one can add two digits
  /// \param another CPV Digit
  /// \return true if time stamps are same and absId are same
  bool canAdd(const Digit other) const;
  /// \brief if addable, adds energy and list of primaries.
  /// \param another CPV Digit
  /// \return digit with sum of energies and longer list of primaries
  Digit& operator+=(const Digit& other); //

  /// \brief Absolute sell id
  int getAbsId() const { return mAbsId; }
  void setAbsId(int cellId) { mAbsId = cellId; }

  /// \brief Energy deposited in a cell
  float getAmplitude() const { return mAmplitude; }
  void setAmplitude(float amplitude) { mAmplitude = amplitude; }

  /// \brief index of entry in MCLabels array
  /// \return ndex of entry in MCLabels array
  int getLabel() const { return mLabel; }

  void PrintStream(std::ostream& stream) const;

 private:
  // friend class boost::serialization::access;

  int mAbsId = 0;       ///< pad index (absolute pad ID)
  int mLabel = -1;      ///< Index of the corresponding entry/entries in the MC label array
  float mAmplitude = 0; ///< Amplitude

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace cpv
} // namespace o2
#endif
