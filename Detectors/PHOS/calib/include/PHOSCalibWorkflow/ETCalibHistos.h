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

/// \class ETCalibHistos
/// \brief container to store calibration info for time and energy calibration
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Apr. 1, 2021
///
///

#ifndef PHOS_ETCALIBHISTOS_H
#define PHOS_ETCALIBHISTOS_H

#include <array>
#include <cstring>

namespace o2
{

namespace phos
{

class ETCalibHistos
{
 public:
  // Histogram kinds to be filled
  enum hnames { kReInvMassPerCell,
                kMiInvMassPerCell,
                kReInvMassNonlin,
                kMiInvMassNonlin,
                kTimeHGPerCell,
                kTimeLGPerCell,
                kTimeHGSlewing,
                kTimeLGSlewing };
  static constexpr int nChannels = 14336 - 1793; // 4 full modules -1/2
  static constexpr int offset = 1793;            // 1/2 full module
  // mgg histos
  static constexpr int nMass = 150.;
  static constexpr float massMax = 0.3;
  static constexpr float dm = massMax / nMass;
  // time histograms
  static constexpr int nTime = 200;
  static constexpr float timeMin = -100.e-9;
  static constexpr float timeMax = 100.e-9;
  static constexpr float dt = (timeMax - timeMin) / nTime;

  // pt
  static constexpr int npt = 200;
  static constexpr float ptMax = 20;
  static constexpr float dpt = ptMax / npt;

  /// \brief Constructor
  ETCalibHistos() = default;

  ETCalibHistos& operator=(const ETCalibHistos& other) = default;

  /// \brief Destructor
  virtual ~ETCalibHistos() = default;

  /// \brief Merge statistics in two containers
  /// \param other Another container to be added to current
  void merge(const ETCalibHistos* other)
  {
    for (int i = nChannels; --i;) {
      for (int j = nMass; --j;) {
        mReInvMassPerCell[i][j] += other->mReInvMassPerCell[i][j];
        mMiInvMassPerCell[i][j] += other->mMiInvMassPerCell[i][j];
      }
      for (int j = nTime; --j;) {
        mTimeHGPerCell[i][j] += other->mTimeHGPerCell[i][j];
        mTimeLGPerCell[i][j] += other->mTimeLGPerCell[i][j];
      }
    }
    for (int i = npt; --i;) {
      for (int j = nMass; --j;) {
        mReInvMassNonlin[i][j] += other->mReInvMassNonlin[i][j];
        mMiInvMassNonlin[i][j] += other->mMiInvMassNonlin[i][j];
      }
      for (int j = nTime; --j;) {
        mTimeHGSlewing[i][j] += other->mTimeHGSlewing[i][j];
        mTimeLGSlewing[i][j] += other->mTimeLGSlewing[i][j];
      }
    }
  }

  void fill(int kind, float x, float y)
  {
    if (kind == kReInvMassNonlin || kind == kMiInvMassNonlin) {
      int i = int(x / dm);
      int j = int(y / dpt);
      if (i < nMass && j < npt) {
        if (kind == kReInvMassNonlin) {
          mReInvMassNonlin[i][j]++;
        } else {
          mMiInvMassNonlin[i][j]++;
        }
      }
    }
    if (kind == kTimeHGSlewing || kind == kTimeLGSlewing) {
      int i = int((x - timeMin) / dt);
      int j = int(y / dpt);
      if (i >= 0 && i < nTime && j < npt) {
        if (kind == kTimeHGSlewing) {
          mTimeHGSlewing[i][j]++;
        } else {
          mTimeHGSlewing[i][j]++;
        }
      }
    }
  }
  void fill(int kind, int x, float y)
  {
    if (kind == kReInvMassPerCell || kind == kMiInvMassPerCell) {
      int j = int(y / dm);
      if (j < nMass) {
        if (kind == kReInvMassPerCell) {
          mReInvMassPerCell[x - offset][j]++;
        } else {
          mMiInvMassPerCell[x - offset][j]++;
        }
      }
    }
    if (kind == kTimeHGPerCell || kind == kTimeLGPerCell) {
      int j = int((y - timeMin) / dt);
      if (j >= 0 && j < nTime) {
        if (kind == kTimeHGPerCell) {
          mTimeHGPerCell[x - offset][j]++;
        } else {
          mTimeLGPerCell[x - offset][j]++;
        }
      }
    }
  }
  void reset()
  {
    memset(&mReInvMassPerCell, 0, sizeof(mReInvMassPerCell));
    memset(&mMiInvMassPerCell, 0, sizeof(mMiInvMassPerCell));
    memset(&mReInvMassNonlin, 0, sizeof(mReInvMassNonlin));
    memset(&mMiInvMassNonlin, 0, sizeof(mMiInvMassNonlin));
    memset(&mTimeHGPerCell, 0, sizeof(mTimeHGPerCell));
    memset(&mTimeLGPerCell, 0, sizeof(mTimeLGPerCell));
    memset(&mTimeHGSlewing, 0, sizeof(mTimeHGSlewing));
    memset(&mTimeLGSlewing, 0, sizeof(mTimeLGSlewing));
  }

 public:
  std::array<std::array<float, nMass>, nChannels> mReInvMassPerCell; ///< inv mass per cell
  std::array<std::array<float, nMass>, nChannels> mMiInvMassPerCell; ///< inv mass per cell
  std::array<std::array<float, npt>, nMass> mReInvMassNonlin;        ///< inv mass vs pT
  std::array<std::array<float, npt>, nMass> mMiInvMassNonlin;        ///< inv mass vs pT
  std::array<std::array<float, nTime>, nChannels> mTimeHGPerCell;    ///< time per cell
  std::array<std::array<float, nTime>, nChannels> mTimeLGPerCell;    ///< time per cell
  std::array<std::array<float, npt>, nTime> mTimeHGSlewing;          ///< time vs pT
  std::array<std::array<float, npt>, nTime> mTimeLGSlewing;          ///< time vs pT

  ClassDef(ETCalibHistos, 1);
};

} // namespace phos

} // namespace o2
#endif
