// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TurnOnHistos
/// \brief transient CCDB container to collect histos for calculation of trigger maps and turn-on curves
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Apr. 1, 2021
///
///

#ifndef PHOS_TURNONHISTOS_H
#define PHOS_TURNONHISTOS_H

#include <bitset>
#include <array>
#include "TObject.h"

namespace o2
{

namespace phos
{

class TurnOnHistos
{
 public:
  //class to collect statistics to calculate trigger turn-on curves and trigger bad maps
  static constexpr short NCHANNELS = 3136; ///< Number of trigger channels
  static constexpr short NDDL = 14;        ///< Number of DDLs
  static constexpr short Npt = 200;        ///< Number of bins in pt distribution
  static constexpr float dpt = 0.1;        ///< bin width

  /// \brief Constructor
  TurnOnHistos() = default;

  TurnOnHistos& operator=(const TurnOnHistos& other) = default;

  /// \brief Destructor
  ~TurnOnHistos() = default;

  /// \brief Merge statistics in two containers
  /// \param other Another container to be added to current
  void merge(TurnOnHistos& other);

  /// \brief Fill spectum of all clusters
  /// \param ddl ddl ID
  /// \param e cluster energy
  void fillTotSp(short ddl, float e)
  {
    short bin = e / dpt;
    if (bin < Npt) {
      mTotSp[ddl][bin]++;
    }
  }

  /// \brief Fill spectum of clusters fired trigger
  /// \param ddl ddl ID
  /// \param e cluster energy
  void fillFiredSp(short ddl, float e)
  {
    short bin = e / dpt;
    if (bin < Npt) {
      mTrSp[ddl][bin]++;
    }
  }

  /// \brief Collects entries in good map
  /// \param bitset with channels fired in event
  void fillFiredMap(const std::bitset<NCHANNELS>& bs)
  {
    for (short i = NCHANNELS; --i;) {
      if (bs[i]) {
        mGoodMap[i]++;
      }
    }
  }

  /// \brief Collects entries in noisy map
  /// \param bitset with channels fired in event
  void fillNoisyMap(const std::bitset<NCHANNELS>& bs)
  {
    for (short i = NCHANNELS; --i;) {
      if (bs[i]) {
        mNoisyMap[i]++;
      }
    }
  }

  //getters now
  const std::array<float, Npt>& getTotSpectrum(short ddl) const { return mTotSp[ddl]; }
  const std::array<float, Npt>& getTrSpectrum(short ddl) const { return mTrSp[ddl]; }
  const std::array<float, NCHANNELS>& getGoodMap() const { return mGoodMap; }
  const std::array<float, NCHANNELS>& getNoisyMap() const { return mNoisyMap; }

 private:
  std::array<float, NCHANNELS> mGoodMap;           ///< Container to collect entries in good map
  std::array<float, NCHANNELS> mNoisyMap;          ///< Container to collect entries in noisy map
  std::array<std::array<float, Npt>, NDDL> mTotSp; ///< Spectrum of all clusters
  std::array<std::array<float, Npt>, NDDL> mTrSp;  ///< Spectrum of fired trigger cl.

  ClassDefNV(TurnOnHistos, 1);
};

} // namespace phos

} // namespace o2
#endif
