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

/// \file CalibLaserTracks.h
/// \brief calibration using laser tracks
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef TPC_CalibLaserTracks_H_
#define TPC_CalibLaserTracks_H_

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/LaserTrack.h"

namespace o2::tpc
{

using o2::track::TrackPar;
using o2::track::TrackParCov;

struct TimePair {
  float x1{0.f};
  float x2{0.f};
  float time{0.f};
};

class CalibLaserTracks
{
 public:
  CalibLaserTracks()
  {
    mLaserTracks.loadTracksFromFile();
    updateParameters();
  }

  CalibLaserTracks(const CalibLaserTracks& other) : mTriggerPos{other.mTriggerPos},
                                                    mBz{other.mBz},
                                                    mDriftV{other.mDriftV},
                                                    mZbinWidth{other.mZbinWidth},
                                                    mTFtime{other.mTFtime},
                                                    mDVall{other.mDVall},
                                                    mZmatchPairsTF{other.mZmatchPairsTF},
                                                    mZmatchPairs{other.mZmatchPairs},
                                                    mDVperTF{other.mDVperTF},
                                                    mWriteDebugTree{other.mWriteDebugTree}
  {
  }

  ~CalibLaserTracks() = default;

  void fill(const gsl::span<const TrackTPC> tracks);
  void fill(std::vector<TrackTPC> const& tracks);
  void processTrack(const TrackTPC& track);

  int findLaserTrackID(TrackPar track, int side = -1);
  static float getPhiNearbySectorEdge(const TrackPar& param);
  static float getPhiNearbyLaserRod(const TrackPar& param, int side);

  /// trigger position for track z position correction
  void setTriggerPos(int triggerPos) { mTriggerPos = triggerPos; }

  /// merge data with other calibration object
  void merge(const CalibLaserTracks* other);

  /// End processing of this TF
  void endTF();

  /// Finalize full processing
  void finalize();

  /// print information
  void print() const;

  /// check amount of data (to be improved)
  bool hasEnoughData() const { return mZmatchPairs.size() > 100; }

  /// number of associated laser tracks
  size_t getMatchedPairs() const { return mZmatchPairs.size(); }

  /// time frame time of presently processed time frame
  /// should be called before calling processTrack(s)
  void setTFtime(float tfTime) { mTFtime = tfTime; }
  float getTFtime() const { return mTFtime; }

  void setWriteDebugTree(bool write) { mWriteDebugTree = write; }
  bool getWriteDebugTree() const { return mWriteDebugTree; }

  /// extract DV correction and T0 offset
  TimePair fit(const std::vector<TimePair>& trackMatches) const;

  /// sort TimePoint vectors
  void sort(std::vector<TimePair>& trackMatches);

  /// drift velocity fit information for full data set
  const TimePair& getDVall() { return mDVall; }

 private:
  int mTriggerPos{0};                                            ///< trigger position, if < 0 it treats it as the CE position
  float mBz{0.5};                                                ///< Bz field in kGaus
  float mDriftV{0};                                              ///< drift velocity used during reconstruction
  float mZbinWidth{0};                                           ///< width of a bin in us
  float mTFtime{0};                                              ///< time of present TF
  TimePair mDVall{};                                             ///< fit result over all accumulated data
  std::vector<TimePair> mZmatchPairsTF;                          ///< ideal vs. mesured z poitions for associated laser tracks in present time frame
  std::vector<TimePair> mZmatchPairs;                            ///< ideal vs. mesured z poitions for associated laser tracks assumulated over all time frames
  std::vector<TimePair> mDVperTF;                                ///< drift velocity and time offset per TF
  bool mWriteDebugTree{false};                                   ///< create debug output tree
  LaserTrackContainer mLaserTracks;                              //!< laser track data base
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream; //!< debug output streamer

  /// update reconstruction parameters
  void updateParameters();

  ClassDefNV(CalibLaserTracks, 1);
};

} // namespace o2::tpc
#endif
