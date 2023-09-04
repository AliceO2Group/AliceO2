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
///
/// This class associated tpc tracks with ideal laser track positions from the data base
/// The difference in z-Position, separately on the A- and C-Side, is then used to
/// calculate a drift velocity correction factor as well as the trigger offset.
/// A vector of mathced laser track IDs can be used to monitor the laser system alignment.
///
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef TPC_CalibLaserTracks_H_
#define TPC_CalibLaserTracks_H_

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/LaserTrack.h"
#include "DataFormatsTPC/LtrCalibData.h"

namespace o2::tpc
{

using o2::track::TrackPar;
using o2::track::TrackParCov;

struct TimePair {
  float x1{0.f};
  float x2{0.f};
  uint64_t time{0};
};

class CalibLaserTracks
{
 public:
  static constexpr size_t MinTrackPerSidePerTF = 50;

  CalibLaserTracks()
  {
    mLaserTracks.loadTracksFromFile();
    updateParameters();
  }

  CalibLaserTracks(const CalibLaserTracks& other) : mTriggerPos{other.mTriggerPos},
                                                    mBz{other.mBz},
                                                    mDriftV{other.mDriftV},
                                                    mZbinWidth{other.mZbinWidth},
                                                    mTFstart{other.mTFstart},
                                                    mTFend{other.mTFend},
                                                    mCalibDataTF{other.mCalibDataTF},
                                                    mCalibData{other.mCalibData},
                                                    mZmatchPairsTFA{other.mZmatchPairsTFA},
                                                    mZmatchPairsTFC{other.mZmatchPairsTFC},
                                                    mZmatchPairsA{other.mZmatchPairsA},
                                                    mZmatchPairsC{other.mZmatchPairsC},
                                                    mWriteDebugTree{other.mWriteDebugTree},
                                                    mFinalized{other.mFinalized}
  {
  }

  ~CalibLaserTracks() = default;

  /// process all tracks of one TF
  void fill(const gsl::span<const TrackTPC> tracks);

  /// process all tracks of one TF
  void fill(std::vector<TrackTPC> const& tracks);

  /// process single track
  void processTrack(const TrackTPC& track);

  /// try to associate track with ideal laser track
  /// \return laser track ID; -1 in case of no match
  int findLaserTrackID(TrackPar track, int side = -1);

  /// calculate phi of nearest laser rod
  static float getPhiNearbyLaserRod(const TrackPar& param, int side);

  /// check if param is closer to a laser rod than 1/4 of a sector width
  static bool hasNearbyLaserRod(const TrackPar& param, int side);

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
  /// at least numTFs with laser track candidate and MinTrackPerSidePerTF tracks per side per TF
  bool hasEnoughData(size_t numTFs = 1) const { return mCalibData.processedTFs >= numTFs && mZmatchPairsA.size() > MinTrackPerSidePerTF * numTFs && mZmatchPairsC.size() > MinTrackPerSidePerTF * numTFs; }

  /// number of associated laser tracks on both sides for all processed TFs
  size_t getMatchedPairs() const { return getMatchedPairsA() + getMatchedPairsC(); }

  /// number of associated laser tracks for all processed TFs on the A-Side
  size_t getMatchedPairsA() const { return mZmatchPairsA.size(); }

  /// number of associated laser tracks for all processed TFs on the C-Side
  size_t getMatchedPairsC() const { return mZmatchPairsC.size(); }

  /// number of associated laser tracks presently processed TFs on the A-Side
  size_t getMatchedPairsTFA() const { return mZmatchPairsTFA.size(); }

  /// number of associated laser tracks presently processed TFs on the C-Side
  size_t getMatchedPairsTFC() const { return mZmatchPairsTFC.size(); }

  /// time frame time of presently processed time frame
  /// should be called before calling processTrack(s)
  void setTFtimes(uint64_t tfStart, uint64_t tfEnd = 0)
  {
    mTFstart = tfStart;
    mTFend = tfEnd;
  }

  uint64_t getTFstart() const { return mTFstart; }
  uint64_t getTFend() const { return mTFend; }

  void setWriteDebugTree(bool write) { mWriteDebugTree = write; }
  bool getWriteDebugTree() const { return mWriteDebugTree; }

  /// extract DV correction and T0 offset
  TimePair fit(const std::vector<TimePair>& trackMatches) const;

  /// sort TimePoint vectors
  void sort(std::vector<TimePair>& trackMatches);

  /// drift velocity fit information for presently processed time frame
  const LtrCalibData& getCalibDataTF() { return mCalibDataTF; }

  /// drift velocity fit information for full data set
  const LtrCalibData& getCalibData() { return mCalibData; }

  /// name of the debug output tree
  void setDebugOutputName(std::string_view name) { mDebugOutputName = name; }

  void setVDriftRef(float v) { mDriftV = v; }

 private:
  int mTriggerPos{0};                                          ///< trigger position, if < 0 it treats it as the CE position
  float mBz{0.5};                                              ///< Bz field in Tesla
  float mDriftV{0};                                            ///< drift velocity used during reconstruction
  float mZbinWidth{0};                                         ///< width of a bin in us
  uint64_t mTFstart{0};                                        ///< start time of processed time frames
  uint64_t mTFend{0};                                          ///< end time of processed time frames
  LtrCalibData mCalibDataTF{};                                 ///< calibration data for single TF (debugging)
  LtrCalibData mCalibData{};                                   ///< final calibration data
  std::vector<TimePair> mZmatchPairsTFA;                       ///< ideal vs. mesured z positions in present time frame A-Side (debugging)
  std::vector<TimePair> mZmatchPairsTFC;                       ///< ideal vs. mesured z positions in present time frame C-Side (debugging)
  std::vector<TimePair> mZmatchPairsA;                         ///< ideal vs. mesured z positions assumulated over all time frames A-Side
  std::vector<TimePair> mZmatchPairsC;                         ///< ideal vs. mesured z positions assumulated over all time frames C-Side
  bool mWriteDebugTree{false};                                 ///< create debug output tree
  bool mFinalized{false};                                      ///< if the finalize method was already called
  std::string mDebugOutputName{"CalibLaserTracks_debug.root"}; ///< name of the debug output tree

  LaserTrackContainer mLaserTracks;                              //!< laser track data base
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream; //!< debug output streamer

  /// update reconstruction parameters
  void updateParameters();

  /// perform fits on the matched z-position pairs to extract the drift velocity correction factor and trigger offset
  void fillCalibData(LtrCalibData& calibData, const std::vector<TimePair>& pairsA, const std::vector<TimePair>& pairsC);

  ClassDefNV(CalibLaserTracks, 1);
};

} // namespace o2::tpc
#endif
