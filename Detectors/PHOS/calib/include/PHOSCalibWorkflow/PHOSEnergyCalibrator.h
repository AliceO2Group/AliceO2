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

#ifndef O2_CALIBRATION_PHOSENERGY_CALIBRATOR_H
#define O2_CALIBRATION_PHOSENERGY_CALIBRATOR_H

/// @file   PHOSEnergyCalibtor.h
/// @brief  Device to collect energy and time PHOS energy and time calibration.

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cluster.h"
#include "PHOSCalibWorkflow/RingBuffer.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "PHOSBase/Geometry.h"
#include "PHOSCalibWorkflow/ETCalibHistos.h"

#include <TLorentzVector.h>
#include <TVector3.h>
#include <array>

using namespace o2::framework;

namespace o2
{
namespace phos
{

// structure used to store digit info for re-calibration
union CalibDigit {
  uint32_t mDataWord;
  struct {
    uint32_t mAddress : 14; ///< Bits  0 - 13: Hardware address
    uint32_t mAdcAmp : 10;  ///< Bits 14 - 23: ADC counts
    uint32_t mHgLg : 1;     ///< Bit  24: LG/HG
    uint32_t mCluster : 7;  ///< Bits 25-32: index of cluster in event
  };
};
// Event header for energy calibraton. Allow accessing external info with vertex position and collision time
union EventHeader {
  uint32_t mDataWord;
  struct {
    uint32_t mMarker : 14; ///< Bits  0 - 13: non-existing address to separate events 16383
    uint32_t mBC : 18;     ///< Bits 14-32: event BC (16bit in InterationRecord. Orbit (32bit) will be stored in next word)
  };
};

class PHOSEnergySlot
{

 public:
  static constexpr short kMaxCluInEvent = 128; /// maximal number of clusters per event to separate digits from them (7 bits in digit map)

  PHOSEnergySlot();
  PHOSEnergySlot(const PHOSEnergySlot& other);

  ~PHOSEnergySlot() = default;

  void print() const;
  void fill(const gsl::span<const Cluster>& clusters, const gsl::span<const CluElement>& cluelements, const gsl::span<const TriggerRecord>& cluTR);
  void fill(const gsl::span<const Cluster>& /*c*/){}; // not used
  void merge(const PHOSEnergySlot* /*prev*/) {}       // not used
  void clear();

  const ETCalibHistos* getCollectedHistos() const { return mHistos.get(); }
  const std::vector<uint32_t>& getCollectedDigits() const { return mDigits; }

  void setRunStartTime(long tf) { mRunStartTime = tf; }
  void setCalibration(const CalibParams* c) { mCalibParams = c; }
  void setBadMap(const BadChannelsMap* map) { mBadMap = map; }
  void setCuts(float ptMin, float eminHGTime, float eminLGTime, float eDigMin, float eCluMin)
  {
    mPtMin = ptMin;
    mEminHGTime = eminHGTime;
    mEminLGTime = eminLGTime;
    mDigitEmin = eDigMin;
    mClusterEmin = eCluMin;
  }

 private:
  void fillTimeMassHisto(const Cluster& clu, const gsl::span<const CluElement>& cluelements);
  bool checkCluster(const Cluster& clu);

  long mRunStartTime = 0;                 /// start time of the run (sec)
  std::unique_ptr<RingBuffer> mBuffer;    /// Buffer for current and previous events
  const CalibParams* mCalibParams;        /// current calibration
  const BadChannelsMap* mBadMap;          /// current bad map
  Geometry* mGeom;                        /// Pointer to PHOS singleton geometry
  std::unique_ptr<ETCalibHistos> mHistos; /// final histograms
  uint32_t mEvBC = 0;
  uint32_t mEvOrbit = 0;
  uint32_t mEvent = 0;
  float mPtMin = 1.5;            /// minimal pi0 pt to fill inv. mass histo
  float mEminHGTime = 1.5;       /// minimal cell energy to fill HG time histo
  float mEminLGTime = 5.;        /// minimal cell energy to fill LG time histo
  float mDigitEmin = 0.05;       /// minimal energy of stored digits
  float mClusterEmin = 0.4;      /// minimal energy of cluster which digits will be stored
  std::vector<uint32_t> mDigits; /// list of calibration digits
};

class PHOSEnergyCalibrator final : public o2::calibration::TimeSlotCalibration<o2::phos::Cluster, o2::phos::PHOSEnergySlot>
{
  using Slot = o2::calibration::TimeSlot<o2::phos::PHOSEnergySlot>;

 public:
  PHOSEnergyCalibrator();

  bool hasEnoughData(const Slot& slot) const final { return true; } // no need to merge Slots
  void initOutput() final {}
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  bool process(uint64_t tf, const gsl::span<const Cluster>& clusters, const gsl::span<const CluElement>& cluelements, const gsl::span<const TriggerRecord>& cluTR, std::vector<uint32_t>& outputDigits);

  void endOfStream();

  const ETCalibHistos* getCollectedHistos() const { return mHistos.get(); }
  void setCalibration(const CalibParams* c) { mCalibParams = c; }
  void setBadMap(const BadChannelsMap* map) { mBadMap = map; }
  void setCuts(float ptMin, float eminHGTime, float eminLGTime, float eDigMin, float eCluMin)
  {
    mPtMin = ptMin;
    mEminHGTime = eminHGTime;
    mEminLGTime = eminLGTime;
    mDigitEmin = eDigMin;
    mClusterEmin = eCluMin;
  }

 private:
  bool calculateCalibrations();

 private:
  long mRunStartTime = 0;                    /// start time of the run (ns)
  float mPtMin = 1.5;                        /// minimal pi0 pt to fill inv. mass histo
  float mEminHGTime = 1.5;                   /// minimal cell energy to fill HG time histo
  float mEminLGTime = 5.;                    /// minimal cell energy to fill LG time histo
  float mDigitEmin = 0.05;                   /// minimal energy of stored digits
  float mClusterEmin = 0.4;                  /// minimal energy of cluster which digits will be stored
  const CalibParams* mCalibParams = nullptr; /// Current calibration object
  const BadChannelsMap* mBadMap = nullptr;   /// Current BadMap
  std::unique_ptr<ETCalibHistos> mHistos;    /// final histograms
};

} // namespace phos
} // namespace o2

#endif
