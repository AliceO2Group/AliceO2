// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_PHOS_CALIB_COLLECTOR_H
#define O2_PHOS_CALIB_COLLECTOR_H

/// @file   PHOSCalibCollectorSpec.h
/// @brief  Device to collect information for PHOS energy and time calibration.

#include "Framework/Task.h"
#include "Framework/ProcessingContext.h"
#include "Framework/WorkflowSpec.h"
#include "PHOSReconstruction/FullCluster.h"
#include "PHOSCalib/CalibParams.h"
#include "PHOSCalib/BadChannelMap.h"
#include "PHOSBase/Geometry.h"

#include <TLorentzVector.h>
#include <TH2.h>
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
    uint32_t mAddress : 14;   ///< Bits  0 - 13: Hardware address
    uint32_t mAdcAmp : 10;    ///< Bits 14 - 23: ADC counts
    uint32_t mHgLg : 1;       ///< Bit  24: LG/HG
    uint32_t mBadChannel : 1; ///< Bit  25: Bad channel status
    uint32_t mCluster : 6;    ///< Bits 26-32: index of cluster in event
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

// For real/mixed distribution calculation
class RungBuffer
{
 public:
  RungBuffer() = default;
  ~RungBuffer() = default;

  short size()
  {
    if (mFilled) {
      return kBufferSize;
    } else {
      return mCurrent;
    }
  }
  void addEntry(TLorentzVector& v)
  {
    mBuffer[mCurrent] = v;
    mCurrent++;
    if (mCurrent >= kBufferSize) {
      mFilled = true;
      mCurrent -= kBufferSize;
    }
  }
  const TLorentzVector getEntry(short index)
  {
    //get entry from (mCurrent-1) corresponding to index=size()-1 down to size
    if (mFilled) {
      index += mCurrent;
    }
    index = index % kBufferSize;
    return mBuffer[index];
  }
  //mark that next added entry will be from next event
  void startNewEvent() { mStartCurrentEvent = mCurrent; }

  bool isCurrentEvent(short index)
  {
    if (mCurrent >= mStartCurrentEvent) {
      return (index >= mStartCurrentEvent && index < mCurrent);
    } else {
      return (index >= mStartCurrentEvent || index < mCurrent);
    }
  }

 private:
  static constexpr short kBufferSize = 100;        ///< Total size of the buffer
  std::array<TLorentzVector, kBufferSize> mBuffer; ///< buffer
  bool mFilled = false;                            ///< if buffer fully filled
  short mCurrent = 0;                              ///< where next object will be added
  short mStartCurrentEvent = 0;                    ///< start of current event
};

class PHOSCalibCollector : public o2::framework::Task
{

  //Histogram kinds to be filled
  enum hnames { kReInvMassPerCell,
                kMiInvMassPerCell,
                kReInvMassNonlin,
                kMiInvMassNonlin,
                kTimeHGPerCell,
                kTimeLGPerCell,
                kTimeHGSlewing,
                kTimeLGSlewing };

 public:
  PHOSCalibCollector() = default;
  PHOSCalibCollector(short mode) : mMode(mode) {}

  ~PHOSCalibCollector() override = default;

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  /// Scan input clusters fill histograms and prepare calibDigits
  void scanClusters(o2::framework::ProcessingContext& pc);

  /// Read and scan previously stored calibDigits
  void readDigits();

  //construct new cluster from digits
  //no need to reclusterize as digits wrote in order.
  bool nextCluster(std::vector<uint32_t>::const_iterator digIt, std::vector<uint32_t>::const_iterator digEnd, FullCluster& clu, bool& isNewEvent);

  /// Fill histograns for one cluster
  void fillTimeMassHisto(const FullCluster& clu);

  /// Check cluster properties and bad map
  bool checkCluster(const FullCluster& clu);

  /// Write selected digits
  void writeOutputs();

  /// Evaluate calibrations from inv masses
  void calculateCalibrations() {}

  // Compare calibration to previous  TODO!
  void compareCalib() {}

  //Check if cluster OK
  bool checkCluster(FullCluster& clu);

  // Send results to (temporary of final CCDB) and to QC
  void sendOutput(DataAllocator& out);

 private:
  static constexpr short kMaxCluInEvent = 64; /// maximal number of clusters per event to separate digits from them (6 bits in digit map)
  short mMode = 0;                            /// modes 0: collect new data; 1: re-scan data with new calibration; 2: produce new calibration
  uint32_t mEvBC = 0;
  uint32_t mEvOrbit = 0;
  uint32_t mEvent = 0;
  float mPtMin = 1.5; /// minimal energy to fill inv. mass histo
  float mEminHGTime = 1.5;
  float mEminLGTime = 5.;
  std::vector<uint32_t> mDigits;             /// list of calibration digits to fill
  std::vector<TH2F> mHistos;                 /// list of histos to fill
  std::unique_ptr<RungBuffer> mBuffer;       /// Buffer for current and previous events
  std::unique_ptr<CalibParams> mCalibParams; /// Final calibration object
  std::unique_ptr<BadChannelMap> mBadMap;    /// Final calibration object
  Geometry* mGeom;                           /// Pointer to PHOS singleton geometry
  TVector3 mVertex;
  std::string mdigitsfilename = "";
  std::string mhistosfilename = "";
  std::string mdigitsfilelist = "";
  std::string mfilenameCalib = "";
  ClassDefNV(PHOSCalibCollector, 1);
};

o2::framework::DataProcessorSpec getPHOSCalibCollectorDeviceSpec(int mode);

} // namespace phos
} // namespace o2

#endif
