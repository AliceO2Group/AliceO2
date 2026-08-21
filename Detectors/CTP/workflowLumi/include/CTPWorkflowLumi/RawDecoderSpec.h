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

#ifndef O2_CTP_RAWDECODER_H
#define O2_CTP_RAWDECODER_H

#include <vector>
#include <deque>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"
#include "CTPReconstruction/RawDataDecoder.h"
#include "DataFormatsParameters/AggregatedRunInfo.h"

namespace o2
{
namespace ctp
{
namespace reco_workflow
{

/// \class RawDecoderSpec
/// \brief Coverter task for Raw data to CTP digits
/// \author Roman Lietava from CPV example
///
class RawDecoderSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param propagateMC If true the MCTruthContainer is propagated to the output
  RawDecoderSpec(bool digits, bool lumi) : mDoDigits(digits), mDoLumi(lumi) {}
  /// \brief Destructor
  ~RawDecoderSpec() override = default;
  /// \brief Initializing the RawDecoderSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output HW errors: {"CTP", "RAWHWERRORS", 0, Lifetime::Timeframe} -later
  void run(framework::ProcessingContext& ctx) final;
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  /// \brief Compute per BC luminosity from the interaction counts from CTP digits
  /// \param ctpdigits Vector of CTP digits to be processed
  /// \return Array of luminosity values for each BC
  // std::pair<std::array<double, o2::constants::lhc::LHCMaxBunches>, std::array<double, o2::constants::lhc::LHCMaxBunches>>
  void computeLumiPerBC(const o2::pmr::vector<CTPDigit>& ctpdigits, uint32_t firstOrbit, uint32_t orbitsPerTF);
  /// \brief Integrate luminosity per BC over multiple time frames
  /// \param perInterval Array of luminosity values for each BC for a given time interval
  void integrateLumi(const std::array<double, o2::constants::lhc::LHCMaxBunches>& tfCounts1, const std::array<double, o2::constants::lhc::LHCMaxBunches>& tfCounts2, int64_t unixTime, uint32_t nOrbitsThisTF);
  void writeMassiLinePerBC(int bc, int64_t unixTime, double lumi, double lumiErr, double correctedRate, double correctedLumi, double mu);
  void writeMassiLineLumi(int64_t unixTime, double lumi, double lumiErr);
  int64_t unixTimeForOrbitStart(uint32_t orbit) const;
  int yearFromUnixTime(int64_t unixTime) const;
  void fetchRunInfo(int runNumber);

 protected:
 private:
  // for digits
  bool mDoDigits = true;
  o2::pmr::vector<CTPDigit> mOutputDigits;
  int mMaxInputSize = 0;
  bool mMaxInputSizeFatal = 0;
  // for lumi
  bool mDoLumi = true;
  //
  LumiInfo mOutputLumiInfo;
  bool mVerbose = false;
  uint64_t mCountsT = 0;
  uint64_t mCountsV = 0;
  uint32_t mNTFToIntegrate = 1;
  uint32_t mNHBIntegrated = 0;
  uint32_t mNHBIntegratedT = 0;
  uint32_t mNHBIntegratedV = 0;
  uint32_t mNHBToIntegrate = 1;
  uint32_t mFirstOrbit = 0;
  uint32_t mOrbitsInCurrentWindow = 0;
  uint32_t mTFsInCurrentWindow = 0;
  double mWindowStartTime = 0.0;
  bool mDecodeinputs = 0;
  std::deque<size_t> mHistoryT;
  std::deque<size_t> mHistoryV;
  RawDataDecoder mDecoder;
  // Errors
  int mLostDueToShiftInps = 0;
  int mErrorIR = 0;
  int mErrorTCR = 0;
  int mIRRejected = 0;
  int mTCRRejected = 0;
  std::array<uint64_t, o2::ctp::CTP_NCLASSES> mClsEA{};
  std::array<uint64_t, o2::ctp::CTP_NCLASSES> mClsEB{}; // from inputs
  std::array<uint64_t, o2::ctp::CTP_NCLASSES> mClsA{};
  std::array<uint64_t, o2::ctp::CTP_NCLASSES> mClsB{}; // from inputs
  bool mCheckConsistency = false;
  std::array<double, o2::constants::lhc::LHCMaxBunches> mCountsPerBC1{};
  std::array<double, o2::constants::lhc::LHCMaxBunches> mCountsPerBC2{};
  double totalTime = 0.0;
  uint32_t mOrbitsPerTF = 0;
  const double tfTime = mOrbitsPerTF * o2::constants::lhc::LHCOrbitMUS * 1e-6; // total time in seconds for one timeframe
  std::bitset<3564> mLHCBCs;
  static constexpr double orbitTime = o2::constants::lhc::LHCOrbitMUS * 1e-6; // one HBF
  std::array<double, o2::constants::lhc::LHCMaxBunches> mTotalCountsPerBC1{};
  std::array<double, o2::constants::lhc::LHCMaxBunches> mTotalCountsPerBC2{};
  double mTotalElapsedTime = 0.0;
  // Massi file output
  std::string mFillNumber = "unknown";
  std::string mMassiOutDir;
  int mMassiYear = 0;
  double mOrbitResetTimeSec = 0.0;
  bool mStableBeams = false;
  std::map<int, std::ofstream> mMassiFiles; // one open file per RF bucket
  o2::parameters::AggregatedRunInfo mRunInfo;
  double mCrossSection = 1.0;
  double mTFsInMin = 0.0;
  uint32_t mPrevTFLastOrbit = 0;
  bool mHavePrevTF = false;
  int mRunStartTime = 0;
  int mRunEndTime = 0;
  struct PendingTF {
    std::array<double, o2::constants::lhc::LHCMaxBunches> countsPerBC1{};
    std::array<double, o2::constants::lhc::LHCMaxBunches> countsPerBC2{};
    int64_t unixTimeStart;
    uint32_t nOrbitsThisTF;
  };
  std::map<uint32_t, PendingTF> mPendingTFs;
  uint32_t mReorderDepth = 5;
  void flushReadyTFs();
  void flushAllPendingTFs();
  std::pair<double, double> pileupCorrection(double rate) const;
};

/// \brief Creating DataProcessorSpec for the CTP
///
o2::framework::DataProcessorSpec getRawDecoderSpec(bool askSTFDist, bool digits, bool lumi);

} // namespace reco_workflow

} // namespace ctp

} // namespace o2

#endif
