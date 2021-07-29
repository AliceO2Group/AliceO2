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

#include <chrono>
#include <vector>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "Headers/DataHeader.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALReconstruction/CaloRawFitter.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class RawToCellConverterSpec
/// \brief Coverter task for Raw data to EMCAL cells
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since December 10, 2019
///
class RawToCellConverterSpec : public framework::Task
{
 public:
  /// \brief Default constructor
  RawToCellConverterSpec() = default;

  /// \brief Destructor
  ~RawToCellConverterSpec() override;

  /// \brief Initializing the RawToCellConverterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of raw data to cells
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input RawData: {"ROUT", "RAWDATA", 0, Lifetime::Timeframe}
  /// Output cells: {"EMC", "CELLS", 0, Lifetime::Timeframe}
  /// Output cells trigger record: {"EMC", "CELLSTR", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

  /// \brief Set max number of error messages printed
  /// \param maxMessages Max. amount of messages printed
  ///
  /// Error messages will be suppressed once the maximum is reached
  void setMaxErrorMessages(int maxMessages) { mMaxErrorMessages = maxMessages; }

  /// \brief Set the noise threshold
  /// \param threshold Noise threshold
  ///
  /// ADC values below the noise threshold are suppressed in the raw fit
  void setNoiseThreshold(int threshold) { mNoiseThreshold = threshold; }

  /// \brief Get the noise threshold
  /// \return Noise threshold
  ///
  /// ADC values below the noise threshold are suppressed in the raw fit
  int getNoiseThreshold() const { return mNoiseThreshold; }

  /// \brief Set ID of the subspecification
  /// \param subspecification
  ///
  /// Can be used to define differenciate between output in case
  /// different processors run in parallel (i.e. on different FLPs)
  void setSubspecification(header::DataHeader::SubSpecificationType subspecification) { mSubspecification = subspecification; }

  /// \brief Get ID of the subspecification
  /// \return subspecification
  ///
  /// Can be used to define differenciate between output in case
  /// different processors run in parallel (i.e. on different FLPs)
  header::DataHeader::SubSpecificationType getSubspecification() const { return mSubspecification; }

 private:
  /// \struct RecCellInfo
  /// \brief Internal bookkeeping for cell double counting
  ///
  /// In case the energy is in the overlap region between the
  /// two digitizers 2 channels exist for the same cell. In this
  /// case the low gain cells are used above a certain threshold.
  /// In certain error cases the information from the other digitizer
  /// might be missing. Such cases must be fitered out, however this
  /// can be done only after all cells are processed. The overlap information
  /// needs to be propagated for the filtering but is not part of the
  /// final cell object
  struct RecCellInfo {
    o2::emcal::Cell mCellData; ///< Cell information
    bool mIsLGnoHG;            ///< Cell has only LG digits
    bool mHGOutOfRange;        ///< Cell has only HG digits which are out of range
    int mFecID;                ///< FEC ID of the channel (for monitoring)
    int mDDLID;                ///< DDL of the channel (for monitoring)
    int mHWAddressLG;          ///< HW address of LG (for monitoring)
    int mHWAddressHG;          ///< HW address of HG (for monitoring)
  };

  /// \brief Determine whether the timeframe is empty
  /// \param ctx Processing context with all inputs
  ///
  /// Empty timeframes are detected via the deadbeaf
  /// subspecification of the input channel.
  bool isLostTimeframe(framework::ProcessingContext& ctx) const;

  /// \brief Send data to output channels
  /// \param cells Container with output cells for timeframe
  /// \param triggers Container with trigger records for timeframe
  /// \param decodingErrors Container with decoding errors for timeframe
  /// \param subspecification Output subspecification
  ///
  /// Send data to all output channels for the given subspecification. The subspecification
  /// is determined on the fly in the run method and therefore used as parameter. Consumers
  /// must use wildcard subspecification via ConcreteDataTypeMatcher.
  void sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors, header::DataHeader::SubSpecificationType subspecification) const;

  header::DataHeader::SubSpecificationType mSubspecification = UINT32_MAX; ///< Subspecification for output channels
  int mNoiseThreshold = 0;                                                 ///< Noise threshold in raw fit
  int mNumErrorMessages = 0;                                               ///< Current number of error messages
  int mErrorMessagesSuppressed = 0;                                        ///< Counter of suppressed error messages
  int mMaxErrorMessages = 100;                                             ///< Max. number of error messages
  bool mAutoDefineSubspec = false;                                         ///< Automatically determine the subspecification based on the FEE ID of the first RDH
  bool mMergeLGHG = true;                                                  ///< Merge low and high gain cells
  bool mPrintTrailer = false;                                              ///< Print RCU trailer
  bool mDisablePedestalEvaluation = false;                                 ///< Disable pedestal evaluation independent of settings in the RCU trailer
  bool mPrintSubspecification = true;                                      ///< Printout output subspecification
  std::chrono::time_point<std::chrono::system_clock> mReferenceTime;       ///< Reference time for muting messages
  Geometry* mGeometry = nullptr;                                           ///!<! Geometry pointer
  std::unique_ptr<MappingHandler> mMapper = nullptr;                       ///!<! Mapper
  std::unique_ptr<CaloRawFitter> mRawFitter;                               ///!<! Raw fitter
  std::vector<Cell> mOutputCells;                                          ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;                        ///< Container with output cells
  std::vector<ErrorTypeFEE> mOutputDecoderErrors;                          ///< Container with decoder errors
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
/// \param askDISTSTF if true the task subscribes also to FLP/DISTSUBTIMEFRAME
///
/// Refer to RawToCellConverterSpec::run for input and output specs.
framework::DataProcessorSpec getRawToCellConverterSpec(bool askDISTSTF);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
