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
  /// \brief Constructor
  /// \param subspecification Output subspecification for parallel running on multiple nodes
  RawToCellConverterSpec(int subspecification) : framework::Task(), mSubspecification(subspecification){};

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

  void setNoiseThreshold(int thresold) { mNoiseThreshold = thresold; }
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
  bool isLostTimeframe(framework::ProcessingContext& ctx) const;
  void sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors) const;

  header::DataHeader::SubSpecificationType mSubspecification = 0; ///< Subspecification for output channels
  int mNoiseThreshold = 0;                                        ///< Noise threshold in raw fit
  int mNumErrorMessages = 0;                                      ///< Current number of error messages
  int mErrorMessagesSuppressed = 0;                               ///< Counter of suppressed error messages
  int mMaxErrorMessages = 100;                                    ///< Max. number of error messages
  bool mPrintTrailer = false;
  Geometry* mGeometry = nullptr;                                  ///!<! Geometry pointer
  std::unique_ptr<MappingHandler> mMapper = nullptr;              ///!<! Mapper
  std::unique_ptr<CaloRawFitter> mRawFitter;                      ///!<! Raw fitter
  std::vector<Cell> mOutputCells;                                 ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;               ///< Container with output cells
  std::vector<ErrorTypeFEE> mOutputDecoderErrors;                 ///< Container with decoder errors
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec(bool askDISTSTF, int subspecification);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
