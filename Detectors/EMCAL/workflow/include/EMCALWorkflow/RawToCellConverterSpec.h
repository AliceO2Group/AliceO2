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

#ifndef O2_EMCAL_RAWTOCELLCONVERTER_SPEC
#define O2_EMCAL_RAWTOCELLCONVERTER_SPEC

#include <chrono>
#include <exception>
#include <vector>

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "Headers/DataHeader.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALReconstruction/CaloRawFitter.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "EMCALReconstruction/RecoContainer.h"
#include "EMCALReconstruction/ReconstructionErrors.h"
#include "EMCALWorkflow/CalibLoader.h"

namespace o2
{

namespace emcal
{

class AltroDecoderError;
class MinorAltroDecodingError;
class RawDecodingError;

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
  /// \param hasDecodingErrors Option to swich on/off creating raw decoding error objects for later monitoring
  /// \param loadRecoParamsFromCCDB Option to load the RecoParams from the CCDB
  /// \param calibhandler Calibration object handler
  RawToCellConverterSpec(int subspecification, bool hasDecodingErrors, std::shared_ptr<CalibLoader> calibhandler) : framework::Task(), mSubspecification(subspecification), mCreateRawDataErrors(hasDecodingErrors), mCalibHandler(calibhandler){};

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

  /// \brief Handle objects obtained from the CCDB
  /// \param matcher Matcher providing the CCDB path of the object
  /// \param obj CCDB object loaded by the CCDB interface
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

  /// \brief Set max number of error messages printed
  /// \param maxMessages Max. amount of messages printed
  ///
  /// Error messages will be suppressed once the maximum is reached
  void setMaxErrorMessages(int maxMessages)
  {
    mMaxErrorMessages = maxMessages;
  }

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
  /// \class ModuleIndexException
  /// \brief Exception handling errors in calculation of the absolute module ID
  class ModuleIndexException : public std::exception
  {
   public:
    /// \enum ModuleType_t
    /// \brief Type of module raising the exception
    enum class ModuleType_t {
      CELL_MODULE,  ///< Cell module type
      LEDMON_MODULE ///< LEDMON module type
    };

    /// \brief Constructor for cell indices
    /// \param moduleIndex Index of the module raising the exception
    /// \param column Column of the cell
    /// \param row Row of the cell
    /// \param columnshifted Shifted column index
    /// \param rowshifted Shifted row index
    ModuleIndexException(int moduleIndex, int column, int row, int columnshifted, int rowshifted);

    /// \brief Constructor for LEDMON indices
    /// \param moduleIndex Index of the module raising the exception
    ModuleIndexException(int moduleIndex);

    /// \brief Destructor
    ~ModuleIndexException() noexcept final = default;

    /// \brief Access to error message
    /// \return Error message
    const char* what() const noexcept final { return "Invalid cell / LEDMON index"; }

    /// \brief Get type of module raising the exception
    /// \return Module type
    ModuleType_t getModuleType() const { return mModuleType; }

    /// \brief Get index of the module raising the exception
    /// \return Index of the module
    int getIndex() const { return mIndex; }

    /// \brief Get column raising the exception (cell-case)
    /// \return Column
    int getColumn() const { return mColumn; }

    /// \brief Get row raising the exception (cell-case)
    /// \return Row
    int getRow() const { return mRow; }

    /// \brief Get shifted column raising the exception (cell-case)
    /// \return Shifted column
    int getColumnShifted() const { return mColumnShifted; }

    /// \brief Get shifted row raising the exception (cell-case)
    /// \return Shifted row
    int getRowShifted() const { return mRowShifted; }

   private:
    ModuleType_t mModuleType; ///< Type of the module raising the exception
    int mIndex = -1;          ///< Index raising the exception
    int mColumn = -1;         ///< Column of the module (cell-case)
    int mRow = -1;            ///< Row of the module (cell-case)
    int mColumnShifted = -1;  ///< shifted column of the module (cell-case)
    int mRowShifted = -1;     /// << shifted row of the module (cell-case)
  };

  /// \brief Check if the timeframe is empty
  /// \param ctx Processing context of timeframe
  /// \return True if the timeframe is empty, false otherwise
  ///
  /// Emtpy timeframes do not have RAWDATA from any physical link in the
  /// processing context, instead they contain RAWDATA from link 0xDEADBEEF
  /// and a message in FLP/DISTSUBTIMEFRAME
  bool isLostTimeframe(framework::ProcessingContext& ctx) const;

  /// \brief Update calibration objects
  void updateCalibrationObjects();

  /// \brief Select cells and put them on the output container
  /// \param cells Cells to select
  /// \param isLEDMON Distinction whether input is cell or LEDMON
  /// \return Number of accepted cells
  ///
  /// Cells or LEDMONs are rejected if
  /// - They have a low gain but no high gain channel
  /// - They only have a high gain channel which is out of range
  int bookEventCells(const gsl::span<const o2::emcal::RecCellInfo>& cells, bool isLELDMON);

  /// \brief Send data to output channels
  /// \param cells Container with output cells for timeframe
  /// \param triggers Container with trigger records for timeframe
  /// \param decodingErrors Container with decoding errors for timeframe
  ///
  /// Send data to all output channels for the given subspecification. The subspecification
  /// is determined on the fly in the run method and therefore used as parameter. Consumers
  /// must use wildcard subspecification via ConcreteDataTypeMatcher.
  void sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors) const;

  /// \brief Get absolute Cell ID from column/row in supermodule
  /// \param supermoduleID Index of the supermodule
  /// \param column Column of the tower within the supermodule
  /// \param row Row of the tower within the supermodule
  /// \return Cell absolute ID
  /// \throw ModuleIndexException in case of invalid module indices
  int getCellAbsID(int supermoduleID, int column, int row);

  /// \brief Get the absoulte ID of LEDMON from the module ID in supermodule
  /// \param supermoduleID Index of the supermodule
  /// \param module Index of the module within the supermodule
  /// \return LEDMON absolute ID
  /// \throw ModuleIndexException in case of invalid module indices
  int geLEDMONAbsID(int supermoduleID, int module);

  void handleAddressError(const Mapper::AddressNotFoundException& error, int ddlID, int hwaddress);

  void handleAltroError(const o2::emcal::AltroDecoderError& altroerror, int ddlID);

  void handleMinorAltroError(const o2::emcal::MinorAltroDecodingError& altroerror, int ddlID);

  void handleDDLError(const MappingHandler::DDLInvalid& error, int feeID);

  void handleGeometryError(const ModuleIndexException& e, int supermoduleID, int cellID, int hwaddress, ChannelType_t chantype);

  void handleFitError(const o2::emcal::CaloRawFitter::RawFitterError_t& fiterror, int ddlID, int cellID, int hwaddress);

  /// \brief handler function for gain type errors
  /// \param errortype Gain error type
  /// \param ddlID ID of the DDL
  /// \param hwaddress Hardware address
  void handleGainError(const o2::emcal::reconstructionerrors::GainError_t& errortype, int ddlID, int hwaddress);

  void handlePageError(const RawDecodingError& e);

  void handleMinorPageError(const RawReaderMemory::MinorError& e);

  header::DataHeader::SubSpecificationType mSubspecification = 0;    ///< Subspecification for output channels
  int mNoiseThreshold = 0;                                           ///< Noise threshold in raw fit
  int mNumErrorMessages = 0;                                         ///< Current number of error messages
  int mErrorMessagesSuppressed = 0;                                  ///< Counter of suppressed error messages
  int mMaxErrorMessages = 100;                                       ///< Max. number of error messages
  bool mMergeLGHG = true;                                            ///< Merge low and high gain cells
  bool mActiveLinkCheck = true;                                      ///< Run check for active links
  bool mPrintTrailer = false;                                        ///< Print RCU trailer
  bool mDisablePedestalEvaluation = false;                           ///< Disable pedestal evaluation independent of settings in the RCU trailer
  bool mCreateRawDataErrors = false;                                 ///< Create raw data error objects for monitoring
  std::chrono::time_point<std::chrono::system_clock> mReferenceTime; ///< Reference time for muting messages
  Geometry* mGeometry = nullptr;                                     ///!<! Geometry pointer
  RecoContainer mCellHandler;                                        ///< Manager for reconstructed cells
  std::shared_ptr<CalibLoader> mCalibHandler;                        ///< Handler for calibration objects
  std::unique_ptr<MappingHandler> mMapper = nullptr;                 ///!<! Mapper
  std::unique_ptr<CaloRawFitter> mRawFitter;                         ///!<! Raw fitter
  std::vector<Cell> mOutputCells;                                    ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;                  ///< Container with output cells
  std::vector<ErrorTypeFEE> mOutputDecoderErrors;                    ///< Container with decoder errors
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
/// \param askDISTSTF Include input spec FLP/DISTSUBTIMEFRAME
/// \param loadRecoParamsFromCCDB Obtain reco params from the CCDB
/// \param subspecification Subspecification used in the output spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec(bool askDISTSTF, bool disableDecodingError, int subspecification);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2

#endif // O2_EMCAL_RAWTOCELLCONVERTER_SPEC
