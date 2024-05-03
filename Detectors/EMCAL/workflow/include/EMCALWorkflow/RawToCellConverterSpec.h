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
#include "DataFormatsEMCAL/CompressedTriggerData.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "Headers/DataHeader.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALBase/TriggerMappingV2.h"
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
class Channel;
class MinorAltroDecodingError;
class RawDecodingError;
class FastORIndexException;
class TRUIndexException;

namespace reco_workflow
{

/// \class RawToCellConverterSpec
/// \brief Coverter task for Raw data to EMCAL cells and trigger objects
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALworkflow
/// \since December 10, 2019
////
/// General reconstruction task of EMCAL raw data of FEC and trigger sources, running during the synchronous
/// reconstruction. The task decodees pages from ALTRO, Fake ALTRO and STU sources, and mergees the data according
/// to events (triggers). The data is then further processed based on the origin of the data:
///
/// - in case of FEC data (cells or LEDMONS) a raw fit is performed, and data from high- and low-gain channels are
///   merged, always preferring the high-gain data if not saturated (better resolution)
/// - in case of Fake ALTRO data (TRU) the L0 timesums, trigger patches and TRU information are reconstructed. For
///   the trigger patches a small peak finder selects the time sample with the max. patch energy. For the L0 timesums
///   a fixed L0 time (8) is used for all event types
///
/// FEC and trigger information are ordered according to events and streamed to their output buffers, where corresponding
/// trigger records mark ranges in the buffer belonging to the same trigger. Tasks subscribing to outputs from this
/// task must always subscribe to teh trigger records in addition.
///
/// Several components of the task (raw parsing, ALTRO decoding, channel mapping and geometry, raw fit) can end in error
/// states, particularly due to unexpected data. Error handling is performed internally at different stages. Errors are
/// cathegoriezed as major or minor errors, where major errors prevent decoding the page while minor errors only lead
/// to loss of certain segments. For monitoring tasks must subscribe to EMC/DECODERERR.
///
/// In order to guarantee data consistency a link checker compares the links contributing to data from a certain trigger
/// to the links activated in the DCS. If not all links are present the timeframe is discarded. In order to switch off
/// that feature the option --no-checkactivelinks must be activated.
///
/// Inputs:
/// | Input spec           | Optional | CCDB | Purpose                                      |
/// |----------------------|----------|------|----------------------------------------------|
/// | EMC/RAWDATA          | no       | no   | EMCAL raw data                               |
/// | FLP/DISTSUBTIMEFRAME | yes      | no   | Message send when no data was received in TF |
/// | EMC/RECOPARAM        | no       | yes  | Reconstruction parameters                    |
/// | EMC/FEEDCS           | no       | yes  | FEE DCS information                          |
///
/// Outputs:
/// | Input spec           | Subspec (default) | Optional  | Purpose                                            |
/// |----------------------|-------------------|-----------|----------------------------------------------------|
/// | EMC/CELLS            | 1                 | no        | EMCAL cell (tower) data                            |
/// | EMC/CELLSTRGR        | 1                 | no        | Trigger records related to cell data               |
/// | EMC/DECODERERR       | 1                 | yes       | Decoder errors (for QC), if enabled                |
/// | EMC/TRUS             | 1                 | yes       | TRU information, if trigger reconstruction enabled |
/// | EMC/TRUSTRGR         | 1                 | yes       | Trigger reconrds related to TRU information        |
/// | EMC/PATCHES          | 1                 | yes       | Trigger patches, if trigger reconstruction enabled |
/// | EMC/PATCHESTRGR      | 1                 | yes       | Trigger reconrds related to trigger patches        |
/// | EMC/FASTORS          | 1                 | yes       | L0 timesums, if trigger reconstruction enabled     |
/// | EMC/FASTORSTRGR      | 1                 | yes       | Trigger reconrds related to L0 timesums            |
///
/// Workflow options (via --EMCALRawToCellConverterSpec ...):
/// | Option              | Default | Possible values | Purpose                                        |
/// |---------------------|---------|-----------------|------------------------------------------------|
/// | fitmethod           | gamma2  | gamma2,standard | Raw fit method                                 |
/// | maxmessage          | 100     | any int         | Max. amount of error messages on infoLogger    |
/// | printtrailer        | false   | set (bool)      | Print RCU trailer (for debugging)              |
/// | no-mergeHGLG        | false   | set (bool)      | Do not merge HG and LG channels for same tower |
/// | no-checkactivelinks | false   | set (bool)      | Do not check for active links per BC           |
/// | no-evalpedestal     | false   | set (bool)      | Disable pedestal evaluation                    |
///
/// Global switches of the EMCAL reco workflow related to the RawToCellConverter:
/// | Option                         | Default | Purpose                                       |
/// | -------------------------------|---------|-----------------------------------------------|
/// | disable-decoding-errors        | false   | Disable sending decoding errors               |
/// | disable-trigger-reconstruction | false   | Disable trigger reconstruction                |
/// | ignore-dist-stf                | false   | disable subscribing to FLP/DISTSUBTIMEFRAME/0 |
class RawToCellConverterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param subspecification Output subspecification for parallel running on multiple nodes
  /// \param hasDecodingErrors Option to swich on/off creating raw decoding error objects for later monitoring
  /// \param hasTriggerReconstruction Perform trigger reconstruction and add trigger-related outputs
  /// \param calibhandler Calibration object handler
  RawToCellConverterSpec(int subspecification, bool hasDecodingErrors, bool hasTriggerReconstruction, std::shared_ptr<CalibLoader> calibhandler) : framework::Task(), mSubspecification(subspecification), mCreateRawDataErrors(hasDecodingErrors), mDoTriggerReconstruction(hasTriggerReconstruction), mCalibHandler(calibhandler){};

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

  /// \brief Set noise threshold for gain type errors
  /// \param threshold Noise threshold
  void setNoiseThreshold(int threshold) { mNoiseThreshold = threshold; }

  /// \brief Get the noise threshold for gain type errors
  /// \return Noise threshold
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

  /// \struct CellTimeCorrection
  /// \brief  Correction for cell time
  struct CellTimeCorrection {
    double mTimeShift; ///< Constant time shift
    int mBcMod4;       ///< BC-dependent shift

    /// \brief Get the corrected cell time
    /// \param rawtime Raw time from fit
    /// \return Corrected time
    ///
    /// The time is corrected for an average shift and the BC phase
    double getCorrectedTime(double rawtime) const { return rawtime - mTimeShift - 25. * mBcMod4; }
  };

  /// \struct LocalPosition
  /// \brief Position in the supermodule coordinate system
  struct LocalPosition {
    uint16_t mSupermoduleID; ///< Supermodule ID
    uint16_t mFeeID;         ///< FEE ID
    uint8_t mColumn;         ///< Column in supermodule
    uint8_t mRow;            ///< Row in supermodule
  };

  using TRUContainer = std::vector<o2::emcal::CompressedTRU>;
  using PatchContainer = std::vector<o2::emcal::CompressedTriggerPatch>;

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
  /// \param ctx target processing context
  ///
  /// Send data to all output channels for the given subspecification. The subspecification
  /// is determined on the fly in the run method and therefore used as parameter. Consumers
  /// must use wildcard subspecification via ConcreteDataTypeMatcher.
  void sendData(framework::ProcessingContext& ctx) const;

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

  /// \brief Add FEE channel to the current evnet
  /// \param currentEvent Event to add the channel to
  /// \param currentchannel Current FEE channel
  /// \param timeCorrector Handler for correction of the time
  /// \param position Channel coordinates
  /// \param chantype Channel type (High Gain, Low Gain, LEDMON)
  ///
  /// Performing a raw fit of the bunches in the channel to extract energy and time, and
  /// adding them to the container for FEE data of the given event.
  void addFEEChannelToEvent(o2::emcal::EventContainer& currentEvent, const o2::emcal::Channel& currentchannel, const CellTimeCorrection& timeCorrector, const LocalPosition& position, ChannelType_t chantype);

  /// \brief Add TRU channel to the event
  /// \param currentEvent Event to add the channel to
  /// \param currentchannel Current TRU channel
  /// \param position Channel coordinates
  ///
  /// TRU channels are encoded in colums:
  /// - 0-95: FastOR timeseries (time-reversed)
  /// - 96-105: bitmap with fired patches and TRU header
  /// The TRU index is taken from the hardware address, while the FastOR index is taken from the
  /// column number. The TRU and patch times are taken from the time sample in which the corresponding
  /// bit is found.
  void addTRUChannelToEvent(o2::emcal::EventContainer& currentEvent, const o2::emcal::Channel& currentchannel, const LocalPosition& position);

  /// @brief Build L0 patches from FastOR time series and TRU data of the current event
  /// @param currentevent Current event to process
  /// @return Compressed patches
  ///
  /// Only reconstruct patches which were decoded as fired from the raw data. The patch energy and time
  /// are calculated from the corresponding FastOR time series as the energy and time with the largest
  /// patch energy extracted from all possible time integrals (see reconstructTriggerPatch)
  std::tuple<TRUContainer, PatchContainer> buildL0Patches(const EventContainer& currentevent) const;

  /// @brief Build L0 timesums with respect to a given L0 time
  /// @param currentevent Current event with FastOR time series
  /// @param l0time L0 time of the event
  /// @return Container with time series
  std::vector<o2::emcal::CompressedL0TimeSum> buildL0Timesums(const o2::emcal::EventContainer& currentevent, uint8_t l0time) const;

  /// \brief Reconstruct trigger patch energy and time from its FastOR time series
  /// \param fastors FastORs contributing to the patch (only present)
  /// \return Tuple with 0 - patch ADC, 1 - patch time
  ///
  /// For all possible combinations reconstruct the 4-time integral of the patches from
  /// its contributing FastORs. The patch is reconstructed when the ADC reached its
  /// maximum. The patch time is the start time of the 4-integral
  std::tuple<uint16_t, uint8_t> reconstructTriggerPatch(const gsl::span<const FastORTimeSeries*> fastors) const;

  /// \brief Handling of mapper hardware address errors
  /// \param error Exception raised by the mapper
  /// \param ddlID DDL ID of the segment raising the exception
  /// \param hwaddress Hardware address raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleAddressError(const Mapper::AddressNotFoundException& error, int ddlID, int hwaddress);

  /// \brief Handler function for major ALTRO decoder errors
  /// \param altroerror Exception raised by the ALTRO decoder
  /// \param ddlID DDL ID of the segment raising the exception
  /// \param hwaddress Hardware address raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleAltroError(const o2::emcal::AltroDecoderError& altroerror, int ddlID);

  /// \brief Handler function for minor ALTRO errors
  /// \param altroerror Minor errors created by the ALTRO decoder
  /// \param ddlID DDL ID of the segment raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleMinorAltroError(const o2::emcal::MinorAltroDecodingError& altroerror, int ddlID);

  /// \brief Handler function of mapper errors related to invalid DDL
  /// \param error Exception raised by the mapper
  /// \param feeID FEE ID (DDL ID) of the segment raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleDDLError(const MappingHandler::DDLInvalid& error, int feeID);

  /// \brief Handler function of errors related to geometry (invalid supermodule / module/ tower ...)
  /// \param error Geometry exception
  /// \param supermoduleID Supermodule ID of the exception
  /// \param cellID Cell (Tower) ID of the exception
  /// \param hwaddress Hardware address raising the exception
  /// \param chantype Channel type of the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleGeometryError(const ModuleIndexException& error, int supermoduleID, int cellID, int hwaddress, ChannelType_t chantype);

  /// \brief Handler function of mapper errors related to invalid DDL
  /// \param error Exception raised by the mapper
  /// \param feeID FEE ID (DDL ID) of the segment raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleFitError(const o2::emcal::CaloRawFitter::RawFitterError_t& fiterror, int ddlID, int cellID, int hwaddress);

  /// \brief Handler function for gain type errors
  /// \param errortype Gain error type
  /// \param ddlID DDL ID of the segment raising the exception
  /// \param hwaddress Hardware address raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleGainError(const o2::emcal::reconstructionerrors::GainError_t& errortype, int ddlID, int hwaddress);

  /// \brief Handler function for raw page decoding errors (i.e. header/trailer corruptions)
  /// \param error Raw page error
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handlePageError(const RawDecodingError& error);

  /// \brief Handler function for minor raw page decoding errors (i.e. header/trailer corruptions)
  /// \param error Raw page error
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleMinorPageError(const RawReaderMemory::MinorError& error);

  /// \brief Handler function for FastOR indexing errors
  /// \param error FastOR index error
  /// \param linkID DDL raising the exception
  /// \param indexTRU TRU raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleFastORErrors(const FastORIndexException& error, unsigned int linkID, unsigned int indexTRU);

  /// \brief Handler function patch index exception
  /// \param error Patch index error
  /// \param linkID DDL raising the exception
  /// \param indexTRU TRU raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handlePatchError(const TRUDataHandler::PatchIndexException& error, unsigned int linkID, unsigned int indexTRU);

  /// \brief Handler function for TRU index exception
  /// \param error TRU index error
  /// \param linkID DDL raising the exception
  /// \param hwaddress Hardware address of the channel raising the exception
  ///
  /// Errors are printed to the infoLogger until a user-defiened
  /// threshold is reached. In case the export of decoder errors
  /// is activated an error object with additional information is
  /// produced.
  void handleTRUIndexError(const TRUIndexException& error, unsigned int linkID, unsigned int hwaddress);

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
  bool mDoTriggerReconstruction = false;                             ///< Do trigger reconstruction
  std::chrono::time_point<std::chrono::system_clock> mReferenceTime; ///< Reference time for muting messages
  Geometry* mGeometry = nullptr;                                     ///!<! Geometry pointer
  RecoContainer mCellHandler;                                        ///< Manager for reconstructed cells
  std::shared_ptr<CalibLoader> mCalibHandler;                        ///< Handler for calibration objects
  std::unique_ptr<MappingHandler> mMapper = nullptr;                 ///!<! Mapper
  std::unique_ptr<TriggerMappingV2> mTriggerMapping;                 ///!<! Trigger mapping
  std::unique_ptr<CaloRawFitter> mRawFitter;                         ///!<! Raw fitter
  std::vector<Cell> mOutputCells;                                    ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;                  ///< Container with output trigger records for cells
  std::vector<ErrorTypeFEE> mOutputDecoderErrors;                    ///< Container with decoder errors
  std::vector<CompressedTRU> mOutputTRUs;                            ///< Compressed output TRU information
  std::vector<TriggerRecord> mOutputTRUTriggerRecords;               ///< Container with trigger records for TRU data
  std::vector<CompressedTriggerPatch> mOutputPatches;                ///< Compressed trigger patch information
  std::vector<TriggerRecord> mOutputPatchTriggerRecords;             ///< Container with trigger records for Patch data
  std::vector<CompressedL0TimeSum> mOutputTimesums;                  ///< Compressed L0 timesum information
  std::vector<TriggerRecord> mOutputTimesumTriggerRecords;           ///< Trigger records for L0 timesum
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
/// \param askDISTSTF Include input spec FLP/DISTSUBTIMEFRAME
/// \param disableDecodingErrors Obtain reco params from the CCDB
/// \param disableTriggerReconstruction Do not run trigger reconstruction
/// \param subspecification Subspecification used in the output spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec(bool askDISTSTF, bool disableDecodingError, bool disableTriggerReconstruction, int subspecification);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2

#endif // O2_EMCAL_RAWTOCELLCONVERTER_SPEC
