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

#include <climits>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <boost/container_hash/hash.hpp>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/ErrorTypeFEE.h"
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
  /// \struct MessageChannel
  /// \brief Handling of messages per HW channel
  struct MessageChannel {
    int mDDL;             ///< DDK ID
    int mFEC;             ///< FEC in DDL
    int mHWAddress;       ///< Full hardware address
    std::string mMessage; ///< Error message

    /// \brief Check whehter message is equal to other message
    /// \param other Message to compare to
    /// \return True if the messages (including channel information) are the same, false otherwise
    bool operator==(const MessageChannel& other) const
    {
      return mDDL == other.mDDL && mFEC == other.mFEC && mHWAddress == other.mHWAddress && mMessage == other.mMessage;
    }

    /// \brief Create combined error message with channel information
    /// \return Error message to be logged
    std::string print();
  };

  /// \struct MessageChannelHasher
  /// \brief Hash functor for channel-based error messages
  struct MessageChannelHasher {

    /// \brief Hash functor
    /// \param s MessageChannel object to be hashed
    /// \return Hash value
    size_t operator()(const MessageChannel& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mDDL);
      boost::hash_combine(seed, s.mFEC);
      boost::hash_combine(seed, s.mHWAddress);
      boost::hash_combine(seed, std::hash<std::string>{}(s.mMessage));
      return seed;
    }
  };

  /// \struct MessageDDL
  /// \brief Handling messages per DDL
  struct MessageDDL {
    int mDDL;             ///< DDL ID
    std::string mMessage; ///< Error message

    /// \brief Check whehter message is equal to other message
    /// \param other Message to compare to
    /// \return True if the messages (including DDL information) are the same, false otherwise
    bool operator==(const MessageDDL& other) const
    {
      return mDDL == other.mDDL && mMessage == other.mMessage;
    }

    /// \brief Create combined error message with DDL information
    /// \return Error message to be logged
    std::string print();
  };

  /// \struct MessageDDLHasher
  /// \brief Hash functor for DDL-based error messages
  struct MessageDDLHasher {

    /// \brief Hash functor
    /// \param s MessageDDL object to be hashed
    /// \return Hash value
    size_t operator()(const MessageDDL& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mDDL);
      boost::hash_combine(seed, std::hash<std::string>{}(s.mMessage));
      return seed;
    }
  };

  /// \struct MessageCell
  /// \brief Handling messages per Cell
  struct MessageCell {
    int mSMID;            ///< Supermodule ID
    int mCellID;          ///< Cell ID
    std::string mMessage; ///< Error message

    /// \brief Check whehter message is equal to other message
    /// \param other Message to compare to
    /// \return True if the messages (including cell information) are the same, false otherwise
    bool operator==(const MessageCell& other) const
    {
      return mSMID == other.mSMID && mCellID == other.mCellID && mMessage == other.mMessage;
    }

    /// \brief Create combined error message with cell information
    /// \return Error message to be logged
    std::string print();
  };

  /// \struct MessageCellHasher
  /// \brief Hash functor for Cell-based error messages
  struct MessageCellHasher {

    /// \brief Hash functor
    /// \param s MessageDDL object to be hashed
    /// \return Hash value
    size_t operator()(const MessageCell& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mCellID);
      boost::hash_combine(seed, s.mSMID);
      boost::hash_combine(seed, std::hash<std::string>{}(s.mMessage));
      return seed;
    }
  };

  /// \class MessageHandler
  /// \brief Logging of error message suppressing multiple occurrences
  ///
  /// In order to prevent multiple occurrences of the same error message
  /// which can end up in a message flood on the infoBrowser an internal
  /// map keeps track of messages already printed before. In case the
  /// message is fouund in the map it is suppressed until the end of the run.
  template <typename MessageType, typename Hasher>
  class MessageHandler
  {
   public:
    /// \enum Severity
    /// \brief Logging severity
    enum class Severity {
      WARNING, ///< Waring level
      ERROR,   ///< Error level
      FATAL    ///< Fatal level
    };

    /// \brief Constructor
    MessageHandler() = default;

    /// \brief Destructor
    ~MessageHandler() = default;

    /// \brief Logging error message
    /// \brief msg Message object (channel and message)
    /// \brief level Log severity
    ///
    /// Printing message in case the message has not been printed
    /// before for the same channel
    void log(MessageType msg, Severity level);

   private:
    std::unordered_map<MessageType, unsigned long, Hasher> mErrorsPerEntry; ///< Counter of the number of suppressed error messages
  };

  using ChannelMessageHandler = MessageHandler<MessageChannel, MessageChannelHasher>;
  using CellMessageHandler = MessageHandler<MessageCell, MessageCellHasher>;
  using DDLMessageHandler = MessageHandler<MessageDDL, MessageDDLHasher>;

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

  /// \brief Check if the timeframe is an empty timeframe (contains DEADBEEF message)
  /// \return True if the timeframe is a lost timeframe, false if it is OK
  bool isLostTimeframe(framework::ProcessingContext& ctx) const;

  /// \brief Send data to output channels
  /// \param cells Container with cells from all events in the timeframe
  /// \param triggers Trigger records with ranges in cell container
  /// \param decodingErrors Container with raw decoding errors (not separated per event)
  void sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors) const;

  header::DataHeader::SubSpecificationType mSubspecification = 0; ///< Subspecification for output channels
  int mNoiseThreshold = 0;                                        ///< Noise threshold in raw fit
  int mNumErrorMessages = 0;                                      ///< Current number of error messages
  int mErrorMessagesSuppressed = 0;                               ///< Counter of suppressed error messages
  int mMaxErrorMessages = 100;                                    ///< Max. number of error messages
  bool mMergeLGHG = true;                                         ///< Merge low and high gain cells
  bool mPrintTrailer = false;                                     ///< Print RCU trailer
  Geometry* mGeometry = nullptr;                                  ///!<! Geometry pointer
  std::unique_ptr<MappingHandler> mMapper = nullptr;              ///!<! Mapper
  std::unique_ptr<CaloRawFitter> mRawFitter;                      ///!<! Raw fitter
  std::vector<Cell> mOutputCells;                                 ///< Container with output cells
  std::vector<TriggerRecord> mOutputTriggerRecords;               ///< Container with output cells
  std::vector<ErrorTypeFEE> mOutputDecoderErrors;                 ///< Container with decoder errors
  ChannelMessageHandler mChannelMessages;                         ///< Handling of error message per Channel
  CellMessageHandler mCellMessages;                               ///< Handling of error message per Cell
  DDLMessageHandler mDDLMessages;                                 ///< Handling of error messages per DDL
};

/// \brief Creating DataProcessorSpec for the EMCAL Cell Converter Spec
///
/// Refer to RawToCellConverterSpec::run for input and output specs
framework::DataProcessorSpec getRawToCellConverterSpec(bool askDISTSTF, int subspecification);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
