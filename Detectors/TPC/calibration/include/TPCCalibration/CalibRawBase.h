// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALIBRAWBASE_H_
#define ALICEO2_TPC_CALIBRAWBASE_H_

/// \file   CalibRawBase.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <limits>
#include <vector>
#include <memory>
#include <algorithm>

#include "TString.h"
#include "Rtypes.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Mapper.h"

#include "TPCReconstruction/GBTFrameContainer.h"
#include "TPCReconstruction/RawReader.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCReconstruction/RawReaderEventSync.h"

namespace o2
{
namespace tpc
{

/// \brief Base class for raw data calibrations
///
/// This class is the base class for raw data calibrations
/// It implements base raw reader functionality and calls
/// an 'update' function for each digit
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class CalibRawBase
{
 public:
  enum class ProcessStatus : char {
    Ok,             ///< Processing ok
    Truncated,      ///< Read fewer time bins than mTimeBinsPerCall
    NoMoreData,     ///< No data read
    LastEvent,      ///< Last event
    NoReaders,      ///< No raw reader configures
    IncompleteEvent ///< Read data is incomplete
  };

  CalibRawBase(PadSubset padSubset = PadSubset::ROC) : mMapper(Mapper::instance()), mDebugLevel(0), mNevents(0), mTimeBinsPerCall(500), mProcessedTimeBins(0), mPresentEventNumber(std::numeric_limits<size_t>::max()), mPadSubset(padSubset), mGBTFrameContainers(), mRawReaders() { ; }

  virtual ~CalibRawBase() = default;

  /// update function called once per digit
  ///
  /// \param roc readout chamber
  /// \param row row in roc
  /// \param pad pad in row
  /// \param timeBin time bin
  /// \param signal ADC signal
  virtual Int_t updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                          const Int_t timeBin, const Float_t signal) = 0;

  /// update function called once per digit
  ///
  /// \param cru CRU
  /// \param row row in CRU
  /// \param pad pad in row
  /// \param timeBin time bin
  /// \param signal ADC signal
  virtual Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                          const Int_t timeBin, const Float_t signal) = 0;

  /// add GBT frame container to process
  void addGBTFrameContainer(GBTFrameContainer* cont) { mGBTFrameContainers.push_back(std::unique_ptr<GBTFrameContainer>(cont)); }

  /// add RawReader
  void addRawReader(RawReader* reader) { mRawReaders.push_back(std::unique_ptr<RawReader>(reader)); }

  /// set number of time bins to process in one call to processEvent
  void setTimeBinsPerCall(Int_t nTimeBins) { mTimeBinsPerCall = nTimeBins; }

  /// return the number of time bins processed in one call to processEvent
  Int_t getTimeBinsPerCall() const { return mTimeBinsPerCall; }

  /// return pad subset type used
  PadSubset getPadSubset() const { return mPadSubset; }

  /// Process one event
  /// \param eventNumber: Either number >=0 or -1 (next event) or -2 (previous event)
  ProcessStatus processEvent(int eventNumber = -1);

  /// set up the raw readers
  /// \param fileInfo file description information
  /// \param verbosity verbosity level of the raw readers
  /// \param debugLevel debug level of the raw readers
  void setupContainers(TString fileInfo, uint32_t verbosity = 0, uint32_t debugLevel = 0);

  /// Set the debug level
  /// \param debugLevel debug level
  void setDebugLevel(int debugLevel = 1) { mDebugLevel = debugLevel; }

  /// Rewind the events
  void rewindEvents();

  /// Dump the relevant data to file
  virtual void dumpToFile(const std::string filename) {}

  /// number of processed events
  size_t getNumberOfProcessedEvents() const { return mNevents; }

  /// get present event number
  size_t getPresentEventNumber() const { return mPresentEventNumber; }

  /// check if present event is complete
  bool isPresentEventComplete() const { return mRawReaderCRUManager.isEventComplete(mPresentEventNumber); }

  /// number of processed time bins in last event
  size_t getNumberOfProcessedTimeBins() const { return mProcessedTimeBins; }

  /// Debug level
  int getDebugLevel() const { return mDebugLevel; }

 protected:
  const Mapper& mMapper; //!< TPC mapper
  int mDebugLevel;       //!< debug level

 private:
  size_t mNevents;            //!< number of processed events
  int mTimeBinsPerCall;       //!< number of time bins to process in processEvent
  size_t mProcessedTimeBins;  //!< number of processed time bins in last event
  size_t mPresentEventNumber; //!< present event number

  PadSubset mPadSubset;                                                //!< pad subset type used
  std::vector<std::unique_ptr<GBTFrameContainer>> mGBTFrameContainers; //!< raw reader pointer
  std::vector<std::unique_ptr<RawReader>> mRawReaders;                 //!< raw reader pointer
  RawReaderCRUManager mRawReaderCRUManager{};                          //!< cru type raw readers

  virtual void resetEvent() = 0;
  virtual void endEvent() = 0;
  virtual void endReader(){};

  /// Process one event with mTimeBinsPerCall length using GBTFrameContainers
  ProcessStatus processEventGBT();

  /// Process one event using RawReader
  ProcessStatus processEventRawReader(int eventNumber = -1);

  /// Process one event using RawReaderCRU
  ProcessStatus processEventRawReaderCRU(int eventNumber = -1);
};

//----------------------------------------------------------------
// Inline Functions
//----------------------------------------------------------------
inline CalibRawBase::ProcessStatus CalibRawBase::processEvent(int eventNumber)
{
  if (mGBTFrameContainers.size()) {
    return processEventGBT();
  } else if (mRawReaders.size()) {
    return processEventRawReader(eventNumber);
  } else if (mRawReaderCRUManager.getNumberOfReaders()) {
    return processEventRawReaderCRU(eventNumber);
  } else {
    return ProcessStatus::NoReaders;
  }
}

//______________________________________________________________________________
inline CalibRawBase::ProcessStatus CalibRawBase::processEventGBT()
{
  if (!mGBTFrameContainers.size())
    return ProcessStatus::NoReaders;
  resetEvent();

  // loop over raw readers, fill digits for 500 time bins and process
  // digits

  const int nRowIROC = mMapper.getNumberOfRowsROC(0);

  ProcessStatus status = ProcessStatus::Ok;

  std::vector<Digit> digits(80);
  for (auto& reader_ptr : mGBTFrameContainers) {
    auto reader = reader_ptr.get();
    int readTimeBins = 0;
    for (int i = 0; i < mTimeBinsPerCall; ++i) {
      if (reader->getData(digits)) {
        for (auto& digi : digits) {
          CRU cru(digi.getCRU());
          const int roc = cru.roc();
          // TODO: OROC case needs subtraction of number of pad rows in IROC
          const PadRegionInfo& regionInfo = mMapper.getPadRegionInfo(cru.region());
          const PartitionInfo& partInfo = mMapper.getPartitionInfo(cru.partition());

          // row is local in region (CRU)
          const int row = digi.getRow();
          const int pad = digi.getPad();
          if (row == 255 || pad == 255)
            continue;

          int rowOffset = 0;
          switch (mPadSubset) {
            case PadSubset::ROC: {
              rowOffset = regionInfo.getGlobalRowOffset();
              rowOffset -= (cru.rocType() == RocType::OROC) * nRowIROC;
              break;
            }
            case PadSubset::Region: {
              break;
            }
            case PadSubset::Partition: {
              rowOffset = regionInfo.getGlobalRowOffset();
              rowOffset -= partInfo.getGlobalRowOffset();
              break;
            }
          }

          // modify row depending on the calibration type used
          const int timeBin = i; //digi.getTimeStamp();
          const float signal = digi.getChargeFloat();

          updateCRU(cru, row, pad, timeBin, signal);
          updateROC(roc, row + rowOffset, pad, timeBin, signal);
        }
        ++readTimeBins;
      }

      digits.clear();
    }

    // set status, don't overwrite decision
    if (status == ProcessStatus::Ok) {
      if (readTimeBins == 0) {
        return ProcessStatus::NoMoreData;
      } else if (readTimeBins < mTimeBinsPerCall) {
        status = ProcessStatus::Truncated;
      }
    }

    // notify that one raw reader processing finalized for this event
    endReader();
  }

  endEvent();
  ++mNevents;
  return status;
}

//______________________________________________________________________________
inline CalibRawBase::ProcessStatus CalibRawBase::processEventRawReader(int eventNumber)
{
  if (!mRawReaders.size())
    return ProcessStatus::NoReaders;
  resetEvent();

  // loop over raw readers, fill digits for 500 time bins and process
  // digits

  const int nRowIROC = mMapper.getNumberOfRowsROC(0);

  ProcessStatus status = ProcessStatus::Ok;

  mProcessedTimeBins = 0;
  size_t processedReaders = 0;
  bool hasData = false;

  int64_t lastEvent = 0;
  for (auto& reader_ptr : mRawReaders) {
    auto reader = reader_ptr.get();

    lastEvent = std::max(lastEvent, reader->getLastEvent());

    if (eventNumber >= 0) {
      reader->loadEvent(eventNumber);
      mPresentEventNumber = eventNumber;
    } else if (eventNumber == -1) {
      mPresentEventNumber = reader->loadNextEvent();
    } else if (eventNumber == -2) {
      mPresentEventNumber = reader->loadPreviousEvent();
    }

    o2::tpc::PadPos padPos;
    while (std::shared_ptr<std::vector<uint16_t>> data = reader->getNextData(padPos)) {
      if (!data)
        continue;

      mProcessedTimeBins = std::max(mProcessedTimeBins, data->size());

      CRU cru(reader->getRegion());
      const int roc = cru.roc();
      // TODO: OROC case needs subtraction of number of pad rows in IROC
      const PadRegionInfo& regionInfo = mMapper.getPadRegionInfo(cru.region());
      const PartitionInfo& partInfo = mMapper.getPartitionInfo(cru.partition());

      // row is local in region (CRU)
      const int row = padPos.getRow();
      const int pad = padPos.getPad();
      if (row == 255 || pad == 255)
        continue;

      int timeBin = 0;
      for (const auto& signalI : *data) {

        int rowOffset = 0;
        switch (mPadSubset) {
          case PadSubset::ROC: {
            rowOffset = regionInfo.getGlobalRowOffset();
            rowOffset -= (cru.rocType() == RocType::OROC) * nRowIROC;
            break;
          }
          case PadSubset::Region: {
            break;
          }
          case PadSubset::Partition: {
            rowOffset = regionInfo.getGlobalRowOffset();
            rowOffset -= partInfo.getGlobalRowOffset();
            break;
          }
        }

        // modify row depending on the calibration type used
        const float signal = float(signalI);
        //const FECInfo& fecInfo = mTPCmapper.getFECInfo(PadSecPos(roc, row, pad));
        //printf("Call update: %d, %d, %d, %d (%d), %.3f -- reg: %02d -- FEC: %02d, Chip: %02d, Chn: %02d\n", roc, row, pad, timeBin, i, signal, cru.region(), fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel());
        updateCRU(cru, row, pad, timeBin, signal);
        updateROC(roc, row + rowOffset, pad, timeBin, signal);
        ++timeBin;
        hasData = true;
      }
    }

    // notify that one raw reader processing finalized for this event
    endReader();
    ++processedReaders;
  }

  // set status, don't overwrite decision
  if (!hasData) {
    return ProcessStatus::NoMoreData;
  } else if (processedReaders < mRawReaders.size()) {
    status = ProcessStatus::Truncated;
  } else if (mPresentEventNumber == size_t(lastEvent)) {
    status = ProcessStatus::LastEvent;
  }

  endEvent();
  ++mNevents;
  return status;
}

//______________________________________________________________________________
inline CalibRawBase::ProcessStatus CalibRawBase::processEventRawReaderCRU(int eventNumber)
{
  if (!mRawReaderCRUManager.getNumberOfReaders())
    return ProcessStatus::NoReaders;
  resetEvent();

  mRawReaderCRUManager.init();

  const int nRowIROC = mMapper.getNumberOfRowsROC(0);

  ProcessStatus status = ProcessStatus::Ok;

  mProcessedTimeBins = 0;
  size_t processedReaders = 0;
  bool hasData = false;

  int64_t lastEvent = mRawReaderCRUManager.getNumberOfEvents() - 1;
  for (auto& reader_ptr : mRawReaderCRUManager.getReaders()) {
    auto reader = reader_ptr.get();

    LOG(INFO) << "Processing event number " << eventNumber << " (" << mNevents << ") - RawReader#: " << processedReaders << " ptr: " << reader;

    if (eventNumber >= 0) {
      reader->setEventNumber(eventNumber);
      mPresentEventNumber = eventNumber;
    } else if (eventNumber == -1) {
      if (mPresentEventNumber == std::numeric_limits<size_t>::max()) {
        mPresentEventNumber = 0;
      } else {
        mPresentEventNumber = (reader->getEventNumber() + 1) % reader->getNumberOfEvents();
      }
    } else if (eventNumber == -2) {
      auto readerNumber = reader->getEventNumber();
      if (readerNumber > 0) {
        mPresentEventNumber = readerNumber - 1;
      } else {
        mPresentEventNumber = reader->getNumberOfEvents() - 1;
      }
    }
    reader->setEventNumber(mPresentEventNumber);

    // process data
    reader->processLinks();

    const auto& cru = reader->getCRU();

    LOG(INFO) << "  Found ADC values: " << reader->getADCMap().size();
    // loop over pads
    for (const auto& pair : reader->getADCMap()) {
      const auto& padPos = pair.first;
      const auto& dataVector = pair.second;

      // TODO: fix this?
      mProcessedTimeBins = std::max(mProcessedTimeBins, dataVector.size());

      const int roc = cru.roc();
      // TODO: OROC case needs subtraction of number of pad rows in IROC
      const PadRegionInfo& regionInfo = mMapper.getPadRegionInfo(cru.region());
      const PartitionInfo& partInfo = mMapper.getPartitionInfo(cru.partition());

      // row is local in region (CRU)
      const int row = padPos.getRow();
      const int pad = padPos.getPad();
      if (row == 255 || pad == 255)
        continue;

      int timeBin = 0;
      for (const auto& signalI : dataVector) {

        int rowOffset = 0;
        switch (mPadSubset) {
          case PadSubset::ROC: {
            rowOffset = regionInfo.getGlobalRowOffset();
            rowOffset -= (cru.rocType() == RocType::OROC) * nRowIROC;
            break;
          }
          case PadSubset::Region: {
            break;
          }
          case PadSubset::Partition: {
            rowOffset = regionInfo.getGlobalRowOffset();
            rowOffset -= partInfo.getGlobalRowOffset();
            break;
          }
        }

        // modify row depending on the calibration type used
        const float signal = float(signalI);
        const FECInfo& fecInfo = mMapper.getFECInfo(PadROCPos(roc, row, pad));
        //printf("Call update: %d, %d (%d), %d, %d, %.3f -- cru: %03d, reg: %02d -- FEC: %02d, Chip: %02d, Chn: %02d\n", roc, row, rowOffset, pad, timeBin, signal, cru.number(), cru.region(), fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel());
        updateCRU(cru, row, pad, timeBin, signal);
        updateROC(roc, row + rowOffset, pad, timeBin, signal);
        ++timeBin;
        hasData = true;
      }
    }
    LOG(INFO) << "Found time bins: " << mProcessedTimeBins << "\n";

    reader->clearMap();

    // notify that one raw reader processing finalized for this event
    endReader();
    ++processedReaders;
  }
  // set status, don't overwrite decision
  if (!hasData) {
    return ProcessStatus::NoMoreData;
  } else if (processedReaders < mRawReaderCRUManager.getNumberOfReaders()) {
    status = ProcessStatus::Truncated;
  } else if (!isPresentEventComplete()) {
    status = ProcessStatus::IncompleteEvent;
  } else if (mPresentEventNumber == size_t(lastEvent)) {
    status = ProcessStatus::LastEvent;
  }

  LOG(INFO) << "Present event number : " << mPresentEventNumber;
  LOG(INFO) << "Last event           : " << lastEvent;
  LOG(INFO) << "Status               : " << int(status);

  endEvent();
  ++mNevents;
  return status;
}

} // namespace tpc

} // namespace o2
#endif
