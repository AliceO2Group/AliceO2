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

/// \file RecoContainer.h
/// \brief Reconstruction container for EMCAL Cells and LEDMONs
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since May 30, 2023

#include <array>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <gsl/span>
#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <DataFormatsEMCAL/Cell.h>
#include <EMCALBase/TriggerMappingV2.h>
#include <EMCALReconstruction/FastORTimeSeries.h>
#include <EMCALReconstruction/TRUDataHandler.h>

#ifndef ALICEO2_EMCAL_RECOCONTAINER_H
#define ALICEO2_EMCAL_RECOCONTAINER_H

namespace o2::emcal
{
/// \struct RecCellInfo
/// \brief Container class for cell information for merging
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
  int mDDLID;                ///< DDL of the channel (for monitoring)
  int mHWAddressLG;          ///< HW address of LG (for monitoring)
  int mHWAddressHG;          ///< HW address of HG (for monitoring)
};

/// \class EventContainer
/// \brief Containter of cells for a given event
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALreconstruction
/// \since May 30, 2023
///
/// The EventContainer is part of the reco container and keeps the cell / LEDMON / TRU
/// data of a given event (trigger) defined by the BC and orbit in the raw data header.
/// In case the gain type merging is activated the event container performs on-the-fly
/// merging of high- and low-gain data for cells and LEDMONS keeping only high- or
/// low-gain and preferring the high-gain due to better resolution if not saturated.
///
/// Error handling:
/// The EventContainer can raise a TRUIndexException in case TRU information with an
/// unexpected index (>= 52) is added / requested.
class EventContainer
{
 public:
  /// \class TRUIndexException
  /// \brief Handler for access of TRU data with invalid TRU index
  /// \ingroup EMCALreconstruction
  class TRUIndexException final : public std::exception
  {
   public:
    /// \brief Constructor
    /// \param index TRU index raising the exception
    TRUIndexException(std::size_t index);

    /// \brief Destructor
    ~TRUIndexException() noexcept final = default;

    /// \brief Get the error message of the exception
    /// \return Error message
    const char* what() const noexcept final { return mMessage.data(); }

    /// \brief Get the TRU index raising the exception
    /// \return TRU index
    std::size_t getIndex() const { return mIndex; }

    /// \brief Print error message on stream
    /// \param stream Stream to print on
    void printStream(std::ostream& stream) const;

   private:
    std::size_t mIndex;   ///< TRU index raising the exception
    std::string mMessage; ///< Buffer for error message
  };

  /// \brief Constructor
  EventContainer() = default;

  /// \brief Constructor, setting interaction record
  /// \param currentIR Interaction record of the trigger
  EventContainer(const o2::InteractionRecord& currentIR);

  /// \brief Destructor
  ~EventContainer() = default;

  /// \brief Set trigger bits of the interaction
  /// \param triggerbits Trigger bits
  void setTriggerBits(uint64_t triggerbits) { mTriggerBits = triggerbits; }

  /// \brief Set interaction record
  /// \param currentIR Interaction record of the trigger
  void setInteractionRecord(const o2::InteractionRecord& currentIR) { mInteractionRecord = currentIR; }

  /// \brief Get trigger bits of the interaction
  /// \return Trigger bits
  uint64_t getTriggerBits() const { return mTriggerBits; }

  /// \brief Get interaction record of the event
  /// \return Interaction record connected to this event
  const o2::InteractionRecord& getInteractionRecord() const { return mInteractionRecord; }

  /// \brief Get cells in container
  /// \return List of cells in container
  const gsl::span<const RecCellInfo> getCells() const { return mCells; }

  /// \brief Get LEDMONs in container
  /// \return List of LEDMONs
  const gsl::span<const RecCellInfo> getLEDMons() const { return mLEDMons; }

  /// \brief Get the number of cells in the event
  /// \return Number of cells
  int getNumberOfCells() const { return mCells.size(); }

  /// \brief Get the number of LEDMONs in the event
  /// \return Number of LEDMONs
  int getNumberOfLEDMONs() const { return mLEDMons.size(); }

  /// \brief Read and write access TRU data of a given TRU
  /// \param truIndex Index of the TRU
  /// \return TRU data handler for the TRU
  /// \throw TRUIndexException in case the TRU index is invalid (>= 52)
  TRUDataHandler& getTRUData(std::size_t truIndex);

  /// \brief Read-only access TRU data of a given TRU
  /// \param truIndex Index of the TRU
  /// \return TRU data handler for the TRU
  /// \throw TRUIndexException in case the TRU index is invalid (>= 52)
  const TRUDataHandler& readTRUData(std::size_t truIndex) const;

  /// \brief Access to container with FastOR time series
  /// \return Container with time series
  const std::unordered_map<uint16_t, FastORTimeSeries>& getTimeSeriesContainer() const { return mL0FastORs; }

  /// \brief Add cell information to the event container
  /// \param tower Tower ID
  /// \param energy Cell energy
  /// \param time Cell time
  /// \param celltype Cell type (high gain or low gain)
  /// \param hwaddress Hardware address
  /// \param ddlID ID of the DDL
  /// \param doMergeHGLG If true merge with existing HG/LG cell
  ///
  /// In case of merge mode the priory is given to the HG digitizer (better resolution).
  /// As long as the energy is not in the saturation region (approx 16 GeV) the HG is selected,
  /// otherwise the LG digit is used.
  void setCell(int tower, double energy, double time, ChannelType_t celltype, int hwaddress, int ddlID, bool doMergeHGLG)
  {
    setCellCommon(tower, energy, time, celltype, false, hwaddress, ddlID, doMergeHGLG);
  }

  /// \brief Add LEDMON information to the event container
  /// \param tower LEDMON ID
  /// \param energy LEDMON energy
  /// \param time LEDMON time
  /// \param celltype LEDMON type (high gain or low gain)
  /// \param hwaddress Hardware address
  /// \param ddlID ID of the DDL
  /// \param doMergeHGLG If true merge with existing HG/LG LEDMON
  ///
  /// In case of merge mode the priory is given to the HG digitizer (better resolution).
  /// As long as the energy is not in the saturation region (approx 16 GeV) the HG is selected,
  /// otherwise the LG digit is used.
  void setLEDMONCell(int tower, double energy, double time, ChannelType_t celltype, int hwaddress, int ddlID, bool doMergeHGLG)
  {
    setCellCommon(tower, energy, time, celltype, true, hwaddress, ddlID, doMergeHGLG);
  }

  /// \brief Add bunch of time series to the container
  /// \param fastORAbsID Absolute ID of the FastOR
  /// \param starttime Start time of the bunch
  /// \param timesamples Time samples of the bunch in time-reversed format
  ///
  /// In case a TimeSeries is already present for the given FastOR abs. ID in the container
  /// the bunch is added to this, otherwise a new TimeSeries is added with the ADCs of the
  /// bunch.
  void setFastOR(uint16_t fastORAbsID, uint8_t starttime, const gsl::span<const uint16_t> timesamples);

  /// \brief Sort Cells / LEDMONs in container according to tower / module ID
  /// \param isLEDmon Switch between Cell and LEDMON
  void sortCells(bool isLEDmon);

 private:
  /// \brief Common handler for adding cell/LEDMON information to the event container
  /// \param tower Tower / LEDMON ID
  /// \param energy Energy
  /// \param time Time
  /// \param celltype Digitizer type (high gain or low gain)
  /// \param hwaddress Hardware address
  /// \param ddlID ID of the DDL
  /// \param doMergeHGLG Switch for merge mode
  void setCellCommon(int tower, double energy, double time, ChannelType_t celltype, bool isLEDmon, int hwaddress, int ddlID, bool doMergeHGLG);

  /// \brief Check whether the energy is in the saturation limit
  /// \return True if the energy is in the saturation region, false otherwise
  bool isCellSaturated(double energy) const;

  /// \brief Initialize the TRU handlers
  void initTRUs();

  o2::InteractionRecord mInteractionRecord;                       ///< Interaction record of the event
  uint64_t mTriggerBits = 0;                                      ///< Trigger bits of the event
  std::vector<RecCellInfo> mCells;                                ///< Container of cells in event
  std::vector<RecCellInfo> mLEDMons;                              ///< Container of LEDMONs in event
  std::array<TRUDataHandler, TriggerMappingV2::ALLTRUS> mTRUData; ///< TRU status
  std::unordered_map<uint16_t, FastORTimeSeries> mL0FastORs;      ///< L0 FastOR time series
};

/// \class RecoContainer
/// \brief Handler for cells/LEDMONS/Trigger data in timeframes
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALreconstruction
/// \since May 30, 2023
///
/// The RecoContainer handles the cell/LEDMON/trigger data of all events of a given
/// timeframe during the reconstruction. Event data are handled internally via the
/// EventContainer, where the RecoContainer provides access to.
///
/// Error handling:
/// The RecoContainer can raise an InteractionNotFoundException in case read access
/// is requested for an interaction based on the o2::InteractionRecord which is not
/// for which no data was inserted into the container.
class RecoContainer
{
 public:
  /// \class InteractionNotFoundException
  /// \brief Handling of access to trigger interaction record not present in container
  class InteractionNotFoundException final : public std::exception
  {
   public:
    /// \brief Constructor
    /// \param currentIR Interaction record raising the exception
    InteractionNotFoundException(const o2::InteractionRecord& currentIR) : mCurrentIR(currentIR)
    {
      mMessage = "Interaction record not found: Orbit " + std::to_string(mCurrentIR.orbit) + ", BC " + std::to_string(mCurrentIR.bc);
    }

    /// \brief Destructor
    ~InteractionNotFoundException() noexcept final = default;

    /// \brief Get error message of the exception
    /// \return Error message
    const char* what() const noexcept final
    {
      return mMessage.data();
    };

    /// \brief Get interaction record raising the exception
    /// \return Interaction record
    const o2::InteractionRecord& getInteractionRecord() const { return mCurrentIR; }

   private:
    o2::InteractionRecord mCurrentIR; ///< Interaction record raising the exception
    std::string mMessage;             ///< Error message
  };

  /// \brief Constructor
  RecoContainer() = default;

  /// \brief Destructor
  ~RecoContainer() = default;

  /// \brief Get container for trigger
  /// \param currentIR Interaction record of the trigger
  /// \return Container for trigger (creating new container if not yet present)
  EventContainer& getEventContainer(const o2::InteractionRecord& currentIR);

  /// \brief Get container for trigger (read-only)
  /// \param currentIR Interaction record of the trigger
  /// \return Container for trigger
  /// \throw InteractionNotFoundException if interaction record is not present
  const EventContainer& getEventContainer(const o2::InteractionRecord& currentIR) const;

  /// \brief Get sorted vector interaction records of triggers in container
  /// \return Sorted vector of container
  std::vector<o2::InteractionRecord> getOrderedInteractions() const;

  /// \brief Get number of events in container
  /// \return Number of events
  std::size_t getNumberOfEvents() const { return mEvents.size(); }

  /// \brief Clear container
  void reset() { mEvents.clear(); }

 private:
  std::unordered_map<o2::InteractionRecord, EventContainer> mEvents; ///< Containers in event
};

/// \class RecoContainerReader
/// \brief Iterator over reco containers
/// \author Markus Fasel <narkus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALreconstruction
/// \since May 30, 2023
///
/// The RecoContainerReader iterates over the events stored in the RecoContainer in
/// a time-ordered sequence. The function hasNext checks whether there are more
/// events to iterate over, while nextEvent provides access to the next event.
///
/// Error handling:
/// The RecoContainerReader can raise an InvalidAccessException in case access is tried to
/// invalid data.
class RecoContainerReader
{
 public:
  /// \class InvalidAccessException
  /// \brief Handling of access to objects beyond container boundary
  class InvalidAccessException : public std::exception
  {
   public:
    /// \brief Constructor
    InvalidAccessException() = default;

    /// \brief Destructor
    ~InvalidAccessException() noexcept final = default;

    /// \brief Create error message
    /// \return Error message
    const char* what() const noexcept final { return "Access to invalid element in reco container"; }
  };

  /// \brief Constructor
  /// \param container Container to be iterated over
  RecoContainerReader(RecoContainer& container);
  RecoContainerReader(RecoContainer&& container) = delete;

  /// \brief Destructor
  ~RecoContainerReader() = default;

  /// \brief Get the next event in container
  /// \return Next event in reco container (ordered)
  /// \throw InvalidAccessException
  EventContainer& nextEvent();

  /// \brief Check whehter there are more events in the container
  /// \return True if the event is not the last event, false otherwise.
  bool hasNext() const { return mCurrentEvent < mOrderedInteractions.size(); }

  /// \brief Get the number of events in the container
  /// \return Number of events
  std::size_t getNumberOfEvents() const { return mDataContainer.getNumberOfEvents(); }

 private:
  RecoContainer& mDataContainer;                           ///< Reference to container which is iterated over
  std::vector<o2::InteractionRecord> mOrderedInteractions; ///< Ordered list of the interaction records
  std::size_t mCurrentEvent = 0;                           ///< Index of the current event in the sorted event container
};

} // namespace o2::emcal

#endif