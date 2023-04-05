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
#ifndef ALICEO2_EMCAL_EVENTHANDLER_H_
#define ALICEO2_EMCAL_EVENTHANDLER_H_

#include <cstdint>
#include <exception>
#include <iterator>
#include <gsl/span>
#include <vector>
#include "Rtypes.h"
#include "fmt/format.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/EventData.h"
#include "DataFormatsEMCAL/MCLabel.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace emcal
{

/// \class EventHandler
/// \brief Handler for EMCAL event data
/// \ingroup EMCALbase
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since March 1st, 2020
///
/// #Initializing the event handler
///
/// The event handler is initialized with EMCAL data (clusters and / or cells) from an entire
/// time frame. In order to identify objects within the time frame belonging to the same collision
/// the event handler needs to be always initialized with the object and the corresponding range
/// of trigger records. The event handler supports full and partial event information. Different
/// constructors allow the initialization either as full event (with cluster and cell information)
/// or with either cluster or cell information.
///
/// Example (full event):
/// ~~~.{cxx}
/// auto cells = ctx.inputs().get<gsl::span<o2::emcal::Cell>("cells");
/// auto celltriggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("CellsTR");
/// auto cluster = ctx.inputs().get<gsl::span<o2::emcal::Cluster>>("clusters");
/// auto clustertriggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("ClustersTR");
/// auto cellIndices = ctx.inputs().get<gsl::span<int>>("cellindices");
/// auto citriggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("CITR");
/// o2::emcal::EventHandler<o2::emcal::Cell> evhandler(clusters, cellIndices, cells, clustertriggers, citriggers, celltriggers);
/// ~~~.{cxx}
///
/// #Getting the event data
///
/// Access to the data is possible either directly via the buildEvent method for a certain event
/// number or via an iterator iterating over the triggers in the Trigger record collection. For the
/// buildEvent method the event ID must be in a valid range.
///
/// Example (buildEvent):
/// ~~~.{cxx}
/// o2::emcal::EventHander<o2::emcal::Cell> handler(clusters, cellIndices, cells, clustertriggers, citriggers, celltriggers);
/// for(int iev = 0; iev < handler.getNumberOfEvents(); iev++) {
///   auto event = handler.buildEvent(iev);
///   // Do something with the event ...
///}
/// ~~~.{cxx}
///
/// Example (iterator):
/// ~~~.{cxx}
/// o2::emcal::EventHander<o2::emcal::Cell> handler(clusters, cellIndices, cells, clustertriggers, citriggers, celltriggers);
/// for(const auto &ev : handler) {
///   // Do something with the event ...
/// }
/// ~~~.{cxx}
///
/// #Supported cell input types
/// The event handler can support o2::emcal::Digit or o2::emcal::Cell as input type for cells.
template <class CellInputType>
class EventHandler
{
 public:
  using TriggerRange = gsl::span<const TriggerRecord>;
  using ClusterRange = gsl::span<const Cluster>;
  using CellIndexRange = gsl::span<const int>;
  using CellRange = gsl::span<const CellInputType>;

  /// \class RangeException
  /// \brief Exception handling errors due to exceeding the range of triggers handled by the handler
  class RangeException final : public std::exception
  {
   public:
    /// \brief Constructor defining the error
    /// \param eventID Event ID responsible for the exception
    /// \param maxEvents Maximum number of events handled by the handler
    RangeException(int eventID, int maxEvents) : std::exception(),
                                                 mEventID(eventID),
                                                 mMaxEvents(maxEvents),
                                                 mErrorMessage()
    {
      mErrorMessage = fmt::format("Exceeding range: %d, max %d", mEventID, mMaxEvents);
    }

    /// \brief Destructor
    ~RangeException() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message connected to this exception
    const char* what() const noexcept final { return mErrorMessage.data(); }

    /// \brief Get the ID of the event raising the exception
    /// \return Event ID
    int getEventID() const { return mEventID; }

    /// \brief Get the maximum number of events handled by the event handler
    /// \return Max. number of event
    int getMaxNumberOfEvents() const { return mMaxEvents; }

   private:
    int mEventID = 0;          ///< Event ID raising the exception
    int mMaxEvents = 0;        ///< Max. number of events handled by this event handler
    std::string mErrorMessage; ///< Error message
  };

  /// \class NotInitializedException
  /// \brief Exception handling unitialized event handler
  class NotInitializedException final : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    NotInitializedException() = default;

    /// \brief Destructor
    ~NotInitializedException() noexcept final = default;

    /// \brief Creating error message of the exception
    /// \return Error message of the exception
    const char* what() const noexcept final { return "EventHandler not initialized"; }
  };

  /// \class InteractionRecordInvalidException
  /// \brief Error handling in case the interaction records from various sources do not match
  class InteractionRecordInvalidException final : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    /// \param irclusters Interaction reccord from the cluster trigger record container
    /// \param ircells Interaction record from the cell trigger record container
    InteractionRecordInvalidException(const InteractionRecord& irclusters, const InteractionRecord& ircells) : mInteractionRecordClusters(irclusters),
                                                                                                               mInteractionRecordCells(ircells)
    {
    }

    /// \brief Destructor
    ~InteractionRecordInvalidException() noexcept final = default;

    /// \brief Creating error message of the exception
    /// \return Error message of the exception
    const char* what() const noexcept final { return "Interaction records for clusters and cells not matching"; }

    /// \brief Get the interaction record for the cluster subevent
    /// \return Interaction record for the cluster subevent
    const InteractionRecord& getInteractionRecordClusters() const { return mInteractionRecordClusters; }

    /// \brief Get the interaction record for the cells subevent
    /// \return Interaction record of the cell subevent
    const InteractionRecord& getInteractionRecordCells() const { return mInteractionRecordCells; }

   private:
    InteractionRecord mInteractionRecordClusters; ///< Interaction record for clusters
    InteractionRecord mInteractionRecordCells;    ///< Interaction record for cells
  };

  /// \class TriggerBitsInvalidException
  /// \brief Error handling in case the trigger bits from various sources do not match
  class TriggerBitsInvalidException final : public std::exception
  {
   public:
    /// \brief Constructor initializing the exception
    /// \param bitsclusters Trigger bits from the cluster trigger record container
    /// \param bitscells Trigger bits from the cell trigger record container
    TriggerBitsInvalidException(uint64_t bitsclusters, uint64_t bitscells) : mTriggerBitsClusters(bitsclusters),
                                                                             mTriggerBitsCells(bitscells)
    {
    }

    /// \brief Destructor
    ~TriggerBitsInvalidException() noexcept final = default;

    /// \brief Creating error message of the exception
    /// \return Error message of the exception
    const char* what() const noexcept final { return "Tigger bits for clusters and cells not matching"; }

    /// \brief Get the trigger bits for the cluster subevent
    /// \return Trigger bits for the cluster subevent
    uint64_t getTriggerBitsClusters() const { return mTriggerBitsClusters; }

    /// \brief Get the trigger bits for the cells subevent
    /// \return Trigger bits of the cell subevent
    uint64_t getTriggerBitsCells() const { return mTriggerBitsCells; }

   private:
    uint64_t mTriggerBitsClusters; ///< Trigger bits from cluster trigger record container
    uint64_t mTriggerBitsCells;    ///< Trigger bits from cell trigger record container
  };

  /// \class EventIterataor
  /// \brief Iterator of the event handler
  ///
  /// The iterator is defined as bi-directional iterator, can iterate in forward or
  /// backward direction.
  class EventIterator : public std::iterator_traits<EventData<CellInputType>>
  {
   public:
    /// \brief Constructor, initializing the iterator
    /// \param handler Event handler to iterate over
    /// \param eventID Event ID from which to start the iteration
    /// \param forward Direction of the iteration (true = forward)
    EventIterator(const EventHandler& handler, int eventID, bool forward);

    /// \brief Copy constructor
    /// \param other Reference for the copy
    EventIterator(const EventIterator& other) = default;

    /// \brief Assignment operator
    /// \param other Reference for assignment
    /// \return Iterator after assignment
    EventIterator& operator=(const EventIterator& other) = default;

    /// \brief Destructor
    ~EventIterator() = default;

    /// \brief Check for equalness
    /// \param rhs Iterator to compare to
    /// \return True if iterators are the same, false otherwise
    ///
    /// Check is done on same event handler, event ID and direction
    bool operator==(const EventIterator& rhs) const;

    /// \brief Check for not equalness
    /// \param rhs Iterator to compare to
    /// \return True if iterators are different, false otherwise
    ///
    /// Check is done on same event handler, event ID and direction
    bool operator!=(const EventIterator& rhs) const { return !(*this == rhs); }

    /// \brief Prefix incrementation operator
    /// \return Iterator after incrementation
    EventIterator& operator++();

    /// \brief Postfix incrementation operator
    /// \return Iterator before incrementation
    EventIterator operator++(int);

    /// \brief Prefix decrementation operator
    /// \return Iterator after decrementation
    EventIterator& operator--();

    /// \brief Postfix decrementation operator
    /// \return Iterator before decrementation
    EventIterator operator--(int);

    /// \brief Get pointer to the current event
    /// \return Pointer to the current event
    EventData<CellInputType>* operator*() { return &mCurrentEvent; }

    /// \brief Get reference to the current event
    /// \return Reference to the current event of the iterator
    EventData<CellInputType>& operator&() { return mCurrentEvent; }

    /// \brief Get the index of the current event
    /// \return Index of the current event
    int current_index() const { return mEventID; }

   private:
    const EventHandler& mEventHandler;      ///< Event handler connected to the iterator
    EventData<CellInputType> mCurrentEvent; ///< Cache for current event
    int mEventID = 0;                       ///< Current event ID within the event handler
    bool mForward = true;                   ///< Iterator direction (forward or backward)
  };

  /// \brief Dummy constructor
  EventHandler() = default;

  /// \brief Constructor, initializing event handler for cells only
  /// \param cells Container with cells for the full time frame
  /// \param triggers Container with the trigger records corresponding to the cell container
  EventHandler(CellRange cells, TriggerRange triggers);

  /// \brief Constructor, initializing event handler for clusters only
  /// \param clusters Container with clusters for the full time frame
  /// \param cellIndices Container with cell indices used by the clusters in the cluster container
  /// \param triggerCluster Container with trigger records corresponding to the cluster container
  /// \param triggersCellIndex Container with trigger records corresponding to the cell index container
  EventHandler(ClusterRange clusters, CellIndexRange cellIndices, TriggerRange triggersCluster, TriggerRange triggersCellIndex);

  /// \brief Constructor, initializing event handler for clusters and cells
  /// \param clusters Container with clusters for the full time frame
  /// \param cellIndices Container with cell indices used by the clusters in the cluster container
  /// \param cells Container with cells for the full time frame
  /// \param triggerCluster Container with trigger records corresponding to the cluster container
  /// \param triggersCellIndex Container with trigger records corresponding to the cell index container
  EventHandler(ClusterRange clusters, CellIndexRange cellIndices, CellRange cells, TriggerRange triggersCluster, TriggerRange triggersCellIndex, TriggerRange triggersCell);

  /// \brief Destructor
  ~EventHandler() = default;

  /// \brief Get forward start iterator
  /// \return Start iterator
  EventIterator begin() const { return EventIterator(*this, 0, true); }

  /// \brief Get forward end iteration marker
  /// \return Iteration end marker
  EventIterator end() const { return EventIterator(*this, getNumberOfEvents(), true); }

  /// \brief Get backward start iterator
  /// \return Start iterator
  EventIterator rbegin() const { return EventIterator(*this, getNumberOfEvents() - 1, false); };

  /// \brief Get backward end iteration marker
  /// \return Iteration end marker
  EventIterator rend() const { return EventIterator(*this, -1, false); };

  /// \brief Get the number of events handled by the event handler
  /// \return Number of events
  /// \throw NotInitializedException If the event handler is not initialized
  int getNumberOfEvents() const;

  /// \brief Get the interaction record for the given event
  /// \param eventID ID of the event in timeframe
  /// \return Interaction record for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw InteractionRecordInvalidException If the interaction record for the same event from differnt containers has different content
  /// \throw NotInitializedException in case the event handler is not initialized
  ///
  /// Requires at least on interaction record container to be set (clusters, cells or cell indices).
  InteractionRecord getInteractionRecordForEvent(int eventID) const;

  /// \brief Get the interaction record for the given event
  /// \param eventID ID of the event in timeframe
  /// \return Interaction record for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw TriggerBitsInvalidException in case the trigger bits for the same event from differnt containers are different
  /// \throw NotInitializedException in case the event handler is not initialized
  ///
  /// Requires at least on interaction record container to be set (clusters, cells or cell indices).
  uint64_t getTriggerBitsForEvent(int eventID) const;

  /// \brief Get range of clusters belonging to the given event
  /// \param eventID ID of the event
  /// \return Cluster range for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw NotInitializedException in case the event handler is not initialized for clusters
  const ClusterRange getClustersForEvent(int eventID) const;

  /// \brief Get range of cells belonging to the given event
  /// \param eventID ID of the event
  /// \return Cell range for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw NotInitializedException in case the event handler is not initialized for cell
  const CellRange getCellsForEvent(int eventID) const;

  /// \brief Get vector of MC labels belonging to the given event
  /// \param eventID ID of the event
  /// \return vector of MC labels for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw NotInitializedException in case the event handler is not initialized for cell
  std::vector<gsl::span<const o2::emcal::MCLabel>> getCellMCLabelForEvent(int eventID) const;

  /// \brief Get range of cluster cell indices belonging to the given event
  /// \param eventID ID of the event
  /// \return Cluster cell index range for the event
  /// \throw RangeException in case the required event ID exceeds the maximum number of events
  /// \throw NotInitializedException in case the event handler is not initialized for clusters/cellIndices
  const CellIndexRange getClusterCellIndicesForEvent(int eventID) const;

  /// \brief Check whether event handler has cluster data
  /// \return True in case trigger records connected to the cluster container are found
  bool hasClusters() const { return mTriggerRecordsClusters.size() > 0; }

  /// \brief Check whether event handler has cell index data
  /// \return True in case trigger records connected to the cell index container are found
  bool hasClusterIndices() const { return mTriggerRecordsCellIndices.size() > 0; }

  /// \brief Check whether event handler has cell data
  /// \return True in case trigger records connected to the cell container are found
  bool hasCells() const { return mTriggerRecordsCells.size() > 0; }

  /// \brief Setting data at cluster level
  /// \param clusters Container with clusters for the full time frame
  /// \param cellIndices Container with cell indices used by the clusters in the cluster container
  /// \param triggerCluster Container with trigger records corresponding to the cluster container
  /// \param triggersCellIndex Container with trigger records corresponding to the cell index container
  void setClusterData(ClusterRange clusters, CellIndexRange cellIndices, TriggerRange triggersCluster, TriggerRange triggersCellIndex)
  {
    mClusters = clusters;
    mClusterCellIndices = cellIndices;
    mTriggerRecordsClusters = triggersCluster;
    mTriggerRecordsCellIndices = triggersCellIndex;
  }

  /// \brief Setting the data at cell level
  /// \param cells Container for cells within the timeframe
  /// \param triggers Container with trigger records corresponding to the cell container
  void setCellData(CellRange cells, TriggerRange triggers)
  {
    mCells = cells;
    mTriggerRecordsCells = triggers;
  }

  /// \brief Setting the pointer for the MCTruthContainer for cells
  /// \param mclabels Pointer to the MCTruthContainer for cells in timeframe
  void setCellMCTruthContainer(const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>* mclabels)
  {
    mCellLabels = mclabels;
  }

  /// \brief Reset containers with empty ranges
  void reset();

  /// \brief Build event information for a given event number within the timeframe
  /// \param eventID Number of the event within the timeframe
  /// \return Event data for the given event
  /// \throw RangeException in case the requested event ID is outside the range
  /// \throw InteractionRecordInvalidException in case the interaction records from cells and clusters do not match
  /// \throw NotInitializedException in case the event handler is not initialized
  ///
  /// Building new EMCAL event for a certain event number. Based on the input data
  /// specified in the constructors or the different setters the event contains either
  /// all data consisting of clusters, cell indices and cells, or either clusters and
  /// cell indices or cells. eventID must be a valid event within the list of triggers.
  /// In case the full event is built the trigger records from the various contributors
  /// must match.
  EventData<CellInputType> buildEvent(int eventID) const;

 private:
  /// \brief Compare two interaction records for equalness
  /// \param lhs First interaction record
  /// \param rhs Second interaction records
  /// \return True if the interaction records have the same BC and orbit ID (same collision), false otherwise
  bool compareInteractionRecords(const InteractionRecord& lhs, const InteractionRecord& rhs) const;

  TriggerRange mTriggerRecordsClusters;    ///< Trigger record for cluster type
  TriggerRange mTriggerRecordsCellIndices; ///< trigger record for cluster cell index type
  TriggerRange mTriggerRecordsCells;       ///< Trigger record for cell type

  ClusterRange mClusters;                                                             /// container for clusters in timeframe
  CellIndexRange mClusterCellIndices;                                                 /// container for cell indices in timeframe
  CellRange mCells;                                                                   /// container for cells in timeframe
  const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>* mCellLabels = nullptr; /// pointer to the MCTruthContainer for cells in timeframe

  ClassDefNV(EventHandler, 1);
};

} // namespace emcal
} // namespace o2

#endif // ALICEO2_EMCAL_EVENTHANDLER_H__
