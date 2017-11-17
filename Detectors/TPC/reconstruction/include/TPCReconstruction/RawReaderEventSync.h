// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RAWREADEREVENTSYNC_H_
#define O2_TPC_RAWREADEREVENTSYNC_H_

/// \file RawReaderEventSync.h
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <map>
#include <utility>

namespace o2 {
namespace TPC {

/// \class RawReaderEventSync.h
/// \brief Class for time synchronization of RawReader instances.
///
/// To synchronize an event across several readout links, the time offset given
/// in the event header is used. This class expects the distance of the event
/// time stamp with respect to the time stamp of the synchronization pattern (or
/// the first complete readout cycle in mode 2). This number corresponds to the
/// number of received GBT frames in the receiver card (T-RORC / CRU). Since 8
/// frames are needed to complete a full readout cycle, the event offset is set
/// to the next multiple of 8 (mFrames) of this distance, + one cycle to
/// guarantee always a complete set. This is done for each link individually and
/// the maximum of those time offsets is kept for every event. With that, each
/// individual link can use the same (max.) time stamp to decode the data
/// synchronously.
///
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)
class RawReaderEventSync {
  public:

    /// Default constructor
    RawReaderEventSync() : mFrames(8), mMaxOffset() {};

    /// Copy constructor
    RawReaderEventSync(const RawReaderEventSync& other) = default;

    /// Destructor
    ~RawReaderEventSync() = default;

    /// Add the event offset for a given link
    /// \param event Event number
    /// \param time Offset
    void addEventOffset(int event, long time);

    /// Returns the event ofset for a given link
    /// \param event Event number
    /// \return Event offset
    long getEventOffset(int event) { return mMaxOffset[event]; };

  private:

    int mFrames;                        ///< Number of GBT Frames for complete readout cycle
    std::map<int, long> mMaxOffset;     ///< maximum offset for each event, to be used by all links
};

inline
void RawReaderEventSync::addEventOffset(int event, long time) {

  // set time offset to next multiple of mFrames (+ mFrames)
  const long iTimeOffset = time + (mFrames - (time%mFrames)) + mFrames;

  mMaxOffset[event] = std::max(mMaxOffset[event],iTimeOffset);
}

}
}
#endif
