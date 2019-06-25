// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALBase/Geometry.h"
#include "EMCALCalib/BadChannelMap.h"

#include "FairLogger.h"

#include <TH2.h>

#include <iostream>

using namespace o2::emcal;

void BadChannelMap::addBadChannel(unsigned short channelID, MaskType_t mask)
{
  switch (mask) {
    case MaskType_t::GOOD_CELL:
      mBadCells.reset(channelID);
      mWarmCells.reset(channelID);
      mDeadCells.reset(channelID);
      break;
    case MaskType_t::WARM_CELL:
      mBadCells.reset(channelID);
      mWarmCells.set(channelID);
      mDeadCells.reset(channelID);
      break;
    case MaskType_t::BAD_CELL:
      mBadCells.set(channelID);
      mWarmCells.reset(channelID);
      mDeadCells.reset(channelID);
      break;
    case MaskType_t::DEAD_CELL:
      mDeadCells.set(channelID);
      mBadCells.reset(channelID);
      mWarmCells.reset(channelID);
      break;
  };
}

BadChannelMap::MaskType_t BadChannelMap::getChannelStatus(unsigned short channelID) const
{
  auto status = MaskType_t::GOOD_CELL;
  if (mDeadCells.test(channelID))
    status = MaskType_t::DEAD_CELL;
  else if (mBadCells.test(channelID))
    status = MaskType_t::BAD_CELL;
  else if (mWarmCells.test(channelID))
    status = MaskType_t::WARM_CELL;
  return status;
}

TH2* BadChannelMap::getHistogramRepresentation() const
{
  const int MAXROWS = 208,
            MAXCOLS = 96;
  auto hist = new TH2S("badchannelmap", "Bad Channel Map", MAXCOLS, -0.5, double(MAXCOLS) - 0.5, MAXROWS, -0.5, double(MAXROWS) - 0.5);
  hist->SetDirectory(nullptr);
  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return hist;
  }

  for (size_t cellID = 0; cellID < mBadCells.size(); cellID++) {
    int value(0);
    if (mBadCells.test(cellID))
      value = 1;
    else if (mDeadCells.test(cellID))
      value = 2;
    else if (mWarmCells.test(cellID))
      value = 3;
    if (value) {
      auto position = geo->GlobalRowColFromIndex(cellID);
      hist->Fill(std::get<1>(position), std::get<0>(position), value);
    }
  }
  return hist;
}

BadChannelMap& BadChannelMap::operator+=(const BadChannelMap& rhs)
{
  for (size_t cellID = 0; cellID < mDeadCells.size(); cellID++) {
    if (rhs.mDeadCells.test(cellID)) {
      mDeadCells.set(cellID);
      if (mBadCells.test(cellID))
        mBadCells.reset(cellID);
      if (mWarmCells.test(cellID))
        mWarmCells.reset(cellID);
    }
    if (rhs.mBadCells.test(cellID) && !mDeadCells.test(cellID)) {
      mBadCells.set(cellID);
      if (mWarmCells.test(cellID))
        mWarmCells.reset(cellID);
    }
    if (rhs.mWarmCells.test(cellID) && !(mBadCells.test(cellID) || mDeadCells.test(cellID)))
      mWarmCells.set(cellID);
  }
  return *this;
}

bool BadChannelMap::operator==(const BadChannelMap& other) const
{
  return mBadCells == other.mBadCells && mWarmCells == other.mWarmCells && mDeadCells == other.mDeadCells;
}

void BadChannelMap::PrintStream(std::ostream& stream) const
{
  // first sort bad channel IDs
  stream << "Number of bad cells:  " << mBadCells.count() << "\n";
  stream << "Number of dead cells: " << mDeadCells.count() << "\n";
  stream << "Number of warm cells: " << mWarmCells.count() << "\n";
  for (size_t cellID = 0; cellID < mBadCells.size(); cellID++) {
    if (mBadCells.test(cellID))
      stream << "Channel " << cellID << ": Bad cell\n";
    if (mDeadCells.test(cellID))
      stream << "Channel " << cellID << ": Dead cell\n";
    if (mWarmCells.test(cellID))
      stream << "Channel " << cellID << ": Warm cell\n";
  }
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const BadChannelMap& bcm)
{
  bcm.PrintStream(stream);
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const BadChannelMap::MaskType_t& masktype)
{
  switch (masktype) {
    case BadChannelMap::MaskType_t::GOOD_CELL:
      stream << "Good cell";
      break;
    case BadChannelMap::MaskType_t::WARM_CELL:
      stream << "Warm cell";
      break;
    case BadChannelMap::MaskType_t::BAD_CELL:
      stream << "Bad cell";
      break;
    case BadChannelMap::MaskType_t::DEAD_CELL:
      stream << "Dead cell";
      break;
  };
  return stream;
}