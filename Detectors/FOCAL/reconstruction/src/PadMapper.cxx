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

#include <iostream>
#include <FOCALReconstruction/PadMapper.h>

using namespace o2::focal;

PadMapper::PadMapper()
{
  initInverseMapping();
}

std::tuple<unsigned int, unsigned int> PadMapper::getRowColFromChannelID(unsigned int channelID) const
{
  if (channelID >= NCHANNELS) {
    throw ChannelIDException(channelID);
  }
  return mInverseMapping[channelID];
}
unsigned int PadMapper::getRow(unsigned int channelID) const
{
  return std::get<1>(getRowColFromChannelID(channelID));
}
unsigned int PadMapper::getColumn(unsigned int channelID) const
{
  return std::get<0>(getRowColFromChannelID(channelID));
}

unsigned int PadMapper::getChannelID(unsigned int col, unsigned int row) const
{
  if (col >= NCOLUMN || row >= NROW) {
    throw PositionException(col, row);
  }
  return mMapping[col][row];
}

void PadMapper::initInverseMapping()
{
  for (unsigned int icol = 0; icol < NCOLUMN; icol++) {
    for (unsigned int irow = 0; irow < NROW; irow++) {
      mInverseMapping[mMapping[icol][irow]] = std::make_tuple(icol, irow);
    }
  }
}

void PadMapper::ChannelIDException::print(std::ostream& stream) const
{
  stream << mMessage;
}

void PadMapper::PositionException::print(std::ostream& stream) const
{
  stream << mMessage;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PadMapper::ChannelIDException& except)
{
  except.print(stream);
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PadMapper::PositionException& except)
{
  except.print(stream);
  return stream;
}