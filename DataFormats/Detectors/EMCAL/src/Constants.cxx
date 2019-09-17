// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iostream>
#include "DataFormatsEMCAL/Constants.h"

std::ostream& o2::emcal::operator<<(std::ostream& stream, o2::emcal::ChannelType_t chantype)
{
  stream << o2::emcal::channelTypeToString(chantype);
  return stream;
};

std::string o2::emcal::channelTypeToString(o2::emcal::ChannelType_t chantype)
{
  std::string typestring;
  switch (chantype) {
    case o2::emcal::ChannelType_t::HIGH_GAIN:
      typestring = "high gain";
      break;
    case o2::emcal::ChannelType_t::LOW_GAIN:
      typestring = "low gain";
      break;
    case o2::emcal::ChannelType_t::TRU:
      typestring = "tru";
      break;
    case o2::emcal::ChannelType_t::LEDMON:
      typestring = "LEDmon";
      break;
  };
  return typestring;
}

int o2::emcal::channelTypeToInt(o2::emcal::ChannelType_t chantype)
{
  switch (chantype) {
    case o2::emcal::ChannelType_t::HIGH_GAIN:
      return 1;
    case o2::emcal::ChannelType_t::LOW_GAIN:
      return 0;
    case o2::emcal::ChannelType_t::TRU:
      return 2;
    case o2::emcal::ChannelType_t::LEDMON:
      return 3;
  };
  throw o2::emcal::InvalidChanneltypeException(int(chantype));
}

o2::emcal::ChannelType_t o2::emcal::intToChannelType(int chantype)
{
  switch (chantype) {
    case 0:
      return o2::emcal::ChannelType_t::LOW_GAIN;
    case 1:
      return o2::emcal::ChannelType_t::HIGH_GAIN;
    case 2:
      return o2::emcal::ChannelType_t::TRU;
    case 3:
      return o2::emcal::ChannelType_t::LEDMON;
  };
  throw o2::emcal::InvalidChanneltypeException(chantype);
}
