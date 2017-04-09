/// @file   TimeStamp.cxx
/// @author Matthias Richter
/// @since  2017-01-25
/// @brief  A std chrono implementation of LHC clock and timestamp

#include "Headers/TimeStamp.h"

using namespace o2::Header;

// the only reason for the cxx file is the implementation of the
// constants
TimeStamp::TimeUnitID const TimeStamp::sClockLHC("AC");
TimeStamp::TimeUnitID const TimeStamp::sMicroSeconds("US");
