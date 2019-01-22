#pragma once

#include "DataReader.h"
#include "DigitParser.h"

#include <args/args.hxx>


namespace gpucf
{

using DigitReader = DataReader<Digit, DigitParser>;

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
