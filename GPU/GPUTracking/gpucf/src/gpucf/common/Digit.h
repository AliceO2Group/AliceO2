#pragma once

#include <shared/Digit.h>

#include <iosfwd>
#include <string>


std::ostream &operator<<(std::ostream &, const Digit &);

std::string serialize(const Digit &);

// vim: set ts=4 sw=4 sts=4 expandtab:
