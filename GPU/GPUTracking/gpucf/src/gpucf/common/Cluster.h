#pragma once

#include <shared/Cluster.h>

#include <iosfwd>


std::ostream &operator<<(std::ostream &, const Cluster &);

bool operator==(const Cluster &, const Cluster &);

// vim: set ts=4 sw=4 sts=4 expandtab:
