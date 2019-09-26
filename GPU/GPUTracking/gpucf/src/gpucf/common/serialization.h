// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/SectorMap.h>

#include <filesystem/path.h>

#include <nonstd/span.h>

#include <cstdint>
#include <fstream>
#include <vector>

namespace gpucf
{

template <typename T>
SectorMap<std::vector<T>> read(filesystem::path f)
{
  std::ifstream in(f.str(), std::ios::binary);

  SectorMap<uint64_t> header;
  in.read(reinterpret_cast<char*>(header.data()),
          sizeof(uint64_t) * header.size());

  SectorMap<std::vector<T>> sectordata;
  for (size_t sector = 0; sector < header.size(); sector++) {
    size_t n = header[sector];
    std::vector<T>& data = sectordata[sector];

    data.resize(n);
    in.read(reinterpret_cast<char*>(data.data()),
            data.size() * sizeof(T));
  }

  return sectordata;
}

template <typename R>
void write(filesystem::path f, nonstd::span<const R> data)
{
  std::ofstream out(f.str(), std::ios::binary);

  uint64_t size = data.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));

  out.write(reinterpret_cast<const char*>(data.data()), size * sizeof(R));
}

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
