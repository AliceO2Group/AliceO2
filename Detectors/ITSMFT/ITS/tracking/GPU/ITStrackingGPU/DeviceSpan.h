// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file DeviceSpan.h
/// \brief Device-side span and cluster lookup, without host runtime dependencies.
///

#ifndef ITSTRACKINGGPU_DEVICESPAN_H_
#define ITSTRACKINGGPU_DEVICESPAN_H_

#include "GPUCommonDef.h"
#include "ITStracking/Cluster.h"

namespace o2::its::gpu
{

template <typename T>
struct gpuSpan {
  using value_type = T;
  using ptr = T*;
  using ref = T&;

  GPUd() gpuSpan() : _data(nullptr), _size(0) {}
  GPUd() gpuSpan(ptr data, unsigned int dim) : _data(data), _size(dim) {}
  GPUd() ref operator[](unsigned int idx) const { return _data[idx]; }
  GPUd() unsigned int size() const { return _size; }
  GPUd() bool empty() const { return _size == 0; }
  GPUd() ref front() const { return _data[0]; }
  GPUd() ref back() const { return _data[_size - 1]; }
  GPUd() ptr begin() const { return _data; }
  GPUd() ptr end() const { return _data + _size; }

 protected:
  ptr _data;
  unsigned int _size;
};

template <typename T>
struct gpuSpan<const T> {
  using value_type = T;
  using ptr = const T*;
  using ref = const T&;

  GPUd() gpuSpan() : _data(nullptr), _size(0) {}
  GPUd() gpuSpan(ptr data, unsigned int dim) : _data(data), _size(dim) {}
  GPUd() gpuSpan(const gpuSpan<T>& other) : _data(other._data), _size(other._size) {}
  GPUd() ref operator[](unsigned int idx) const { return _data[idx]; }
  GPUd() unsigned int size() const { return _size; }
  GPUd() bool empty() const { return _size == 0; }
  GPUd() ref front() const { return _data[0]; }
  GPUd() ref back() const { return _data[_size - 1]; }
  GPUd() ptr begin() const { return _data; }
  GPUd() ptr end() const { return _data + _size; }

 protected:
  ptr _data;
  unsigned int _size;
};

#ifdef GPUCA_GPUCODE
GPUdii() gpuSpan<const Cluster> getClustersOnLayer(const int rof,
                                                   const int totROFs,
                                                   const int layer,
                                                   const int** roframesClus,
                                                   const Cluster** clusters)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const Cluster>();
  }
  const int start_clus_id{roframesClus[layer][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[layer][stop_rof] - start_clus_id;
  return gpuSpan<const Cluster>(&(clusters[layer][start_clus_id]), delta);
}
#endif

} // namespace o2::its::gpu

#endif // ITSTRACKINGGPU_DEVICESPAN_H_
