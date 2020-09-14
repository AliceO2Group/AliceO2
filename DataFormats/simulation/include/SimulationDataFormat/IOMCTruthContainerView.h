// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IOMCTruthContainerView.h
/// \brief A special IO container - splitting a given vector to enable ROOT IO
/// \author Sandro Wenzel - August 2020

#ifndef ALICEO2_DATAFORMATS_IOMCTRUTHVIEW_H_
#define ALICEO2_DATAFORMATS_IOMCTRUTHVIEW_H_

#include "GPUCommonRtypes.h" // to have the ClassDef macros
#include <vector>
#include <gsl/span>

namespace o2
{
namespace dataformats
{

///
/// A specially constructed class allowing to stream a very large
/// vector buffer to a ROOT file. This is needed since ROOT currently has a size
/// limitation of ~1GB for data that it can stream per entry in a branch.
/// The solution is based on the ability of ROOT to split entries per data member, so
/// some input buffer gets divided into multiple parts.
///
/// TODO: We could template this class to encode original type information (for the input buffer).
class IOMCTruthContainerView
{
 public:
  IOMCTruthContainerView() = default;

  /// Constructor taking an existing flat vector as input; No copy is done - the
  /// container is just a split view on the original buffer.
  IOMCTruthContainerView(std::vector<char> const& input)
  {
    adopt(input);
  }

  /// "adopt" (without taking ownership) from an existing buffer
  void adopt(std::vector<char> const& input)
  {
    const auto delta = input.size() / N;
    N2 = input.size() - (N - 1) * delta;
    N1 = delta;
    // TODO: this could benefit from a loop expansion
    part1 = &input[0];
    part2 = &input[delta];
    part3 = &input[2 * delta];
    part4 = &input[3 * delta];
    part5 = &input[4 * delta];
    part6 = &input[5 * delta];
    part7 = &input[6 * delta];
    part8 = &input[7 * delta];
    part9 = &input[8 * delta];
    part10 = &input[9 * delta];
  }

  /// A function to recreate a flat output vector from this buffer. This
  /// function is copying the data.
  template <typename Alloc>
  void copyandflatten(std::vector<char, Alloc>& output) const
  {
    // TODO: this could benefit from a loop expansion
    copyhelper(part1, N1, output);
    copyhelper(part2, N1, output);
    copyhelper(part3, N1, output);
    copyhelper(part4, N1, output);
    copyhelper(part5, N1, output);
    copyhelper(part6, N1, output);
    copyhelper(part7, N1, output);
    copyhelper(part8, N1, output);
    copyhelper(part9, N1, output);
    copyhelper(part10, N2, output);
  }

  /// return total size in bytes
  size_t getSize() const { return N1 * (N - 1) + N2; }

 private:
  static constexpr int N = 10;
  int N1 = 0;
  int N2 = 0;
  const char* part1 = nullptr;  //[N1]
  const char* part2 = nullptr;  //[N1]
  const char* part3 = nullptr;  //[N1]
  const char* part4 = nullptr;  //[N1]
  const char* part5 = nullptr;  //[N1]
  const char* part6 = nullptr;  //[N1]
  const char* part7 = nullptr;  //[N1]
  const char* part8 = nullptr;  //[N1]
  const char* part9 = nullptr;  //[N1]
  const char* part10 = nullptr; //[N2]

  template <typename Alloc>
  void copyhelper(const char* input, int size, std::vector<char, Alloc>& output) const
  {
    gsl::span<const char> tmp(input, size);
    std::copy(tmp.begin(), tmp.end(), std::back_inserter(output));
  }

  ClassDefNV(IOMCTruthContainerView, 1);
};
} // namespace dataformats
} // namespace o2

#endif
