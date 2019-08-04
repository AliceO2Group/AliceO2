// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Framework/DPLBoostSerializer.h
/// \brief  DPL wrapper of common utils BoostSeralizer
/// \author Gabriele G. Fronzé <gfronze at cern.ch>
/// \date   17 July 2018

#ifndef ALICEO2_DPLBOOSTSERIALIZER_H
#define ALICEO2_DPLBOOSTSERIALIZER_H

#include "Framework/DataRef.h"
#include "Framework/Output.h"
#include "CommonUtils/BoostSerializer.h"
#include "ProcessingContext.h"

namespace o2
{
namespace framework
{

template <typename ContT>
void DPLBoostSerialize(o2::framework::ProcessingContext& ctx, const std::string& outBinding,
                       const ContT& dataSet)
{
  /// Serialises a data set (in the form of vector, array or list) in a message
  auto buffer = o2::utils::SerializeContainer<ContT>(dataSet);
  int size = buffer.str().length();
  auto msg = ctx.outputs().make<char>({outBinding}, size);
  std::memcpy(&(msg[0]), buffer.str().c_str(), size);
}

template <typename ContT>
void DPLBoostSerialize(o2::framework::ProcessingContext& ctx, const std::string& outBinding,
                       const ContT& dataSet, const unsigned long& nData)
{
  /// Serialises the nData elements of a data set (in the form of vector, array or list) in a message
  ContT subSet(std::begin(dataSet), std::next(std::begin(dataSet), nData));
  DPLBoostSerialize(ctx, outBinding, subSet);
}

template <typename ContT>
void DPLBoostDeserialize(const o2::framework::DataRef& msg, ContT& output)
{
  /// Deserialises a DPL msg to a container type (vector, array or list) of the provided type.
  using dataH = o2::header::DataHeader;
  auto payloadSize = const_cast<dataH*>(reinterpret_cast<const dataH*>(msg.header))->payloadSize;
  output.clear();
  std::string msgStr(msg.payload, payloadSize);
  output = std::move(o2::utils::DeserializeContainer<ContT>(msgStr));
}
} // namespace framework
} // namespace o2

#endif //ALICEO2_DPLBOOSTSERIALIZER_H
