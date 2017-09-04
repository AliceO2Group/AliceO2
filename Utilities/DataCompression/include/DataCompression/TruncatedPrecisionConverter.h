// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TRUNCATEDPRECISIONCONVERTER_H
#define TRUNCATEDPRECISIONCONVERTER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   TruncatedPrecisionConverter.h
//  @author Matthias Richter
//  @since  2016-08-08
//  @brief  A simple converter producing truncated precision
//          according to a parameter model

namespace o2 {
namespace data_compression {

/**
 * @TruncatedPrecisionConverter A simple converter producing truncated precision
 * The converter implements the write function needed to be used as a codec
 * in the data compression framework. Simply a prototype case for the moment.
 *
 * The parameter model is required to implement the method 'convert'.
 */
template<class ParameterModelT>
class TruncatedPrecisionConverter {
public:
  TruncatedPrecisionConverter() : mParameterModel() {}
  ~TruncatedPrecisionConverter() = default;
  TruncatedPrecisionConverter(const TruncatedPrecisionConverter&) = delete;
  TruncatedPrecisionConverter& operator=(const TruncatedPrecisionConverter&) = delete;

  static const std::size_t sMaxLength = ParameterModelT::sBitlength;
  using code_type = typename ParameterModelT::converted_type;

  template <typename T, typename Writer>
  int write(T value, Writer writer) {
    uint8_t bitlength=0;
    code_type content=0;
    mParameterModel.convert(value, content, bitlength);
    return writer(content, bitlength);
  }

  void resetParameterModel() {
    mParameterModel.reset();
  }

  const ParameterModelT& getModel() const {return mParameterModel;}
  ParameterModelT& getModel() {return mParameterModel;}

private:
  /// parameter model defines the conversion to the register type for writing bit pattern
  ParameterModelT mParameterModel;
};
}; // namespace data_compression
}; // namespace o2
#endif
