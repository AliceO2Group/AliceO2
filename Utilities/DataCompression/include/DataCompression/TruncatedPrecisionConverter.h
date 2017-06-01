// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-

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
//  @since  2015-08-08
//  @brief  A simple converter producing truncated precision
//          according to a data model 

namespace AliceO2 {

template<class _ParameterModel>
class TruncatedPrecisionConverter {
public:
  TruncatedPrecisionConverter() : mParameterModel() {}
  ~TruncatedPrecisionConverter() {}

  template <typename T, typename _RegType, typename _Writer>
  int Write(T value, _RegType /*dummy*/, _Writer writer) {
    uint8_t bitlength=0;
    _RegType content=0;
    mParameterModel.Convert(value, content, bitlength);
    return writer(content, bitlength);
  }

  void ResetParameterModel() {
    mParameterModel.Reset();
  }

  const _ParameterModel& GetModel() const {return mParameterModel;}
  _ParameterModel& GetModel() {return mParameterModel;}

private:
  /// forbidden in the first implementation
  TruncatedPrecisionConverter(const TruncatedPrecisionConverter&);
  /// forbidden in the first implementation
  TruncatedPrecisionConverter& operator=(const TruncatedPrecisionConverter&);
  /// parameter model defines the conversion to the register type for writing bit pattern
  _ParameterModel mParameterModel;
};
};
#endif
