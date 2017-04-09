//-*- Mode: C++ -*-

#ifndef COMPONENT_H
#define COMPONENT_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   Component.h
//  @author Matthias Richter
//  @since  2014-05-08
//  @brief  A component running ALICE HLT code

#include "AliHLTDataTypes.h"
#include "MessageFormat.h"
#include <vector>
#include <boost/signals2.hpp>
//using boost::signals2::signal;
typedef boost::signals2::signal<unsigned char* (unsigned int)> cballoc_signal_t;

namespace ALICE {
namespace HLT {
class SystemInterface;

/// @class Component
/// This class handles the creation of an HLT component and data processing
/// via the SystemInterface. Each HLT component is implemented in a library
/// and identified by a "component ID". HLT components support the processing
/// interface and need to be initialized by calling function Init().
///
/// Mandatory arguments:
/// --library       HLT component library to be loaded
/// --component     component ID
///
/// Optional arguments
/// --parameter     parameters of the component
///                 Each componenent is initialized from a list of arguments
///                 which have impact on e.g. algorithm, cuts, data selection
///                 and the type of output being produced. The arguments are
///                 completely under control of component.
/// --run           run no
///                 Currently, all HLT components need configuration and
///                 calibration data from OCDB; OCDB interface needs to
///                 be initialized with run number
/// --msgsize       size of the output buffer in byte
///                 This overrides the default behavior where output buffer
///                 size is determined from the input size and properties
///                 of the component
/// --output-mode   mode of arranging output blocks, @see MessageFormat.h
///                 0  HOMER format
///                 1  blocks in multiple messages
///                 2  blocks concatenated in one message (default)
///
class Component {
public:
  /// default constructor
  Component();
  /// destructor
  ~Component();

  /// Init the component
  int init(int argc, char** argv);

  /// Process one event
  /// Method takes a list of binary buffers which are expected to start with
  /// the AliHLTComponentBlockData header immediately followed by the block
  /// payload. After processing, handles to output blocks are provided in this
  /// list.
  int process(std::vector<o2::AliceHLT::MessageFormat::BufferDesc_t>& dataArray,
              cballoc_signal_t* cbAllocate=nullptr);

  int getEventCount() const {return mEventCount;}

protected:

private:
  // copy constructor prohibited
  Component(const Component&);
  // assignment operator prohibited
  Component& operator=(const Component&);

  /// output buffer to receive the data produced by component
  std::vector<AliHLTUInt8_t> mOutputBuffer;

  /// instance of the system interface
  SystemInterface* mpSystem;
  /// handle of the processing component
  AliHLTComponentHandle mProcessor;
  /// container for handling the i/o buffers
  o2::AliceHLT::MessageFormat mFormatHandler;
  int mEventCount;
};

} // namespace hlt
} // namespace alice
#endif // COMPONENT_H
