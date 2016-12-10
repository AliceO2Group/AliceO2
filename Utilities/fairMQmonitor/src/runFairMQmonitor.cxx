/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#include "runFairMQDevice.h"
#include "fairMQmonitor/FairMQmonitor.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("n",bpo::value<int>()->default_value(-1), "How many loops");
  options.add_options()
    ("sleep",bpo::value<int>()->default_value(0), "sleep between loops in milliseconds");
  options.add_options()
    ("limit",bpo::value<int>()->default_value(0), "limit output of payload to n characters");
  options.add_options()
    ("payload",bpo::value<std::string>()->
     default_value("I am the info payload"), "the info string in the payload");
  options.add_options()
    ("name",bpo::value<std::string>()->default_value(""), "optional name in the header");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
    return new FairMQmonitor();
}
