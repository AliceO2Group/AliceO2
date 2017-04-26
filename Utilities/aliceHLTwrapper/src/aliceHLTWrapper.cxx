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

//  @file   aliHLTWrapper.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include "aliceHLTwrapper/WrapperDevice.h"
#include <iostream>
#include <getopt.h>
#include <memory>
#include <cstring>
#include <sstream>
#include <cerrno>
#include <tools/FairMQTools.h>

#include <FairMQStateMachine.h>
#if defined(FAIRMQ_INTERFACE_VERSION) && FAIRMQ_INTERFACE_VERSION > 0
// FairMQStateMachine interface supports strings as argument for the
// ChangeState function from interface version 1 introduced Feb 2015
#define HAVE_FAIRMQ_INTERFACE_CHANGESTATE_STRING
#endif

// DDS intercom API has been changed in release 1.4
// as it is not a matter of a few lines, DDS support is disabled for
// the moment. We need a major rewrite and testing 
#ifdef ENABLE_DDS
#define ENFORCED_DDS_DISABLE
#undef ENABLE_DDS
#endif

#ifdef ENABLE_DDS
#include <mutex>
#include <condition_variable>
#include "dds_intercom.h" // DDS
#include <boost/asio.hpp>  // boost::lock
#endif

#include <chrono>
#include <boost/chrono.hpp>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::stringstream;
using std::ostream;
using std::vector;

struct SocketProperties_t {
  std::string type;
  int         size;
  std::string method;
  std::string address;
  std::string ddsprop;
  int         ddscount;
  int         ddsminport;
  int         ddsmaxport;
  unsigned    validParams;   // indicates which parameter has been specified

  SocketProperties_t()
    : type()
    , size(0)
    , method()
    , address()
    , ddsprop()
    , ddscount(0)
    , ddsminport(0)
    , ddsmaxport(0)
    , validParams(0)
  {}
  SocketProperties_t(const SocketProperties_t& other)
    = default;
};

ostream& operator<<(ostream &out, const SocketProperties_t& me)
{
  out << "Socket configuration:"
      << " type       = " << me.type
      << " size       = " << me.size
      << " method     = " << me.method
      << " address    = " << me.address
      << " ddsprop    = " << me.ddsprop
      << " ddscount   = " << me.ddscount
      << " ddsminport = " << me.ddsminport
      << " ddsmaxport = " << me.ddsmaxport
    ;
  return out;
}

FairMQDevice* gDevice = nullptr;

int preprocessSockets(vector<SocketProperties_t>& sockets);
int preprocessSocketsDDS(vector<SocketProperties_t>& sockets, std::string networkPrefix="");
std::string buildSocketParameterErrorMsg(unsigned reqMask, unsigned validMask, const char* headerMsg="");
int sendSocketPropertiesDDS(vector<SocketProperties_t>& sockets);
int readSocketPropertiesDDS(vector<SocketProperties_t>& sockets);

  enum socketkeyids{
    TYPE = 0,       // push, pull, publish, subscribe, etc
    SIZE,           // queue size
    METHOD,         // bind or connect
    ADDRESS,        // host, protocol and port address
    PROPERTY,       // DDS property name for connecting sockets
    COUNT,          // DDS property count
    MINPORT,        // DDS port range minimum
    MAXPORT,        // DDS port range maximum
    DDSGLOBAL,      // DDS global property
    DDSLOCAL,       // DDS local property
    lastsocketkey
  };

  const char *socketkeys[] = {
    /*[TYPE]      = */ "type",
    /*[SIZE]      = */ "size",
    /*[METHOD]    = */ "method",
    /*[ADDRESS]   = */ "address",
    /*[PROPERTY]  = */ "property",
    /*[COUNT]     = */ "count",
    /*[MINPORT]   = */ "min-port",
    /*[MAXPORT]   = */ "max-port",
    /*[DDSGLOBAL] = */ "global",
    /*[DDSLOCAL]  = */ "local",
    nullptr
  };


int main(int argc, char** argv)
{
  int iResult = 0;

  // parse options
  int iArg = 0;
  int iDeviceArg = -1;
  bool bPrintUsage = false;
  std::string id;
  int numIoThreads = 0;
  int numInputs = 0;
  int numOutputs = 0;

  vector<SocketProperties_t> inputSockets;
  vector<SocketProperties_t> outputSockets;
  const char* factoryType = "zmq";
  int verbosity = -1;
  int deviceLogInterval = 10000;
  int pollingPeriod = -1;
  int skipProcessing = 0;
  bool bUseDDS = false;
  int timeout=-1;
  bool bInteractive = false;

  static struct option programOptions[] = {
    { "input",       required_argument, nullptr, 'i' }, // input socket
    { "output",      required_argument, nullptr, 'o' }, // output socket
    { "factory-type",required_argument, nullptr, 'f' }, // type of the factory "zmq", "nanomsg"
    { "verbosity",   required_argument, nullptr, 'v' }, // verbosity
    { "loginterval", required_argument, nullptr, 'l' }, // logging interval
    { "poll-period", required_argument, nullptr, 'p' }, // polling period of the device in ms
    { "dry-run",     no_argument      , nullptr, 'n' }, // skip the component processing
    { "dds",         no_argument      , nullptr, 'd' }, // run in dds mode
    { "timeout",     required_argument, nullptr, 't' }, // polling period of the device in ms
    { "interactive", no_argument      , nullptr, 'x' }, // enter interactive mode (from terminal only)
    { nullptr, 0, nullptr, 0 }
  };

  char c = 0;
  int iOption = 0;
  opterr = false;
  optind = 1; // indicate new start of scanning

  // build the string from the option definition
  // hyphen in the beginning indicates custom handling for non-option elements
  // a colon after the option indicates a required argument to that option
  // two colons after the option indicate an optional argument
  std::string optstring = "-";
  for (struct option* programOption = programOptions;
       programOption != nullptr && programOption->name != nullptr;
       programOption++) {
    if (programOption->flag == nullptr) {
      // programOption->val uniquely identifies particular long option
      optstring += ((char)programOption->val);
      if (programOption->has_arg == required_argument) optstring += ":";  // one colon to indicate required argument
      if (programOption->has_arg == optional_argument) optstring += "::"; // two colons to indicate optional argument
    } else {
      throw std::runtime_error(
        "handling of program option flag is not yet implemented, please check option definitions");
    }
  }
  while ((c = getopt_long(argc, argv, optstring.c_str(), programOptions, &iOption)) != -1
         && bPrintUsage == false
         && iDeviceArg < 0) {
    switch (c) {
      case 'i':
      case 'o': {
        char* subopts = optarg;
        char* value = nullptr;
        int keynum = 0;
        SocketProperties_t prop;
        while (subopts && *subopts != 0 && *subopts != ' ') {
          char* saved = subopts;
          int subopt=getsubopt(&subopts, (char**)socketkeys, &value);
          if (subopt>=0) prop.validParams|=0x1<<subopt;
          switch (subopt) {
            case TYPE:     prop.type=value;                       break;
            case SIZE:     std::stringstream(value) >> prop.size; break;
            case METHOD:   prop.method=value;                     break;
            case ADDRESS:  prop.address=value;                    break;
            case PROPERTY: prop.ddsprop=value;                    break;
            case COUNT:    std::stringstream(value) >> prop.ddscount;   break;
            case MINPORT:  std::stringstream(value) >> prop.ddsminport; break;
            case MAXPORT:  std::stringstream(value) >> prop.ddsmaxport; break;
            case DDSGLOBAL:
              // fall-through intentional 
            case DDSLOCAL:
              // key without argument, can be checked from the validParams bitfield
              break;
            default:
              prop.validParams = 0;
              break;
          }
        }
        {
          if (c == 'i')
            inputSockets.push_back(prop);
          else
            outputSockets.push_back(prop);
        }
      } break;
      case 'f':
        factoryType = optarg;
        break;
      case 'v':
        std::stringstream(optarg) >> std::hex >> verbosity;
        break;
      case 'l':
        std::stringstream(optarg) >> deviceLogInterval;
        break;
      case 'p':
        std::stringstream(optarg) >> pollingPeriod;
        break;
      case 't':
        std::stringstream(optarg) >> timeout;
        break;
      case 'n':
        skipProcessing = 1;
        break;
      case 'x':
        bInteractive = true;
        break;
      case 'd':
        bUseDDS = true;
        break;
      case '?': // all remaining arguments passed to the device instance
        iDeviceArg = optind - 1;
        break;
      case '\1': // the first required arguments are without hyphens and in fixed order
        // special treatment of elements not defined in the option string is
        // indicated by the leading hyphen in the option string, that makes
        // getopt to return value '1' for any of those elements allowing
        // for a custom handling, matches only elemments not starting with a hyphen
        switch (++iArg) {
          case 1: id=optarg; break;
          case 2: std::stringstream(optarg) >> numIoThreads; break;
          default:
            bPrintUsage = true;
        }
        break;
      default:
        cerr << "unknown option: '" << c << "'" << endl;
    }
  }

  std::string networkPrefix;
  std::map<string,string> IPs;
  FairMQ::tools::getHostIPs(IPs);

  if(IPs.count("ib0")) {
    networkPrefix+=IPs["ib0"];
  } else {
    networkPrefix+=IPs["eth0"];
  }

  if (bUseDDS) {
#ifndef ENABLE_DDS
#ifdef ENFORCED_DDS_DISABLE
    cerr << "Fatal: device is not compatible with DDS Intercom API any more" << endl;
#else
    cerr << "Fatal: device has not been compiled with DDS support" << endl;
#endif
    exit(ENOSYS);
#endif
    int result=preprocessSocketsDDS(inputSockets, networkPrefix);
    if (result>=0)
      result=preprocessSocketsDDS(outputSockets, networkPrefix);
    bPrintUsage=result<0;
  } else {
    int result=preprocessSockets(inputSockets);
    if (result>=0)
      result=preprocessSockets(outputSockets);
    bPrintUsage=result<0;    
  }

  numInputs = inputSockets.size();
  numOutputs = outputSockets.size();
  if (bPrintUsage || iDeviceArg < 0 || (numInputs == 0 && numOutputs == 0)) {
    cout << endl << argv[0] << ":" << endl;
    cout << "        wrapper to run an ALICE HLT component in a FairRoot/ALFA device" << endl;
    cout << "Usage : " << argv[0] << " ID numIoThreads [--factory type] [--input|--output "
                                     "type=value,size=value,method=value,address=value] componentArguments" << endl;
    cout << "        The first two arguments are in fixed order, followed by optional arguments: " << endl;
    cout << "        --factory-type,-t nanomsg|zmq" << endl;
    cout << "        --poll-period,-p             period_in_ms" << endl;
    cout << "        --loginterval,-l             period_in_ms" << endl;
    cout << "        --verbosity,-v 0xhexval      verbosity level" << endl;
    cout << "        --dry-run,-n                 skip the component processing" << endl;
    cout << "        Multiple slots can be defined by --input/--output options" << endl;
    cout << "        HLT component arguments at the end of the list" << endl;
    cout << "        --library,-l     componentLibrary" << endl;
    cout << "        --component,-c   componentId"	<< endl;
    cout << "        --parameter,-p   parameter"	<< endl;
    cout << "        --run,-r         runNo"            << endl;

    return 0;
  }

  std::string transport;
  if (strcmp(factoryType, "nanomsg") == 0) {
    transport = "nanomsg";
  } else if (strcmp(factoryType, "zmq") == 0) {
    transport = "zeromq";
  } else {
    cerr << "invalid factory type: " << factoryType << endl;
    return -ENODEV;
  }

  if (verbosity >= 0) {
    std::ios::fmtflags oldflags = std::cout.flags();
    std::cout << "Verbosity: 0x" << std::setfill('0') << std::setw(2) << std::hex << verbosity << std::endl;
    std::cout.flags(oldflags);
    // verbosity option is propagated to device, the verbosity level
    // for the HLT component code can be specified as extra parameter,
    // note that this is using a mask, where each bit corresponds to
    // a message catagory, e.g. 0x78 is commonly used to get all
    // above "Info"
  }

  vector<char*> deviceArgs;
  deviceArgs.push_back(argv[0]);
  if (iDeviceArg > 0) deviceArgs.insert(deviceArgs.end(), argv + iDeviceArg, argv + argc);

  gDevice = new ALICE::HLT::WrapperDevice(deviceArgs.size(), &deviceArgs[0], verbosity);
  if (!gDevice) {
    cerr << "failed to create device" << endl;
    return -ENODEV;
  }
  gDevice->CatchSignals();

  { // scope for the device reference variable
    FairMQDevice& device = *gDevice;

    device.SetTransport(transport);
    device.SetProperty(FairMQDevice::Id, id.c_str());
    device.SetProperty(FairMQDevice::NumIoThreads, numIoThreads);
    // device.SetProperty(FairMQDevice::LogIntervalInMs, deviceLogInterval);
    if (pollingPeriod > 0) device.SetProperty(ALICE::HLT::WrapperDevice::PollingPeriod, pollingPeriod);
    if (skipProcessing) device.SetProperty(ALICE::HLT::WrapperDevice::SkipProcessing, skipProcessing);
    for (unsigned iInput = 0; iInput < numInputs; iInput++) {
      std::cout << "input socket " << iInput << " " << inputSockets[iInput] << endl;
      // if running in DDS mode, the address contains now the IP address of the host
      // machine in order to check if a DDS property comes from the same machine.
      // This will be obsolete as soon as DDS propagates properties only within
      // collections. Then the address can be empty for connecting sockets and
      // can be directly used in the creation of the channel. The next two lines are
      // then obsolete
      std::string address=inputSockets[iInput].address.c_str();
      if (inputSockets[iInput].method.compare("connect")==0 && bUseDDS) address="";
      FairMQChannel inputChannel(inputSockets[iInput].type.c_str(), inputSockets[iInput].method.c_str(), address);
      // set High-water-mark for the sockets. in ZMQ, depending on the socket type, some
      // have only send buffers (PUB, PUSH), some only receive buffers (SUB, PULL), and
      // some have both (DEALER, ROUTER, PAIR, REQ, REP)
      // we set both snd and rcv to the same value for the moment
      inputChannel.UpdateSndBufSize(inputSockets[iInput].size);
      inputChannel.UpdateRcvBufSize(inputSockets[iInput].size);
      device.fChannels["data-in"].push_back(inputChannel);
    }
    for (unsigned iOutput = 0; iOutput < numOutputs; iOutput++) {
      std::cout << "output socket " << iOutput << " " << outputSockets[iOutput] << endl;
      // see comment above, next two lines obsolete if DDS implements property
      // propagation within collections
      std::string address=outputSockets[iOutput].address.c_str();
      if (outputSockets[iOutput].method.compare("connect")==0 && bUseDDS) address="";
      FairMQChannel outputChannel(outputSockets[iOutput].type.c_str(), outputSockets[iOutput].method.c_str(), address);
      // we set both snd and rcv to the same value for the moment, see above
      outputChannel.UpdateSndBufSize(outputSockets[iOutput].size);
      outputChannel.UpdateRcvBufSize(outputSockets[iOutput].size);
      device.fChannels["data-out"].push_back(outputChannel);
    }

    // The initialization state runs in its own thread.
    // It initializes sockets that have valid parameters, and then tries every 
    // second to initialize the invalid ones, until all are initialized or timeout is reached
    // (timeout can be set with SetProperty(FairMQDevice::MaxInitializationTime, valueInSeconds), default 120 seconds).
    // This can be used to wait for DDS values.
    device.ChangeState("INIT_DEVICE");
#if defined(HAVE_FAIRMQ_INTERFACE_CHANGESTATE_STRING)
    // Feb 2015: changes in the FairMQStateMachine interface
    // two new state changes introduced. To make the compilation
    // independent of this in future changes, the ChangeState
    // method has been introduced with string argument
    // TODO: change later to this function

    device.WaitForInitialValidation(); // this waits until valid sockets are configured (e.g. those that Bind())

    // port addresses are assigned after BIND and can be propagated using DDS
    if (bUseDDS) {
      for (unsigned iInput = 0; iInput < numInputs; iInput++) {
       if (inputSockets[iInput].method.compare("bind")==1) continue;
        inputSockets[iInput].address=device.fChannels["data-in"].at(iInput).GetAddress();
      }
      for (unsigned iOutput = 0; iOutput < numOutputs; iOutput++) {
        if (outputSockets[iOutput].method.compare("bind")==1) continue;
        outputSockets[iOutput].address=device.fChannels["data-out"].at(iOutput).GetAddress();
      }
      sendSocketPropertiesDDS(inputSockets);
      sendSocketPropertiesDDS(outputSockets);

      readSocketPropertiesDDS(inputSockets);
      readSocketPropertiesDDS(outputSockets);
      for (unsigned iInput = 0; iInput < numInputs; iInput++) {
        if (inputSockets[iInput].method.compare("connect")==1) continue;
        device.fChannels["data-in"].at(iInput).UpdateAddress(inputSockets[iInput].address.c_str());
      }
      for (unsigned iOutput = 0; iOutput < numOutputs; iOutput++) {
        if (outputSockets[iOutput].method.compare("connect")==1) continue;
        device.fChannels["data-out"].at(iOutput).UpdateAddress(outputSockets[iOutput].address.c_str());
      }
    }
    cout << "finnished fetching properties from DDS, now waiting for end of state INIT_DEVICE ..." << endl;
    device.WaitForEndOfState("INIT_DEVICE");
#endif
    device.ChangeState("INIT_TASK");
    device.WaitForEndOfState("INIT_TASK");

    device.ChangeState("RUN");

    if (bInteractive) {
      device.InteractiveStateLoop();
    }
    else { // identation not changed in the following to avoid fake diffs

    auto refTime = boost::chrono::system_clock::now();

    while (device.GetCurrentStateName() == "RUNNING") {
      device.WaitForEndOfStateForMs("RUN", 1000); // Wait end of running state for 1000 ms.
      auto duration = boost::chrono::duration_cast<boost::chrono::seconds>(boost::chrono::system_clock::now() - refTime);
      if (timeout>0) {
        cout << "seconds elapsed since start " << duration.count() << endl;
        if (duration.count()>timeout) {
          cout << "Executing time-out shutdown" << endl;
          // the execution is hanging in the state machine shutdown, has to be debugged
          // simply throw an exeption to terminate in order to allow callgrind to write
          // the statistics
          throw std::runtime_error("terminating by exception");
          gDevice->ChangeState(FairMQDevice::END);
          sleep(5);
          cout << "... done" << endl;
          refTime=boost::chrono::system_clock::now();
        }
      }
    }
  }
    device.ChangeState("RESET_TASK");
    device.WaitForEndOfState("RESET_TASK");

    device.ChangeState("RESET_DEVICE");
    device.WaitForEndOfState("RESET_DEVICE");

    device.ChangeState("END");
  } // scope for the device reference variable

  FairMQDevice* almostdead = gDevice;
  gDevice = nullptr;
  delete almostdead;

  return iResult;
}

int preprocessSockets(vector<SocketProperties_t>& sockets)
{
  // check consistency of socket parameters
  int iResult=0;
  for (vector<SocketProperties_t>::iterator sit=sockets.begin();
       sit!=sockets.end(); sit++) {
    unsigned maskRequiredParams=(0x1<<SIZE)|(0x1<<TYPE)|(0x1<<METHOD)|(0x1<<ADDRESS);
    if ((sit->validParams&maskRequiredParams)!=maskRequiredParams) {
      cerr << buildSocketParameterErrorMsg(maskRequiredParams, sit->validParams, "Error: missing socket parameter(s)") << endl;
      iResult=-1;
      break;
    }
  }
  return iResult; 
}

int preprocessSocketsDDS(vector<SocketProperties_t>& sockets, std::string networkPrefix)
{
  // binding sockets
  // - required arguments: size, type, property name, min, max port
  // - find available port within given range
  // - create address
  // - send property to DDS
  //
  // connecting sockets
  // - required arguments: size, property name, count
  // - replicate according to count
  // - fetch property from DDS
  // - if local, only treat properties from tasks on the same node
  // - translate type from type of opposite socket
  // - add address
  int iResult=0;
  vector<SocketProperties_t> ddsduplicates;
  for (vector<SocketProperties_t>::iterator sit=sockets.begin();
       sit!=sockets.end(); sit++) {
    if (sit->method.compare("bind")==0) {
      unsigned maskRequiredParams=(0x1<<SIZE)|(0x1<<TYPE)|(0x1<<PROPERTY)|(0x1<<MINPORT);
      if ((sit->validParams&maskRequiredParams)!=maskRequiredParams) {
	cerr << buildSocketParameterErrorMsg(maskRequiredParams, sit->validParams, "Error: missing parameter(s) for binding socket") << endl;
	iResult=-1;
	break;
      }
      // the port will be selected by the FairMQ framework during the
      // bind process, the address is a placeholder at the moment
      sit->address="tcp://"+networkPrefix+":";
    } else if (sit->method.compare("connect")==0) {
      unsigned maskRequiredParams=(0x1<<SIZE)|(0x1<<PROPERTY)|(0x1<<COUNT);
      if ((sit->validParams&maskRequiredParams)!=maskRequiredParams) {
	cerr << buildSocketParameterErrorMsg(maskRequiredParams, sit->validParams, "Error: missing parameter(s) for connecting socket") << endl;
	iResult=-1;
	break;
      }
      // the port adress will be read from DDS properties
      // address initialized to the host ip/name which is then used to match
      // properties on the same hostwhen using local mode
      // the actual number of input ports is spefified by the count argument, the
      // corresponding number of properties is expected from DDS
      sit->address=networkPrefix;
      // add n-1 duplicates of the port configuration
      for (int i=0; i<sit->ddscount-1; i++) {
	ddsduplicates.push_back(*sit);
	ddsduplicates.back().ddscount=0;
      }
    } else {
      cerr << "Error: invalid socket method '" << sit->method << "'" << endl;
      iResult=-1; // TODO: find error codes
      break;
    }
  }

  if (iResult>=0) {
    sockets.insert(sockets.end(), ddsduplicates.begin(), ddsduplicates.end());
  }
  return iResult;
}

int sendSocketPropertiesDDS(vector<SocketProperties_t>& sockets)
{
  // send dds property for all binding sockets
  for (vector<SocketProperties_t>::iterator sit=sockets.begin();
       sit!=sockets.end(); sit++) {
    if (sit->method.compare("bind")==1) continue;
    std::stringstream ddsmsg;
    // TODO: send the complete socket configuration to allow the counterpart to
    // set the relevant options
    //ddsmsg << socketkeys[TYPE]    << "=" << sit->type << ",";
    //ddsmsg << socketkeys[METHOD]  << "=" << sit->method << ",";
    //ddsmsg << socketkeys[SIZE]    << "=" << sit->size << ",";
    //ddsmsg << socketkeys[ADDRESS] << "=" << sit->address;
    ddsmsg << sit->address;

#ifdef ENABLE_DDS
    dds::intercom_api::CKeyValue ddsKeyValue;
    ddsKeyValue.putValue(sit->ddsprop, ddsmsg.str());
#endif

    cout << "DDS putValue: " << sit->ddsprop.c_str() << " " << ddsmsg.str() << endl;
  }
  return 0;
}

int readSocketPropertiesDDS(vector<SocketProperties_t>& sockets)
{
  // read dds properties for connecting sockets
  for (vector<SocketProperties_t>::iterator sit=sockets.begin();
       sit!=sockets.end(); sit++) {
    if (sit->method.compare("connect")==1) continue;
    if (sit->ddscount==0) continue; // the previously inserted duplicates

#ifdef ENABLE_DDS
    dds::intercom_api::CKeyValue ddsKeyValue;
    dds::intercom_api::CKeyValue::valuesMap_t values;

    std::string hostaddress=sit->address;
    vector<SocketProperties_t>::iterator workit=sit;
    int socketPropertiesToRead=sit->ddscount;
    std::map<std::string, bool> usedProperties;
    {
      std::mutex keyMutex;
      std::condition_variable keyCondition;
  
      ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

      for (int cycle=1; socketPropertiesToRead>0; cycle++) {
	if (cycle>0) {
	  std::unique_lock<std::mutex> lock(keyMutex);
	  keyCondition.wait_until(lock, std::chrono::system_clock::now() + chrono::milliseconds(1000));
	}
        ddsKeyValue.getValues(sit->ddsprop.c_str(), &values);
	cout << "Info: DDS getValues received " << values.size() << " value(s) of property " << sit->ddsprop
	     << " " << sit->ddscount-socketPropertiesToRead << " of " << sit->ddscount << " sockets processed" << endl;
	for (dds::intercom_api::CKeyValue::valuesMap_t::const_iterator vit = values.begin();
	     vit!=values.end(); vit++) {
	  if (usedProperties.find(vit->first)!=usedProperties.end()) continue; // already processed
	  cout << "Info: processing property " << vit->first << ", value " << vit->second << " on host " << hostaddress << endl;
	  usedProperties[vit->first]=true;
	  if (!(sit->validParams&(0x1<<DDSGLOBAL))) {
	    std::string property=vit->second;
	    if (property.find(hostaddress)==std::string::npos) {
	      cout << "Info: local parameter requested, skipping " << vit->second << " on host " << hostaddress << endl;
	      continue;
	    }
	  }
	  if (workit==sockets.end()) break;
	  workit->address=vit->second;
	  cout << "DDS getValue:" << sit->ddsprop << " " << sit->ddscount-socketPropertiesToRead << ": " << workit->address << endl;
	  socketPropertiesToRead--;
	  while ((++workit)!=sockets.end()) {
	    if (workit->method.compare("connect")==1) continue;
	    if (workit->ddsprop!=sit->ddsprop) continue;
	    break;
	  }
	}
      }
    }
#endif
  }
  return 0;
}

std::string buildSocketParameterErrorMsg(unsigned reqMask, unsigned validMask, const char* headerMsg)
{
  // build error message for required parameters
  std::stringstream errmsg;
  errmsg << headerMsg;
  for (int key=0; key<lastsocketkey; key++) {
    if (!(reqMask&(0x1<<key))) continue;
    if (validMask&(0x1<<key)) continue;
    errmsg << " '" << socketkeys[key] << "'";
  }
  return errmsg.str();
}
