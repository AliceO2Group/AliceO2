#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <string>
#include <thread>

#include "QCMetricsExtractor/MetricsExtractor.h"

using namespace std;
using namespace boost::property_tree;
using namespace boost::posix_time;
using namespace boost::gregorian;
using namespace dds::intercom_api;

namespace o2
{
namespace qc
{
MetricsExtractor::MetricsExtractor(const char* testName) : mTestName(testName), ddsCustomCmd(new CCustomCmd(mService))
{
  metricsOutputFile.open(string("/net/scratch/people/plglesiak/metrics_") + string(mTestName) + string(".json"));
  stateOutputFile.open(string("/net/scratch/people/plglesiak/state_") + string(mTestName) + string(".json"));
}

MetricsExtractor::~MetricsExtractor()
{
  metricsOutputFile.close();
  stateOutputFile.close();
}

void MetricsExtractor::runMetricsExtractor()
{
  try {
    mService.subscribeOnError([](const EErrorCode _errorCode, const string& _errorMsg) {
      cout << "Error received: error code: " << _errorCode << ", error message: " << _errorMsg << endl;
    });

    ddsCustomCmd->subscribe([&](const string& command, const string& condition, uint64_t senderId) {
      stringstream ss;
      ss << command;
      ptree response;
      read_json(ss, response);

      response.put("name", mTestName);

      if (response.get<string>("command") == "state") {
        stateOutputFile << convertToString(response) << endl;
      } else if (response.get<string>("command") == "metrics") {
        metricsOutputFile << convertToString(response) << endl;
      }
    });

    ddsCustomCmd->subscribeOnReply([](const string& _msg) {});

    mService.start();

    while (true) {
      this_thread::sleep_for(chrono::seconds(1));
      sendCheckStateDdsCustomCommand();
      sendGetMetrics();
    }
  } catch (exception& ex) {
    cerr << "Error sending custom command: " << ex.what();
  }
}

string MetricsExtractor::convertToString(ptree& command)
{
  ostringstream commandStream;
  write_json(commandStream, command);

  string fileOutput = commandStream.str();

  boost::erase_all(fileOutput, "\n");
  boost::erase_all(fileOutput, " ");

  return fileOutput;
}

void MetricsExtractor::sendCheckStateDdsCustomCommand()
{
  ptree request;
  request.put("command", "check-state");
  request.put("requestTimestamp", to_iso_extended_string(second_clock::local_time()).substr(0, 19));

  stringstream ss;
  write_json(ss, request);

  ddsCustomCmd->send(ss.str(), "");
}

void MetricsExtractor::sendGetMetrics()
{
  ptree request;
  request.put("command", "get-metrics");
  request.put("requestTimestamp", to_iso_extended_string(second_clock::local_time()).substr(0, 19));

  stringstream ss;
  write_json(ss, request);

  ddsCustomCmd->send(ss.str(), "");
}
}
}