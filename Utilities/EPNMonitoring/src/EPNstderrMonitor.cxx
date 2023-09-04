// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EPNstderrMonitor.cxx
/// \author David Rohr

#include <fairmq/Device.h>
#include <fairmq/runDevice.h>

#include "InfoLogger/InfoLogger.hxx"

#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <regex>
#include <filesystem>
#include <chrono>
#include <fstream>

#include <unistd.h>
#include <sys/inotify.h>
#include <poll.h>

using namespace AliceO2;

static constexpr size_t MAX_LINES_FILE = 30;
static constexpr size_t MAX_BYTES_FILE = MAX_LINES_FILE * 512;
static constexpr size_t MAX_LINES_TOTAL = 1000;
static constexpr size_t MAX_BYTES_TOTAL = MAX_LINES_TOTAL * 256;

struct fileMon {
  std::ifstream file;
  std::string name;
  unsigned int nLines = 0;
  unsigned int nBytes = 0;
  bool stopped = false;

  fileMon(const std::string& path, const std::string& filename);
  fileMon(const std::string& filename, std::ifstream&& f);
};

fileMon::fileMon(const std::string& path, const std::string& filename) : name(filename)
{
  printf("Monitoring file %s\n", filename.c_str());
  file.open(path + "/" + filename, std::ifstream::in);
}

fileMon::fileMon(const std::string& filename, std::ifstream&& f) : file(std::move(f)), name(filename)
{
}

class EPNMonitor
{
 public:
  EPNMonitor(std::string path, bool infoLogger, int runNumber, std::string partition);
  ~EPNMonitor();
  void setRunNr(int nr) { mRunNumber = nr; }

 private:
  void thread();
  void check_add_file(const std::string& filename);
  void sendLog(const std::string& file, const std::string& message);

  bool mInfoLoggerActive;
  volatile bool mTerminate = false;
  std::thread mThread;
  std::unordered_map<std::string, fileMon> mFiles;
  std::string mPath;
  std::vector<std::regex> mFilters;
  volatile unsigned int mRunNumber;
  std::string mPartition;
  unsigned int nLines = 0;
  unsigned int nBytes = 0;
  std::unique_ptr<InfoLogger::InfoLogger> mLogger;
  std::unique_ptr<InfoLogger::InfoLoggerContext> mLoggerContext;
};

EPNMonitor::EPNMonitor(std::string path, bool infoLogger, int runNumber, std::string partition)
{
  mFilters.emplace_back("^Info in <");
  mFilters.emplace_back("^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}");
  mFilters.emplace_back("^Warning in <Fit");
  mFilters.emplace_back("^Warning in <TGraph");
  mFilters.emplace_back("^Warning in <TInterpreter");
  mFilters.emplace_back("Dividing histograms with different labels");
  mInfoLoggerActive = infoLogger;
  mPath = path;
  mRunNumber = runNumber;
  mPartition = partition;
  if (infoLogger) {
    mLogger = std::make_unique<InfoLogger::InfoLogger>();
    mLoggerContext = std::make_unique<InfoLogger::InfoLoggerContext>();
    mLoggerContext->setField(InfoLogger::InfoLoggerContext::FieldName::Partition, partition != "" ? partition : "unspecified");
    mLoggerContext->setField(InfoLogger::InfoLoggerContext::FieldName::System, std::string("STDERR"));
  }
  mThread = std::thread(&EPNMonitor::thread, this);
}

EPNMonitor::~EPNMonitor()
{
  mTerminate = true;
  mThread.join();
}

void EPNMonitor::check_add_file(const std::string& filename)
{
  //printf("Checking '%s'\n", filename.c_str());
  static const std::regex match_stderr("_err\\.log$");
  if (std::regex_search(filename, match_stderr)) {
    mFiles.try_emplace(filename, mPath, filename);
  }
}

void EPNMonitor::sendLog(const std::string& file, const std::string& message)
{
  if (mInfoLoggerActive) {
    mLoggerContext->setField(InfoLogger::InfoLoggerContext::FieldName::Facility, ("stderr/" + file).substr(0, 31));
    mLoggerContext->setField(InfoLogger::InfoLoggerContext::FieldName::Run, mRunNumber != 0 ? std::to_string(mRunNumber) : "unspecified");
    static const InfoLogger::InfoLogger::InfoLoggerMessageOption opt = {InfoLogger::InfoLogger::Severity::Error, 3, InfoLogger::InfoLogger::undefinedMessageOption.errorCode, InfoLogger::InfoLogger::undefinedMessageOption.sourceFile, InfoLogger::InfoLogger::undefinedMessageOption.sourceLine};
    mLogger->log(opt, *mLoggerContext, "stderr: %s", message.c_str());
  } else {
    printf("stderr: %s: %s\n", file.c_str(), message.c_str());
  }
}

void EPNMonitor::thread()
{
  printf("EPN stderr Monitor active\n");

  try {
    std::string syslogfile = "/var/log/infologger_syslog";
    std::ifstream file;
    file.open(syslogfile, std::ifstream::in);
    file.seekg(0, file.end);
    mFiles.emplace(std::piecewise_construct, std::forward_as_tuple(syslogfile), std::forward_as_tuple(std::string("SYSLOG"), std::move(file)));
  } catch (...) {
  }

  int fd;
  int wd;
  static constexpr size_t BUFFER_SIZE = 64 * 1024;
  std::vector<char> evt_buffer(BUFFER_SIZE);
  std::vector<char> text_buffer(8192);
  fd = inotify_init();
  wd = inotify_add_watch(fd, mPath.c_str(), IN_CREATE);
  if (fd < 0) {
    throw std::runtime_error(std::string("Error initializing inotify ") + std::to_string(fd) + " " + std::to_string(wd));
  }
  pollfd pfd = {fd, POLLIN, 0};

  for (const auto& entry : std::filesystem::directory_iterator(mPath)) {
    if (entry.is_regular_file()) {
      check_add_file(entry.path().filename());
    }
  }

  auto lastTime = std::chrono::system_clock::now();
  while (!mTerminate) {
    if (poll(&pfd, 1, 50) > 0) {
      int l = read(fd, evt_buffer.data(), BUFFER_SIZE);
      if (l < 0) {
        throw std::runtime_error(std::string("Error waiting for inotify event ") + std::to_string(l));
      }
      for (int i = 0; i < l; i += sizeof(inotify_event)) {
        inotify_event* event = (inotify_event*)&evt_buffer[i];
        if (event->len && (event->mask & IN_CREATE) && !(event->mask & IN_ISDIR)) {
          check_add_file(event->name);
        }
        i += event->len;
      }
    }
    auto curTime = std::chrono::system_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(curTime - lastTime).count() >= 1000) {
      char* ptr = text_buffer.data();
      std::string line;
      for (auto fit = mFiles.begin(); fit != mFiles.end(); fit++) {
        auto& f = fit->second;
        if (f.stopped) {
          continue;
        }
        auto& file = f.file;
        file.clear();
        do {
          std::getline(file, line);
          if (line.size()) {
            bool filterLine = false;
            for (const auto& filter : mFilters) {
              if (std::regex_search(line, filter)) {
                filterLine = true;
                break;
              }
            }
            if (filterLine) {
              continue;
            }
            f.nLines++;
            f.nBytes += line.size();
            nLines++;
            nBytes += line.size();
            if (f.nLines >= MAX_LINES_FILE || f.nBytes >= MAX_BYTES_FILE) {
              sendLog(f.name, "Exceeded log size for process " + f.name + " (" + std::to_string(f.nLines) + " lines, " + std::to_string(f.nBytes) + " bytes), not reporting any more errors from this file...");
              f.stopped = true;
              break;
            }
            if (nLines >= MAX_LINES_TOTAL || nBytes >= MAX_BYTES_TOTAL) {
              break;
            }
            sendLog(f.name, line);
          }
        } while (!file.eof());
      }
      lastTime = curTime;
    }
    if (nLines >= MAX_LINES_TOTAL || nBytes >= MAX_BYTES_TOTAL) {
      sendLog("", "Max total stderr log size exceeded (" + std::to_string(nLines) + " lines, " + std::to_string(nBytes) + "), not sending any more stderr logs from this node...");
      break;
    }

    usleep(50000);
  }

  inotify_rm_watch(fd, wd);
  close(fd);

  printf("EPN stderr Monitor terminating\n");
}

static std::unique_ptr<EPNMonitor> gEPNMonitor;

namespace bpo = boost::program_options;

struct EPNstderrMonitor : fair::mq::Device {
  void InitTask() override
  {
    std::string path = getenv("DDS_LOCATION") ? (std::string(getenv("DDS_LOCATION")) + "/") : std::string(".");
    bool infoLogger = fConfig->GetProperty<int>("infologger");
    bool dds = false;

    std::string partition = "";
    try {
      partition = fConfig->GetProperty<std::string>("environment_id", "");
      printf("Got environment_id: %s\n", partition.c_str());
    } catch (...) {
      printf("Error getting environment_id\n");
    }

    gEPNMonitor = std::make_unique<EPNMonitor>(path, infoLogger, 0, partition);
  }
  void PreRun() override
  {
    int runNumber = 0;
    try {
      runNumber = atoi(fConfig->GetProperty<std::string>("runNumber", "").c_str());
      printf("Got runNumber: %d\n", runNumber);
    } catch (...) {
      printf("Error getting runNumber\n");
    }
    gEPNMonitor->setRunNr(runNumber);
  }
  bool ConditionalRun() override
  {
    usleep(100000);
    return true;
  }
};

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()("infologger", bpo::value<int>()->default_value(0), "Send via infologger");
}

std::unique_ptr<fair::mq::Device> getDevice(fair::mq::ProgOptions& config)
{
  return std::make_unique<EPNstderrMonitor>();
}
