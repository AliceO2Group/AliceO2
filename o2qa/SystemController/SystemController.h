#pragma once

#include <string>
#include <FairMQDevice.h>
#include <fstream>
#include <memory>

class SystemController : public FairMQDevice
{
public:
  SystemController(std::string controllerId, std::string logFileName, int numIoThreads);
  virtual ~SystemController();

  void establishChannel(std::string type, std::string method, std::string address, std::string channelName);
  void executeRunLoop();
  static void CustomCleanup(void* data, void* hint);

protected:
  virtual void Run();

private:
  std::ofstream mLogFile;

  void getStatusFromSystemNodes();
  std::string getCurrentTime();
  FairMQMessage* sendMessageToNodes();
};
