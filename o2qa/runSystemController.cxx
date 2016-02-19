/**
 * runSystemController.cxx
 *
 * @since 2015-10-21
 * @author Patryk Lesiak
 */

#include <csignal>
#include <FairMQTransportFactoryZMQ.h>

#include "SystemController.h"

using namespace std;

SystemController systemController("CentralSystemController", "systemController_log.txt", 1);

namespace
{

void signal_handler(int signal)
{
    LOG(INFO) << "Caught signal " << signal;
    systemController.ChangeState(SystemController::END);
    LOG(INFO) << "Caught signal " << signal;
}

}

int main(int argc, char** argv)
{

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "SystemController id: " 
              << systemController.GetProperty(SystemController::Id, "default_id");

    systemController.establishChannel("req", "connect", "tcp://localhost:5001", "data");

    systemController.executeRunLoop();
    
}
