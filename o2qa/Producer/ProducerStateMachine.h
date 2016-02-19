#pragma once

#include <string>
#include <TH1F.h>
#include <FairMQDevice.h>
#include <memory>
#include "Producer.h"

class ProducerStateMachine : public FairMQDevice
{
public:
    ProducerStateMachine(std::string producerId, std::string histogramId, float xLow, float xUp, int numIoThreads);
    virtual ~ProducerStateMachine() = default;

    void executeRunLoop();
    void establishChannel(std::string type, std::string method, std::string address, std::string channelName);

protected:
    ProducerStateMachine() = default;
    virtual void Run();
    TH1F* createHistogram();
    static void CustomCleanup(void* data, void* hint);

private:
    std::shared_ptr<Producer> mProducer;
};
