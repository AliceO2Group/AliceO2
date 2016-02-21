#pragma once

#include <TApplication.h>
#include <memory>
#include <TMessage.h>
#include <FairMQDevice.h>
#include <string>
#include <unordered_map>
 
#include "Merger.h"

class MergerDevice : public FairMQDevice
{
public:
    MergerDevice(std::unique_ptr<Merger> merger, std::string producerId, int numIoThreads);
    virtual ~MergerDevice() = default;

    static void CustomCleanup(void* data, void* hint);
    void establishChannel(std::string type,
                          std::string method,
                          std::string address,
                          std::string channelName);
    void executeRunLoop();

protected:
    virtual void Run();

private:
    TObject* receiveDataObjectFromProducer();
    void sendReplyToProducer(std::string* message);

    TMessage* createTMessageForViewer(std::shared_ptr<TObject> objectToSend);
    void sendMergedObjectToViewer(TMessage* viewerMessage, std::unique_ptr<FairMQMessage> viewerReply);

    void handleSystemCommunicationWithController();
    void handleReceivedDataObject();

    std::unique_ptr<Merger> mMerger;
};

