/**
 * HistogramMerger.h
 *
 * @since 2014-10-10
 * @author Patryk Lesiak
 */

#pragma once

#include <TApplication.h>
#include <memory>
#include <TH1F.h>
#include <TMessage.h>
#include <FairMQDevice.h>
#include <string>
#include <unordered_map>

class HistogramMerger : public FairMQDevice
{
public:
    HistogramMerger(std::string producerId, int numIoThreads);
    virtual ~HistogramMerger();

    static void CustomCleanup(void* data, void* hint);
    void establishChannel(std::string type,
                          std::string method,
                          std::string address,
                          std::string channelName);
    void executeRunLoop();

protected:
    virtual void Run();

private:
    TObject* mergeObjectWithGivenCollection(TObject* receivedHistogram, TCollection* mergeList);
    TH1F* receiveHistogramFromProducer();
    void sendHistogramToViewer(TMessage* viewerMessage, std::unique_ptr<FairMQMessage> viewerReply);
    void sendReplyToProducer(std::string* message);
    TCollection* addReceivedObjectToMapByName(TObject* receivedObject);
    TMessage* createTMessageForViewer(TH1F* receivedHistogram, TCollection* histogramsList);
    void handleSystemCommunicationWithController();
    void handleReceivedHistograms();

    std::unordered_map<std::string, std::shared_ptr<TCollection>> mHistogramIdTohistogramMap;
};

