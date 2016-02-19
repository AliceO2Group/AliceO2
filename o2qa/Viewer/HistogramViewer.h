/**
 * HistogramViewer.h
 *
 * @since 2014-10-10
 * @author Patryk Lesiak
 */

#pragma once
#include <TApplication.h>
#include <memory>
#include <TH1F.h>
#include <TList.h>
#include <TCanvas.h>
#include <FairMQDevice.h>
#include <string>
#include <unordered_set>

class HistogramViewer : public FairMQDevice
{
public:
    HistogramViewer(std::string viewerId, int numIoThreads);
    virtual ~HistogramViewer();

    static void CustomCleanup(void *data, void* hint);
    void executeRunLoop();
    void establishChannel(std::string type, std::string method, std::string address, std::string channelName);
protected:
    HistogramViewer();
    virtual void Run();

private:
    TCanvas* mHistogramCanvas;
    std::unordered_set<std::string> mNamesOfHistogramsToDraw;

    std::unique_ptr<FairMQMessage> receiveMessageFromMerger();
    void sendReplyToMerger(std::string* message);
    TH1F* receiveHistogramFromMerger();
    void updateHistogramCanvas(TH1F* receivedHistogram);
    void updateCanvas(TH1F* receivedHistogra);
};

