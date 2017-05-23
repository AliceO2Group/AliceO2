#ifndef TIMEFRAME_PARSER_H_
#define TIMEFRAME_PARSER_H_

#include <iosfwd>
#include <functional>

class FairMQParts;

namespace o2 { namespace DataFlow {

/// An helper function which takes a std::istream pointing
/// to a naively persisted timeframe and pumps its parts to
/// FairMQParts, ready to be shipped via FairMQ.
void streamTimeframe(std::istream &stream,
                     std::function<void(FairMQParts &parts, char *buffer, size_t size)> onAddPart,
                     std::function<void(FairMQParts &parts)> onSend);

void streamTimeframe(std::ostream &stream, FairMQParts &parts);

} } // end

#endif // TIMEFRAME_PARSER_H
