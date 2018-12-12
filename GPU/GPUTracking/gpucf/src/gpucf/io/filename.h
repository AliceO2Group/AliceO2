#pragma once

#include <string>


bool isBinary(const std::string &name);

bool isText(const std::string &name);

std::string getExt(const std::string &name);
std::string getHead(const std::string &name);

bool fileExists(const std::string &name);

// vim: set ts=4 sw=4 sts=4 expandtab:
