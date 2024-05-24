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

// \brief A thin wrapper to manage dynamic library loading, based around boost::dll

#ifndef DLLOADER_H_
#define DLLOADER_H_

#include "Framework/Logger.h"

#include <boost/dll.hpp>
#include <boost/function.hpp>
#include <boost/core/demangle.hpp>

#include <filesystem>
#include <optional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <functional>
#include <typeinfo>

namespace o2::utils
{

// Manages dynamic loading and unloading (or rather ensuring they are not
// unloaded for the duration of the program) of libraries and symbol lookups. It
// ensures thread-safety through the use of a mutex and implements the Meyers
// Singleton pattern to provide a single instance of the manager.
template <typename DerivedType>
class DLLoaderBase
{

 public:
  using library_t = std::shared_ptr<boost::dll::shared_library>;

  // Returns the singleton instance of the manager. Any function should only be
  // accessed through an instance. This being a singleton serves two purposes:
  // 1. Libraries are loaded only once.
  // 2. Once loaded they are not again unloaded until the end of the program.
  static DerivedType& Instance()
  {
    return DerivedType::sInstance;
  }

  // Loads a dynamic library by its name and stores its handle. Returns true
  // if the library is successfully loaded or already loaded.
  bool addLibrary(const std::string& library)
  {
    const std::lock_guard lock(mLock);

    if (mLibraries.find(library) != mLibraries.end()) {
      return true; // Library already loaded
    }

    if (mO2Path.empty()) {
      if (const auto* path = std::getenv("O2_ROOT")) {
        mO2Path = path;
      } else {
        LOGP(error, "$O2_ROOT not set!");
        return false;
      }
    }

    auto path = getO2Path(library);
    if (auto dpath{boost::dll::shared_library::decorate(path)}; !std::filesystem::exists(dpath.c_str())) {
      LOGP(error, "Library under '{}' does not exist!", dpath.c_str());
      return false;
    }

    try {
      auto lib = std::make_shared<boost::dll::shared_library>(path,
                                                              boost::dll::load_mode::append_decorations |
                                                                boost::dll::load_mode::rtld_lazy);
      if (lib == nullptr) {
        throw std::runtime_error("Library handle is nullptr!");
      }
      mLibraries[library] = std::move(lib);
      LOGP(info, "Loaded dynamic library '{}' from '{}'", library, path);
      return true;
    } catch (std::exception& e) {
      LOGP(error, "Failed to load library (path='{}'), failed reason: '{}'", boost::dll::shared_library::decorate(path).c_str(), e.what());
      return false;
    } catch (...) {
      LOGP(error, "Failed to load library (path='{}') for unknown reason!", boost::dll::shared_library::decorate(path).c_str());
      return false;
    }
  }

  // Unloads a given library returns true if this succeeded.
  //
  // Nota bene: Actually, we have very little control here when the unloading
  // hapens since the linkder decides this based on if there is any reference
  // left. And even if the reference counter goes to zero the linker is free to
  // leave the library loaded and clean up whenever it wants.
  bool unloadLibrary(const std::string& library)
  {
    const std::lock_guard lock(mLock);

    if (auto it = mLibraries.find(library); it != mLibraries.end()) {
      mLibraries.erase(it);
      return true;
    }

    LOGP(error, "No '{}' library found, cannot unload it!", library);
    return false;
  }

  // Resets all loaded libraries and O2Path, this invalidates all outside kept
  // references.
  void reset()
  {
    mO2Path.clear();
    mLibraries.clear();
  }

  // Checks if a library contains a specific symbol.
  bool hasSymbol(const std::string& library, const std::string& symbol)
  {
    const std::lock_guard lock(mLock);

    if (mLibraries.find(library) == mLibraries.end()) {
      // Library not loaded, attempt to load it
      if (!addLibrary(library)) {
        return false;
      }
    }

    // Checks if the symbol exists but does not load it.
    return mLibraries[library]->has(symbol);
  }

  // Gets a function alias from a loaded library.
  template <typename ProtoType>
  std::optional<boost::function<ProtoType>> getFunctionAlias(const std::string& library, const std::string& fname)
  {
    const std::lock_guard lock(mLock);

    if (fname.empty()) {
      LOGP(error, "Function name cannot be empty!");
      return std::nullopt;
    }

    if (mLibraries.find(library) == mLibraries.end()) {
      // Library not loaded, attempt to load it
      if (!addLibrary(library)) {
        return std::nullopt;
      }
    }

    const auto& lib = *mLibraries[library];
    if (!lib.has(fname)) {
      LOGP(error, "Library '{}' does not have a symbol '{}'", library, fname);
      return std::nullopt;
    }

    boost::function<ProtoType> func = boost::dll::import_alias<ProtoType>(lib, fname);
    if (func.empty()) {
      LOGP(error, "Library '{}' does not have a symbol '{}' with {}", library, fname, getTypeName<ProtoType>());
      return std::nullopt;
    }
    return func;
  }

  // Immediatley executes a function alias from a loaded library.
  template <typename Ret, typename... Args>
  Ret executeFunctionAlias(const std::string& library, const std::string& fname, Args... args)
  {
    const std::lock_guard lock(mLock);

    using ProtoType = Ret(Args...);
    if (auto func = getFunctionAlias<ProtoType>(library, fname)) {
      return (*func)(args...);
    }

    // cannot execute function at all
    throw std::runtime_error(fmt::format("Cannot get '{}' from '{}'", fname, library));
  }

  // Prints information about the loaded libraries and their symbols.
  void print(bool verbose = false)
  {
    const std::lock_guard lock(mLock);

    if (mO2Path.empty() || mLibraries.empty()) {
      LOGP(info, "No libraries added!");
    }

    LOGP(info, "Printing loaded dynamic library information:");
    for (int i{0}; const auto& [library, handle] : mLibraries) {
      LOGP(info, " {: <3d}: {}", i++, library);
      if (verbose) {
        boost::dll::library_info libInfo{boost::dll::shared_library::decorate(getO2Path(library))};
        for (int j{0}; const auto& sec : libInfo.sections()) {
          LOGP(info, "         SECTION {: <3d}: {}", j++, sec);
          for (int k{0}; const auto& symb : libInfo.symbols(sec)) {
            LOGP(info, "                         SYMBOL {: <3d}: {}", k++, symb);
          }
        }
      }
    }
  }

  // Delete copy and move constructors to enforce singleton pattern.
  DLLoaderBase(const DLLoaderBase&) = delete;
  DLLoaderBase& operator=(const DLLoaderBase&) = delete;
  DLLoaderBase(DLLoaderBase&&) = delete;
  DLLoaderBase& operator=(DLLoaderBase&&) = delete;

 protected:
  // Constructor and destructor are protected to enforce singleton pattern.
  DLLoaderBase() = default;
  ~DLLoaderBase() = default;

 private:
  // Returns the full path to a O2 shared library..
  [[nodiscard]] std::string getO2Path(const std::string& library) const
  {
    return mO2Path + "/lib/" + library;
  }

  // Returns the demangled type name of a prototype, e.g., for pretty printing.
  template <typename ProtoType>
  [[nodiscard]] auto getTypeName() -> std::string
  {
    return boost::core::demangle(typeid(ProtoType).name());
  }

  std::unordered_map<std::string, library_t> mLibraries{}; // Pointers to loaded libraries, calls `unload()' for each library, e.g., correctly destroy this object
  std::recursive_mutex mLock{};                            // While a recursive mutex is more expansive it makes locking easier
  std::string mO2Path{};                                   // Holds the path O2 dynamic library determined by $O2_ROOT
};

} // namespace o2::utils

#define O2DLLoaderDef(classname) \
 private:                        \
  static classname sInstance;    \
  classname() = default;         \
  friend class o2::utils::DLLoaderBase<classname>;

#define O2DLLoaderImpl(classname) classname classname::sInstance;

#endif // DLLOADER_H_
