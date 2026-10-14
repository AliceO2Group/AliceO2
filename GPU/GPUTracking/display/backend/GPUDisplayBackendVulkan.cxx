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

/// \file GPUDisplayBackendVulkan.cxx
/// \author David Rohr

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#include "HandMadeMath.h"

#include "GPUCommonDef.h"
#include "GPUDisplayBackendVulkan.h"
#include "GPUDisplay.h"
#include "GPULogging.h"
#include "GPUParam.h"

#include <mutex>
#include <iostream>

using namespace o2::gpu;

#define GPUCA_DISPLAY_SHADER_TYPE_VULKAN
#include "GPUDisplayLoadShaders.h"

// #define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                                         \
  do {                                                                                                      \
    auto tmp_internal_retVal = cmd;                                                                         \
    if ((int32_t)tmp_internal_retVal < 0) {                                                                 \
      GPUError("VULKAN ERROR: %d: %s (%s: %d)", (int32_t)tmp_internal_retVal, "ERROR", __FILE__, __LINE__); \
      throw std::runtime_error("Vulkan Failure");                                                           \
    }                                                                                                       \
  } while (false)

GPUDisplayBackendVulkan::GPUDisplayBackendVulkan()
{
  mBackendType = TYPE_VULKAN;
  mBackendName = "Vulkan";
}
GPUDisplayBackendVulkan::~GPUDisplayBackendVulkan() = default;

// ---------------------------- VULKAN HELPERS ----------------------------

static int32_t checkVulkanLayersSupported(const std::vector<const char*>& validationLayers)
{
  std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return 1;
    }
  }
  return 0;
}

static uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice physDev)
{
  vk::PhysicalDeviceMemoryProperties memProperties = physDev.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  if ((properties & vk::MemoryPropertyFlagBits::eDeviceLocal) && (properties & vk::MemoryPropertyFlagBits::eHostVisible)) { // No host-visible device memory (e.g. no ReBAR), fall back to host memory
    return findMemoryType(typeFilter, properties & ~vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal), physDev);
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

vk::PresentModeKHR GPUDisplayBackendVulkan::chooseSwapPresentMode()
{
  // Lowest latency first. VSync: no tearing, MAILBOX, then FIFO_LATEST_READY (shows the newest ready image at each vblank). No VSync: IMMEDIATE, else the same (often no IMMEDIATE on Wayland). FIFO is always supported.
  mPresentModeForVSyncLimit = fpsLimitMode() == FPS_LIMIT_VSYNC;
  if (mPresentModeForVSyncLimit) {
    return vk::PresentModeKHR::eFifo; // Refresh rate limit: one frame per refresh, queue depth limited by present wait
  }
  const bool vsync = mDisplay->cfgR().drawQualityVSync;
  const auto& available = mSwapChainDetails.presentModes;
  for (vk::PresentModeKHR mode : {vk::PresentModeKHR::eImmediate, vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifoLatestReady, vk::PresentModeKHR::eFifo}) {
    if ((mode == vk::PresentModeKHR::eImmediate && vsync) || (mode == vk::PresentModeKHR::eFifoLatestReady && !mFifoLatestReadySupported) || std::find(available.begin(), available.end(), mode) == available.end()) {
      continue;
    }
    if (mode != (vsync ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eImmediate) && !mPresentModeErrorShown) {
      mPresentModeErrorShown = true;
      GPUError("VULKAN ERROR: Desired present mode %s not available, using %s", vsync ? "MAILBOX" : "IMMEDIATE", vk::to_string(mode).c_str());
    }
    return mode;
  }
  return vk::PresentModeKHR::eFifo; // Unreachable, FIFO is always supported
}

vk::Extent2D GPUDisplayBackendVulkan::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int32_t width, height;
    mDisplay->frontend()->getSize(width, height);
    vk::Extent2D actualExtent = {(uint32_t)width, (uint32_t)height};
    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

static void cmdImageMemoryBarrier(vk::CommandBuffer cmdbuffer, vk::Image image, vk::PipelineStageFlags2 srcStageMask, vk::AccessFlags2 srcAccessMask, vk::PipelineStageFlags2 dstStageMask, vk::AccessFlags2 dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
{
  vk::ImageMemoryBarrier2 barrier{srcStageMask, srcAccessMask, dstStageMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, {aspect, 0, 1, 0, 1}};
  cmdbuffer.pipelineBarrier2(vk::DependencyInfo{{}, 0, nullptr, 0, nullptr, 1, &barrier});
}

static void cmdBeginRendering(vk::CommandBuffer cmdbuffer, uint32_t width, uint32_t height, const vk::RenderingAttachmentInfo& color, const vk::RenderingAttachmentInfo* depth = nullptr)
{
  cmdbuffer.beginRendering(vk::RenderingInfo{{}, vk::Rect2D{{0, 0}, {width, height}}, 1, 0, 1, &color, depth});
  vk::Viewport viewport{0.f, 0.f, (float)width, (float)height, 0.f, 1.f};
  vk::Rect2D scissor{{0, 0}, {width, height}};
  cmdbuffer.setViewport(0, 1, &viewport);
  cmdbuffer.setScissor(0, 1, &scissor);
}

static void cmdBeginRenderingLoad(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageView view, uint32_t width, uint32_t height) // Render on top of the content of a color attachment written by a previous pass
{
  cmdImageMemoryBarrier(cmdbuffer, image, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
  cmdBeginRendering(cmdbuffer, width, height, vk::RenderingAttachmentInfo{view, vk::ImageLayout::eColorAttachmentOptimal}); // loadOp eLoad, storeOp eStore
}

void GPUDisplayBackendVulkan::updateSwapChainDetails(const vk::PhysicalDevice& device)
{
  mSwapChainDetails.capabilities = device.getSurfaceCapabilitiesKHR(mSurface);
  mSwapChainDetails.formats = device.getSurfaceFormatsKHR(mSurface);
  mSwapChainDetails.presentModes = device.getSurfacePresentModesKHR(mSurface);
}

vk::CommandBuffer GPUDisplayBackendVulkan::getSingleTimeCommandBuffer()
{
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = mCommandPool;
  allocInfo.commandBufferCount = 1;
  vk::CommandBuffer commandBuffer = mDevice.allocateCommandBuffers(allocInfo)[0];
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);
  return commandBuffer;
}

template <class... Args>
uint64_t GPUDisplayBackendVulkan::queueSubmit(uint64_t* timelineSignalValue, Args&&... args)
{
  std::lock_guard<std::mutex> guard(mQueueMutex);
  if (timelineSignalValue) {
    *timelineSignalValue = ++mTimelineValue;
  }
  CHKERR(mGraphicsQueue.submit2(std::forward<Args>(args)...));
  return mTimelineValue;
}

void GPUDisplayBackendVulkan::submitSingleTimeCommandBuffer(vk::CommandBuffer commandBuffer)
{
  commandBuffer.end();
  vk::CommandBufferSubmitInfo commandBufferInfo{commandBuffer};
  vk::SemaphoreSubmitInfo signalInfo{mTimelineSemaphore, 0, vk::PipelineStageFlagBits2::eAllCommands};
  vk::SubmitInfo2 submitInfo{{}, 0, nullptr, 1, &commandBufferInfo, 1, &signalInfo};
  waitTimeline(queueSubmit(&signalInfo.value, 1, &submitInfo, VkFence(VK_NULL_HANDLE)));
  mDevice.freeCommandBuffers(mCommandPool, 1, &commandBuffer);
}

void GPUDisplayBackendVulkan::waitTimeline(uint64_t value)
{
  if (value) {
    vk::SemaphoreWaitInfo waitInfo{{}, 1, &mTimelineSemaphore, &value};
    CHKERR(mDevice.waitSemaphores(&waitInfo, UINT64_MAX));
  }
}

static vk::ImageView createImageViewI(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor, uint32_t mipLevels = 1)
{
  vk::ImageViewCreateInfo viewInfo{};
  viewInfo.image = image;
  viewInfo.viewType = vk::ImageViewType::e2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = mipLevels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  return device.createImageView(viewInfo, nullptr);
}

static void createImageI(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Image& image, vk::DeviceMemory& imageMemory, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::ImageTiling tiling = vk::ImageTiling::eOptimal, vk::SampleCountFlagBits numSamples = vk::SampleCountFlagBits::e1, vk::ImageLayout layout = vk::ImageLayout::eUndefined, uint32_t mipLevels = 1)
{
  vk::ImageCreateInfo imageInfo{};
  imageInfo.imageType = vk::ImageType::e2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = layout;
  imageInfo.usage = usage;
  imageInfo.samples = numSamples;
  imageInfo.sharingMode = vk::SharingMode::eExclusive;
  image = device.createImage(imageInfo);

  vk::MemoryRequirements memRequirements;
  memRequirements = device.getImageMemoryRequirements(image);

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);
  imageMemory = device.allocateMemory(allocInfo, nullptr);

  device.bindImageMemory(image, imageMemory, 0);
}

static uint32_t getMaxUsableSampleCount(vk::PhysicalDeviceProperties& physicalDeviceProperties)
{
  vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
  if (counts & vk::SampleCountFlagBits::e64) {
    return 64;
  } else if (counts & vk::SampleCountFlagBits::e32) {
    return 32;
  } else if (counts & vk::SampleCountFlagBits::e16) {
    return 16;
  } else if (counts & vk::SampleCountFlagBits::e8) {
    return 8;
  } else if (counts & vk::SampleCountFlagBits::e4) {
    return 4;
  } else if (counts & vk::SampleCountFlagBits::e2) {
    return 2;
  }
  return 1;
}

static vk::SampleCountFlagBits getMSAASamplesFlag(uint32_t msaa)
{
  if (msaa == 2) {
    return vk::SampleCountFlagBits::e2;
  } else if (msaa == 4) {
    return vk::SampleCountFlagBits::e4;
  } else if (msaa == 8) {
    return vk::SampleCountFlagBits::e8;
  } else if (msaa == 16) {
    return vk::SampleCountFlagBits::e16;
  } else if (msaa == 32) {
    return vk::SampleCountFlagBits::e32;
  } else if (msaa == 64) {
    return vk::SampleCountFlagBits::e64;
  }
  return vk::SampleCountFlagBits::e1;
}

template <class T, class S>
static inline void clearVector(T& v, S func, bool downsize = true)
{
  std::for_each(v.begin(), v.end(), func);
  if (downsize) {
    v.clear();
  }
}

// ---------------------------- VULKAN DEVICE MANAGEMENT ----------------------------

template <class T> // Feature structs: sType, pNext, then only VkBool32 members (possibly followed by padding)
static bool hasFeatures(const T& available, const T& required)
{
  const vk::Bool32* a = reinterpret_cast<const vk::Bool32*>(reinterpret_cast<const char*>(&available) + sizeof(VkBaseOutStructure));
  const vk::Bool32* r = reinterpret_cast<const vk::Bool32*>(reinterpret_cast<const char*>(&required) + sizeof(VkBaseOutStructure));
  for (size_t i = 0; i < (sizeof(T) - sizeof(VkBaseOutStructure)) / sizeof(vk::Bool32); i++) {
    if (r[i] && !a[i]) {
      return false;
    }
  }
  return true;
}

template <class... T>
static bool hasFeatures(const vk::StructureChain<T...>& available, const vk::StructureChain<T...>& required)
{
  return (hasFeatures(available.template get<T>(), required.template get<T>()) && ...);
}

template <class... T>
static void clearFeatures(vk::StructureChain<T...>& chain) // Zero all feature flags including padding, which the member-wise comparison in hasFeatures reads
{
  ((memset(reinterpret_cast<char*>(&chain.template get<T>()) + sizeof(VkBaseOutStructure), 0, sizeof(T) - sizeof(VkBaseOutStructure))), ...);
}

template <class T> // Enable the first available of the extensions (aliases) if the device supports the feature, chaining the feature struct in front of chain
static bool enableOptionalFeature(vk::PhysicalDevice device, const std::vector<vk::ExtensionProperties>& available, std::initializer_list<const char*> extensions, vk::Bool32 T::* feature, T& featureStruct, std::vector<const char*>& enabledExtensions, void*& chain)
{
  for (const char* ext : extensions) {
    if (std::none_of(available.begin(), available.end(), [ext](const auto& e) { return strcmp(ext, e.extensionName) == 0; })) {
      continue;
    }
    if (!(device.getFeatures2<vk::PhysicalDeviceFeatures2, T>().template get<T>().*feature)) {
      return false;
    }
    enabledExtensions.emplace_back(ext);
    featureStruct.*feature = VK_TRUE;
    featureStruct.pNext = chain;
    chain = &featureStruct;
    return true;
  }
  return false;
}

template <class... T>
static vk::StructureChain<T...> getFeatures(vk::PhysicalDevice device, const vk::StructureChain<T...>&)
{
  return device.getFeatures2<T...>();
}

double GPUDisplayBackendVulkan::checkDevice(vk::PhysicalDevice device, const std::vector<const char*>& reqDeviceExtensions, const FeatureChain& reqDeviceFeatures, bool silent)
{
  double score = -1.;
  vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
  vk::PhysicalDeviceMemoryProperties memoryProperties = device.getMemoryProperties();
  if (deviceProperties.apiVersion < VK_API_VERSION_1_4) {
    if (!silent) {
      GPUInfo("%s ignored, supports only Vulkan %u.%u", &deviceProperties.deviceName[0], VK_API_VERSION_MAJOR(deviceProperties.apiVersion), VK_API_VERSION_MINOR(deviceProperties.apiVersion));
    }
    return (-1);
  }

  std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties(nullptr);
  uint32_t extensionsFound = 0;
  for (uint32_t i = 0; i < reqDeviceExtensions.size(); i++) {
    for (uint32_t j = 0; j < availableExtensions.size(); j++) {
      if (strcmp(reqDeviceExtensions[i], availableExtensions[j].extensionName) == 0) {
        extensionsFound++;
        break;
      }
    }
  }
  if (extensionsFound < reqDeviceExtensions.size()) {
    if (!silent) {
      GPUInfo("%s ignored due to missing extensions", &deviceProperties.deviceName[0]);
    }
    return (-1);
  }

  if (!hasFeatures(getFeatures(device, reqDeviceFeatures), reqDeviceFeatures)) { // Query only after the version and extension checks, the chain contains 1.4 and extension structs
    if (!silent) {
      GPUInfo("%s ignored due to missing features", &deviceProperties.deviceName[0]);
    }
    return (-1);
  }

  std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
  bool found = false;
  for (uint32_t i = 0; i < queueFamilies.size(); i++) {
    if (!(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
      continue;
    }
    vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, mSurface);
    if (!presentSupport) {
      continue;
    }
    mGraphicsFamily = i;
    found = true;
    break;
  }
  if (!found) {
    if (!silent) {
      GPUInfo("%s ignored due to missing queue properties", &deviceProperties.deviceName[0]);
    }
    return (-1);
  }

  updateSwapChainDetails(device);
  if (mSwapChainDetails.formats.empty() || mSwapChainDetails.presentModes.empty()) {
    if (!silent) {
      GPUInfo("%s ignored due to incompatible swap chain", &deviceProperties.deviceName[0]);
    }
    return (-1);
  }

  score = 1;
  if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
    score += 1e12;
  } else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
    score += 1e11;
  }

  for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
    if (memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      score += memoryProperties.memoryHeaps[i].size;
    }
  }

  return score;
}

void GPUDisplayBackendVulkan::createDevice()
{
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  vk::ApplicationInfo appInfo{};
  appInfo.pApplicationName = "GPU CA Standalone display";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "GPU CI Standalone Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_4;

  vk::InstanceCreateInfo instanceCreateInfo;
  instanceCreateInfo.pApplicationInfo = &appInfo;

  const char** frontendExtensions;
  uint32_t frontendExtensionCount = mDisplay->frontend()->getReqVulkanExtensions(frontendExtensions);
  std::vector<const char*> reqInstanceExtensions(frontendExtensions, frontendExtensions + frontendExtensionCount);
  reqInstanceExtensions.insert(reqInstanceExtensions.end(), {VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME}); // The latter two are required by VK_KHR_swapchain_maintenance1
  std::vector<vk::ExtensionProperties> availableInstanceExtensions = vk::enumerateInstanceExtensionProperties(nullptr);
  if (mDisplay->GetProcessingSettings().debugLevel >= 3) {
    std::cout << "available instance extensions: " << availableInstanceExtensions.size() << "\n";
    for (const auto& extension : availableInstanceExtensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  }
  for (const char* ext : reqInstanceExtensions) {
    if (std::none_of(availableInstanceExtensions.begin(), availableInstanceExtensions.end(), [ext](const auto& e) { return strcmp(ext, e.extensionName) == 0; })) {
      throw std::runtime_error(std::string("Required Vulkan instance extension not available: ") + ext);
    }
  }

  const std::vector<const char*> reqValidationLayers = {"VK_LAYER_KHRONOS_validation"};
  auto debugCallback = [](vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
    static int32_t throwOnError = getenv("GPUCA_VULKAN_VALIDATION_THROW") ? atoi(getenv("GPUCA_VULKAN_VALIDATION_THROW")) : 0;
    static bool showVulkanValidationInfo = getenv("GPUCA_VULKAN_VALIDATION_INFO") && atoi(getenv("GPUCA_VULKAN_VALIDATION_INFO"));
    switch (messageSeverity) {
      case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        if (showVulkanValidationInfo) {
          GPUInfo("%s", pCallbackData->pMessage);
        }
        break;
      case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        GPUWarning("%s", pCallbackData->pMessage);
        if (throwOnError > 1) {
          throw std::logic_error("break_on_validation_warning");
        }
        break;
      case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        GPUError("%s", pCallbackData->pMessage);
        if (throwOnError) {
          throw std::logic_error("break_on_validation_error");
        }
        break;
      case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
      default:
        GPUInfo("%s", pCallbackData->pMessage);
        break;
    }
    return false;
  };
  vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (mEnableValidationLayers) {
    if (checkVulkanLayersSupported(reqValidationLayers)) {
      throw std::runtime_error("Requested validation layer support not available");
    }
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(reqValidationLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = reqValidationLayers.data();
    instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

    debugCreateInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    debugCreateInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    debugCreateInfo.pfnUserCallback = debugCallback;
    debugCreateInfo.pUserData = nullptr;
  } else {
    instanceCreateInfo.enabledLayerCount = 0;
  }

  instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(reqInstanceExtensions.size());
  instanceCreateInfo.ppEnabledExtensionNames = reqInstanceExtensions.data();

  mInstance = vk::createInstance(instanceCreateInfo, nullptr);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance);

  if (mEnableValidationLayers) {
    GPUInfo("Enabling Vulkan Validation Layers");
    mDebugMessenger = mInstance.createDebugUtilsMessengerEXT(debugCreateInfo, nullptr);
  }

  if (mDisplay->frontend()->getVulkanSurface(&mInstance, &mSurface)) {
    throw std::runtime_error("Frontend does not provide Vulkan surface");
  }

  const std::vector<const char*> reqDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_KHR_PRESENT_WAIT_EXTENSION_NAME};
  FeatureChain reqDeviceFeatures; // Exactly these features are required and enabled
  clearFeatures(reqDeviceFeatures);
  auto& reqFeatures10 = reqDeviceFeatures.get<vk::PhysicalDeviceFeatures2>().features;
  reqFeatures10.geometryShader = reqFeatures10.wideLines = reqFeatures10.largePoints = reqFeatures10.multiDrawIndirect = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDeviceVulkan14Features>().pushDescriptor = reqDeviceFeatures.get<vk::PhysicalDeviceVulkan14Features>().maintenance5 = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering = reqDeviceFeatures.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDeviceVulkan13Features>().shaderDemoteToHelperInvocation = VK_TRUE; // discard in SPIR-V 1.6
  reqDeviceFeatures.get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>().swapchainMaintenance1 = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDevicePresentIdFeaturesKHR>().presentId = VK_TRUE;
  reqDeviceFeatures.get<vk::PhysicalDevicePresentWaitFeaturesKHR>().presentWait = VK_TRUE;

  mPhysicalDevice = VkPhysicalDevice(VK_NULL_HANDLE);
  std::vector<vk::PhysicalDevice> devices = mInstance.enumeratePhysicalDevices();
  if (devices.size() == 0) {
    throw std::runtime_error("No Vulkan device present!");
  }
  double bestScore = -1.;
  for (uint32_t i = 0; i < devices.size(); i++) {
    double score = checkDevice(devices[i], reqDeviceExtensions, reqDeviceFeatures);
    if (mDisplay->GetProcessingSettings().debugLevel >= 2) {
      vk::PhysicalDeviceProperties deviceProperties = devices[i].getProperties();
      GPUInfo("Available Vulkan device %d: %s - Score %f", i, &deviceProperties.deviceName[0], score);
    }
    if (score > bestScore && score > 0) {
      mPhysicalDevice = devices[i];
      bestScore = score;
    }
  }
  if (mDisplay->cfg().vulkan.forceDevice != -1) {
    if (mDisplay->cfg().vulkan.forceDevice < 0 || mDisplay->cfg().vulkan.forceDevice >= (int32_t)devices.size()) {
      throw std::runtime_error("Invalid Vulkan device selected");
    }
    mPhysicalDevice = devices[mDisplay->cfg().vulkan.forceDevice];
  }
  if (!mPhysicalDevice) {
    throw std::runtime_error("All available Vulkan devices unsuited");
  }
  if (checkDevice(mPhysicalDevice, reqDeviceExtensions, reqDeviceFeatures, true) <= 0) {
    throw std::runtime_error("Selected vulkan device is incompatible");
  }
  mSmoothLinesSupported = getFeatures(mPhysicalDevice, reqDeviceFeatures).get<vk::PhysicalDeviceVulkan14Features>().smoothLines;
  reqDeviceFeatures.get<vk::PhysicalDeviceVulkan14Features>().smoothLines = mSmoothLinesSupported;
  std::vector<const char*> deviceExtensions = reqDeviceExtensions;
  void* featureChain = &reqDeviceFeatures.get<vk::PhysicalDeviceFeatures2>();
  const std::vector<vk::ExtensionProperties> availableDeviceExtensions = mPhysicalDevice.enumerateDeviceExtensionProperties(nullptr);
  vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR fifoLatestReadyFeatures{};
  mFifoLatestReadySupported = enableOptionalFeature(mPhysicalDevice, availableDeviceExtensions, {VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME, VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME}, &vk::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR::presentModeFifoLatestReady, fifoLatestReadyFeatures, deviceExtensions, featureChain);

  updateSwapChainDetails(mPhysicalDevice);
  vk::PhysicalDeviceProperties deviceProperties = mPhysicalDevice.getProperties();
  vk::FormatProperties depth32FormatProperties = mPhysicalDevice.getFormatProperties(vk::Format::eD32Sfloat);
  vk::FormatProperties depth64FormatProperties = mPhysicalDevice.getFormatProperties(vk::Format::eD32SfloatS8Uint);
  GPUInfo("Using physical Vulkan device %s", &deviceProperties.deviceName[0]);
  mMaxMSAAsupported = getMaxUsableSampleCount(deviceProperties);
  mStrictLines = deviceProperties.limits.strictLines;
  mMaxLineWidth = deviceProperties.limits.lineWidthRange[1];
  mMaxPointSize = deviceProperties.limits.pointSizeRange[1];
  mMaxRenderSize = std::min({deviceProperties.limits.maxImageDimension2D, deviceProperties.limits.maxFramebufferWidth, deviceProperties.limits.maxFramebufferHeight});
  mZSupported = (bool)(depth32FormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  mStencilSupported = (bool)(depth64FormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  bool mailboxSupported = std::find(mSwapChainDetails.presentModes.begin(), mSwapChainDetails.presentModes.end(), vk::PresentModeKHR::eMailbox) != mSwapChainDetails.presentModes.end();
  if (mDisplay->GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Max MSAA: %d, 32 bit Z buffer %d, 32 bit Z buffer + stencil buffer %d, Mailbox present mode %d, FIFO latest ready present mode %d, smooth lines %d\n", (int32_t)mMaxMSAAsupported, (int32_t)mZSupported, (int32_t)mStencilSupported, (int32_t)mailboxSupported, (int32_t)mFifoLatestReadySupported, (int32_t)mSmoothLinesSupported);
  }

  vk::DeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.queueFamilyIndex = mGraphicsFamily;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;
  vk::DeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pNext = featureChain;
  deviceCreateInfo.pEnabledFeatures = nullptr;
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
  mDevice = mPhysicalDevice.createDevice(deviceCreateInfo, nullptr);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice);
  mGraphicsQueue = mDevice.getQueue(mGraphicsFamily, 0);

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = mGraphicsFamily;
  mCommandPool = mDevice.createCommandPool(poolInfo, nullptr);
  mPipelineCache = mDevice.createPipelineCache(vk::PipelineCacheCreateInfo{}, nullptr);
}

void GPUDisplayBackendVulkan::clearDevice()
{
  mDevice.destroyCommandPool(mCommandPool, nullptr);
  mDevice.destroyPipelineCache(mPipelineCache, nullptr);
  mDevice.destroy(nullptr);
  mInstance.destroySurfaceKHR(mSurface, nullptr);
  if (mEnableValidationLayers) {
    mInstance.destroyDebugUtilsMessengerEXT(mDebugMessenger, nullptr);
  }
}

// ---------------------------- VULKAN COMMAND BUFFERS ----------------------------

void GPUDisplayBackendVulkan::createCommandBuffers()
{
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = mCommandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = mImageCount;
  mCommandBufferUpToDate.resize(mImageCount, false);
  mCommandBuffers = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersText = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersTexture = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersDownsample = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersMix = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersPresent = mDevice.allocateCommandBuffers(allocInfo);
}

void GPUDisplayBackendVulkan::clearCommandBuffers()
{
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffers.size(), mCommandBuffers.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersText.size(), mCommandBuffersText.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersTexture.size(), mCommandBuffersTexture.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersDownsample.size(), mCommandBuffersDownsample.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersMix.size(), mCommandBuffersMix.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersPresent.size(), mCommandBuffersPresent.data());
}

// ---------------------------- VULKAN SEMAPHORES AND FENCES ----------------------------

void GPUDisplayBackendVulkan::createSemaphores()
{
  vk::SemaphoreCreateInfo semaphoreInfo{};
  mImageAvailableSemaphore.resize(mFramesInFlight);
  mPresentSemaphore.resize(mImageCount);
  for (uint32_t i = 0; i < mFramesInFlight; i++) {
    mImageAvailableSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
  }
  for (uint32_t i = 0; i < mImageCount; i++) {
    mPresentSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
  }
  vk::SemaphoreTypeCreateInfo timelineTypeInfo{vk::SemaphoreType::eTimeline, 0};
  semaphoreInfo.pNext = &timelineTypeInfo;
  mTimelineSemaphore = mDevice.createSemaphore(semaphoreInfo, nullptr);
  mTimelineValue = 0;
  mFrameTimelineValue.assign(mFramesInFlight, 0);
  mImageTimelineValue.assign(mImageCount, 0);
  for (uint32_t i = 0; i < mImageCount; i++) {
    mPresentFences.emplace_back(mDevice.createFence(vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled}, nullptr));
  }
}

void GPUDisplayBackendVulkan::waitPresentFences()
{
  if (mPresentFences.size()) { // Timeout in case a failed present did not signal its fence
    vk::Result retVal = mDevice.waitForFences(mPresentFences.size(), mPresentFences.data(), true, 1'000'000'000);
    if (retVal != vk::Result::eTimeout) {
      CHKERR(retVal);
    }
  }
}

void GPUDisplayBackendVulkan::clearSemaphores()
{
  clearVector(mImageAvailableSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mPresentSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  mDevice.destroySemaphore(mTimelineSemaphore, nullptr);
  clearVector(mPresentFences, [&](auto& x) { mDevice.destroyFence(x, nullptr); });
}

// ---------------------------- VULKAN UNIFORM LAYOUTS AND BUFFERS  ----------------------------

void GPUDisplayBackendVulkan::createUniformLayoutsAndBuffers()
{
  for (int32_t j = 0; j < 3; j++) {
    mUniformBuffersMat[j].resize(mImageCount);
    for (uint32_t i = 0; i < mImageCount; i++) {
      mUniformBuffersMat[j][i] = createBuffer(sizeof(hmm_mat4), nullptr, vk::BufferUsageFlagBits::eUniformBuffer, mDisplay->cfg().vulkan.uniformBuffersInDeviceMemory ? 2 : 0);
    }
  }

  vk::DescriptorSetLayoutBinding uboLayoutBindingMat{};
  uboLayoutBindingMat.binding = 0;
  uboLayoutBindingMat.descriptorType = vk::DescriptorType::eUniformBuffer;
  uboLayoutBindingMat.descriptorCount = 1;
  uboLayoutBindingMat.stageFlags = vk::ShaderStageFlagBits::eVertex;
  vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 2;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
  samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
  vk::DescriptorSetLayoutBinding bindings[2] = {uboLayoutBindingMat, samplerLayoutBinding};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = bindings;
  mUniformDescriptor = mDevice.createDescriptorSetLayout(layoutInfo, nullptr);
  layoutInfo.bindingCount = 2;
  mUniformDescriptorTexture = mDevice.createDescriptorSetLayout(layoutInfo, nullptr);

  mUpdateUBOdrawMatrix = mUpdateUBOtextMatrix = mUpdateUBOmixMatrix = -1;
}

void GPUDisplayBackendVulkan::clearUniformLayoutsAndBuffers()
{
  mDevice.destroyDescriptorSetLayout(mUniformDescriptor, nullptr);
  mDevice.destroyDescriptorSetLayout(mUniformDescriptorTexture, nullptr);
  for (int32_t j = 0; j < 3; j++) {
    clearVector(mUniformBuffersMat[j], [&](auto& x) { clearBuffer(x); });
  }
}

void GPUDisplayBackendVulkan::cmdPushDescriptors(vk::CommandBuffer commandBuffer, vk::PipelineLayout layout, const VulkanBuffer& ubo, vk::ImageView textureView)
{
  vk::DescriptorBufferInfo bufferInfo{ubo.buffer, 0, ubo.size};
  vk::DescriptorImageInfo imageInfo{mTextureSampler, textureView, vk::ImageLayout::eShaderReadOnlyOptimal};
  const vk::WriteDescriptorSet writes[2] = {{{}, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo}, {{}, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo}};
  commandBuffer.pushDescriptorSet(vk::PipelineBindPoint::eGraphics, layout, 0, textureView ? 2 : 1, writes);
}

// ---------------------------- VULKAN TEXTURE SAMPLER ----------------------------

void GPUDisplayBackendVulkan::createTextureSampler()
{
  vk::SamplerCreateInfo samplerInfo{};
  samplerInfo.magFilter = vk::Filter::eLinear;
  samplerInfo.minFilter = vk::Filter::eLinear;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
  samplerInfo.compareEnable = false;
  samplerInfo.compareOp = vk::CompareOp::eAlways;
  samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
  samplerInfo.unnormalizedCoordinates = false;
  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;
  mTextureSampler = mDevice.createSampler(samplerInfo, nullptr);
}

void GPUDisplayBackendVulkan::clearTextureSampler()
{
  mDevice.destroySampler(mTextureSampler, nullptr);
}

// ---------------------------- VULKAN SWAPCHAIN MANAGEMENT ----------------------------

void GPUDisplayBackendVulkan::createSwapChain(vk::SwapchainKHR oldSwapChain)
{
  updateSwapChainDetails(mPhysicalDevice);
  mSurfaceFormat = chooseSwapSurfaceFormat(mSwapChainDetails.formats);
  mPresentMode = chooseSwapPresentMode();
  if (mDisplay->GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Vulkan present mode %s (VSync %d)", vk::to_string(mPresentMode).c_str(), (int32_t)mDisplay->cfgR().drawQualityVSync);
  }
  vk::Extent2D extent = chooseSwapExtent(mSwapChainDetails.capabilities);

  uint32_t imageCount = std::max<uint32_t>(mSwapChainDetails.capabilities.minImageCount + 1, mDisplay->cfg().vulkan.nFramesInFlight + 1);
  if (mSwapChainDetails.capabilities.maxImageCount > 0 && imageCount > mSwapChainDetails.capabilities.maxImageCount) {
    imageCount = mSwapChainDetails.capabilities.maxImageCount;
  }

  mScreenWidth = extent.width;
  mScreenHeight = extent.height;

  vk::SwapchainCreateInfoKHR swapCreateInfo{};
  swapCreateInfo.surface = mSurface;
  swapCreateInfo.minImageCount = imageCount;
  swapCreateInfo.imageFormat = mSurfaceFormat.format;
  swapCreateInfo.imageColorSpace = mSurfaceFormat.colorSpace;
  swapCreateInfo.imageExtent = extent;
  swapCreateInfo.imageArrayLayers = 1;
  swapCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst; // Transfer for screenshots / FSAA downsampling, so that they need no swapchain recreation
  swapCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
  swapCreateInfo.queueFamilyIndexCount = 0;     // Optional
  swapCreateInfo.pQueueFamilyIndices = nullptr; // Optional
  swapCreateInfo.preTransform = mSwapChainDetails.capabilities.currentTransform;
  swapCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  swapCreateInfo.presentMode = mPresentMode;
  swapCreateInfo.clipped = true;
  swapCreateInfo.oldSwapchain = oldSwapChain;
  vk::SwapchainPresentScalingCreateInfoKHR scalingInfo{};
  { // Let the compositor scale the last image while resizing, until the new swapchain is presented
    const vk::SurfacePresentModeKHR surfacePresentMode{mPresentMode};
    const auto caps = mPhysicalDevice.getSurfaceCapabilities2KHR<vk::SurfaceCapabilities2KHR, vk::SurfacePresentScalingCapabilitiesKHR>(vk::PhysicalDeviceSurfaceInfo2KHR{mSurface, &surfacePresentMode}).get<vk::SurfacePresentScalingCapabilitiesKHR>();
    auto gravity = [](vk::PresentGravityFlagsKHR f) { return (f & vk::PresentGravityFlagBitsKHR::eCentered) ? vk::PresentGravityFlagBitsKHR::eCentered : (f & vk::PresentGravityFlagBitsKHR::eMin) ? vk::PresentGravityFlagBitsKHR::eMin
                                                                                                                                                                                                   : vk::PresentGravityFlagsKHR(f & vk::PresentGravityFlagBitsKHR::eMax); };
    scalingInfo.scalingBehavior = (caps.supportedPresentScaling & vk::PresentScalingFlagBitsKHR::eAspectRatioStretch) ? vk::PresentScalingFlagBitsKHR::eAspectRatioStretch : vk::PresentScalingFlagsKHR(caps.supportedPresentScaling & vk::PresentScalingFlagBitsKHR::eStretch);
    scalingInfo.presentGravityX = gravity(caps.supportedPresentGravityX);
    scalingInfo.presentGravityY = gravity(caps.supportedPresentGravityY);
    if (scalingInfo.scalingBehavior && scalingInfo.presentGravityX && scalingInfo.presentGravityY) {
      swapCreateInfo.pNext = &scalingInfo;
    }
  }
  mSwapChain = mDevice.createSwapchainKHR(swapCreateInfo, nullptr);
  mLastPresentId = 0; // Present IDs are per swapchain
  if (oldSwapChain) {
    waitPresentFences(); // Presentation engine done with the old swapchain
    mDevice.destroySwapchainKHR(oldSwapChain, nullptr);
  }

  mSwapChainImages = mDevice.getSwapchainImagesKHR(mSwapChain);
  uint32_t oldFramesInFlight = mFramesInFlight, oldImageCount = mImageCount;
  mFramesInFlight = mDisplay->cfg().vulkan.nFramesInFlight;
  mImageCount = mSwapChainImages.size();
  if (mImageCount > sizeof(mUpdateUBOdrawMatrix) * 8) {
    throw std::runtime_error("Too many swapchain images");
  }

  if (mFramesInFlight != oldFramesInFlight || mImageCount != oldImageCount || !mCommandInfrastructureCreated) {
    if (mCommandInfrastructureCreated) {
      clearUniformLayoutsAndBuffers(); // TODO: Check what must actually be cleaned up if frames in flight changes
      clearCommandBuffers();
      clearSemaphores();
    }
    createUniformLayoutsAndBuffers();
    createCommandBuffers();
    createSemaphores();
    mCommandInfrastructureCreated = true;
  }

  mSwapChainImageViews.resize(mImageCount);
  for (uint32_t i = 0; i < mImageCount; i++) {
    mSwapChainImageViews[i] = createImageViewI(mDevice, mSwapChainImages[i], mSurfaceFormat.format);
    mCommandBuffersPresent[i].reset({});
    mCommandBuffersPresent[i].begin(vk::CommandBufferBeginInfo{});
    cmdImageMemoryBarrier(mCommandBuffersPresent[i], mSwapChainImages[i], vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
    mCommandBuffersPresent[i].end();
  }
}

void GPUDisplayBackendVulkan::clearSwapChain(bool keepSwapChain)
{
  clearVector(mSwapChainImageViews, [&](auto& x) { mDevice.destroyImageView(x, nullptr); });
  if (!keepSwapChain) {
    waitPresentFences();
    mDevice.destroySwapchainKHR(mSwapChain, nullptr);
  }
}

void GPUDisplayBackendVulkan::recreateRendering(bool forScreenshot, bool forMixing)
{
  needRecordCommandBuffers();
  const bool needUpdateSwapChain = mMustUpdateSwapChain;
  const bool needUpdateOffscreenBuffers = needUpdateSwapChain || mDownsampleFactor != getDownsampleFactor(forScreenshot) || mMSAASampleCount != getMSAASamplesFlag(std::min<uint32_t>(mMaxMSAAsupported, mDisplay->cfgR().drawQualityMSAA)) || mZActive != (mZSupported && mDisplay->cfgL().depthBuffer) || mMixingSupported != forMixing;
  const bool smoothLinesChanged = mSmoothLines != useNativeSmoothLines();
  if (!needUpdateOffscreenBuffers && !smoothLinesChanged) {
    return;
  }
  mDevice.waitIdle();
  const vk::Format oldFormat = mSurfaceFormat.format;
  const vk::SampleCountFlagBits oldMSAASampleCount = mMSAASampleCount;
  const bool oldZActive = mZActive;
  if (needUpdateOffscreenBuffers) {
    clearOffscreenBuffers();
    if (needUpdateSwapChain) {
      clearSwapChain(true);
      createSwapChain(mSwapChain);
    }
    createOffscreenBuffers(forScreenshot, forMixing);
    mUpdateUBOtextMatrix = mUpdateUBOmixMatrix = -1; // Screen / render size may have changed
  }
  if (mSurfaceFormat.format != oldFormat || mMSAASampleCount != oldMSAASampleCount || mZActive != oldZActive || smoothLinesChanged) { // Viewport / scissor are dynamic
    clearPipeline();
    createPipeline();
  }
}

// ---------------------------- VULKAN OFFSCREEN BUFFERS ----------------------------

void GPUDisplayBackendVulkan::createOffscreenBuffers(bool forScreenshot, bool forMixing)
{
  mMSAASampleCount = getMSAASamplesFlag(std::min<uint32_t>(mMaxMSAAsupported, mDisplay->cfgR().drawQualityMSAA));
  mZActive = mZSupported && mDisplay->cfgL().depthBuffer;
  mMixingSupported = forMixing;
  mDownsampleFactor = getDownsampleFactor(forScreenshot);
  mDownsampleFSAA = mDownsampleFactor != 1;
  mRenderWidth = mScreenWidth * mDownsampleFactor;
  mRenderHeight = mScreenHeight * mDownsampleFactor;

  const uint32_t imageCountWithMixImages = mImageCount * (mMixingSupported ? 2 : 1);
  mRenderTargetView.resize(imageCountWithMixImages);
  mRenderTargetImage.resize(imageCountWithMixImages);
  if (mDownsampleFSAA) {
    mDownsampleImages.resize(imageCountWithMixImages);
  }
  if (mMSAASampleCount != vk::SampleCountFlagBits::e1) {
    mMSAAImages.resize(imageCountWithMixImages);
  }
  if (mZActive) {
    mZImages.resize(imageCountWithMixImages);
  }
  if (mMixingSupported && !mDownsampleFSAA) {
    mMixImages.resize(mImageCount);
  }

  for (uint32_t i = 0; i < imageCountWithMixImages; i++) {
    if (i < mImageCount) { // Main render chain
      // primary buffer mSwapChainImageViews[i] created as part of createSwapChain, not here
    } else if (!mDownsampleFSAA) { // for rendering to mixBuffer
      createImageI(mDevice, mPhysicalDevice, mMixImages[i - mImageCount].image, mMixImages[i - mImageCount].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
      mMixImages[i - mImageCount].view = createImageViewI(mDevice, mMixImages[i - mImageCount].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
    }
    if (mDownsampleFSAA) {
      vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
      if (i >= mImageCount || mDownsampleFactor != 2) {
        usage |= vk::ImageUsageFlagBits::eSampled; // Mixing, shader downsampling
      }
      if (i < mImageCount && (mDownsampleFactor == 2 || forScreenshot)) {
        usage |= vk::ImageUsageFlagBits::eTransferSrc; // Blit downsampling, scaled screenshots (blit or copy for screenshot factor <= 2)
      }
      createImageI(mDevice, mPhysicalDevice, mDownsampleImages[i].image, mDownsampleImages[i].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
      mDownsampleImages[i].view = createImageViewI(mDevice, mDownsampleImages[i].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
      mRenderTargetView[i] = &mDownsampleImages[i].view;
      mRenderTargetImage[i] = mDownsampleImages[i].image;
    } else {
      mRenderTargetView[i] = i < mImageCount ? &mSwapChainImageViews[i] : &mMixImages[i - mImageCount].view;
      mRenderTargetImage[i] = i < mImageCount ? mSwapChainImages[i] : mMixImages[i - mImageCount].image;
    }
    if (mMSAASampleCount != vk::SampleCountFlagBits::e1) {
      createImageI(mDevice, mPhysicalDevice, mMSAAImages[i].image, mMSAAImages[i].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, mMSAASampleCount);
      mMSAAImages[i].view = createImageViewI(mDevice, mMSAAImages[i].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
    }
    if (mZActive) {
      createImageI(mDevice, mPhysicalDevice, mZImages[i].image, mZImages[i].memory, mRenderWidth, mRenderHeight, vk::Format::eD32Sfloat, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, mMSAASampleCount);
      mZImages[i].view = createImageViewI(mDevice, mZImages[i].image, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, 1);
    }
  }

  if (mMixingSupported) {
    float vertices[6][4] = {
      {0, (float)mRenderHeight, 0.0f, 1.0f},
      {0, 0, 0.0f, 0.0f},
      {(float)mRenderWidth, 0, 1.0f, 0.0f},
      {0, (float)mRenderHeight, 0.0f, 1.0f},
      {(float)mRenderWidth, 0, 1.0f, 0.0f},
      {(float)mRenderWidth, (float)mRenderHeight, 1.0f, 1.0f}};
    mMixingTextureVertexArray = createBuffer(sizeof(vertices), &vertices[0][0], vk::BufferUsageFlagBits::eVertexBuffer, 1);
  }

  mFontVertexBuffer.resize(mImageCount);
}

void GPUDisplayBackendVulkan::clearOffscreenBuffers()
{
  clearVector(mMSAAImages, [&](auto& x) { clearImage(x); });
  clearVector(mDownsampleImages, [&](auto& x) { clearImage(x); });
  clearVector(mZImages, [&](auto& x) { clearImage(x); });
  clearVector(mMixImages, [&](auto& x) { clearImage(x); });
  clearVector(mFontVertexBuffer, [&](auto& x) { if (x.size) { clearBuffer(x); } });
  if (mMixingSupported) {
    clearBuffer(mMixingTextureVertexArray);
  }
}

// ---------------------------- VULKAN PIPELINE ----------------------------

void GPUDisplayBackendVulkan::createPipeline()
{
  vk::PipelineShaderStageCreateInfo shaderStages[2] = {vk::PipelineShaderStageCreateInfo{}, vk::PipelineShaderStageCreateInfo{}};
  vk::PipelineShaderStageCreateInfo& vertShaderStageInfo = shaderStages[0];
  vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
  // vertShaderStageInfo.pNext // below: SPIR-V
  vertShaderStageInfo.pName = "main";
  vk::PipelineShaderStageCreateInfo& fragShaderStageInfo = shaderStages[1];
  fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
  // fragShaderStageInfo.pNext // below: SPIR-V
  fragShaderStageInfo.pName = "main";

  vk::VertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  // bindingDescription.stride // below
  bindingDescription.inputRate = vk::VertexInputRate::eVertex;

  vk::VertexInputAttributeDescription attributeDescriptions{};
  attributeDescriptions.binding = 0;
  attributeDescriptions.location = 0;
  // attributeDescriptions.format // below
  attributeDescriptions.offset = 0;

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = 1;
  vertexInputInfo.pVertexAttributeDescriptions = &attributeDescriptions;
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
  // inputAssembly.topology // below
  inputAssembly.primitiveRestartEnable = false;

  vk::PipelineViewportStateCreateInfo viewportState{}; // Viewport and scissor are dynamic
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  vk::PipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.depthClampEnable = false;
  rasterizer.rasterizerDiscardEnable = false;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.lineWidth = mDisplay->cfgL().lineWidth;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eClockwise;
  rasterizer.depthBiasEnable = false;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f;          // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

  vk::PipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sampleShadingEnable = false;
  // multisampling.rasterizationSamples // below
  multisampling.minSampleShading = 1.0f;       // Optional
  multisampling.pSampleMask = nullptr;         // Optional
  multisampling.alphaToCoverageEnable = false; // Optional
  multisampling.alphaToOneEnable = false;      // Optional

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  // colorBlendAttachment.blendEnable // below
  colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
  colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
  colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
  colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
  colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
  colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

  vk::PipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.logicOpEnable = false;
  colorBlending.logicOp = vk::LogicOp::eCopy;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  vk::PipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.depthTestEnable = true;
  depthStencil.depthWriteEnable = true;
  depthStencil.depthCompareOp = vk::CompareOp::eLess;
  depthStencil.depthBoundsTestEnable = false;
  depthStencil.stencilTestEnable = false;

  vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eLineWidth};
  vk::PipelineDynamicStateCreateInfo dynamicState{};
  // dynamicState.dynamicStateCount // below: viewport + scissor for text / texture, + line width for the main pipelines
  dynamicState.pDynamicStates = dynamicStates;

  vk::PushConstantRange pushConstantRanges[2] = {vk::PushConstantRange{}, vk::PushConstantRange{}};
  pushConstantRanges[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
  pushConstantRanges[0].offset = 0;
  pushConstantRanges[0].size = sizeof(float) * 4;
  pushConstantRanges[1].stageFlags = vk::ShaderStageFlagBits::eVertex;
  pushConstantRanges[1].offset = pushConstantRanges[0].size;
  pushConstantRanges[1].size = 4 * sizeof(float); // Point size, line width, viewport
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptor;
  pipelineLayoutInfo.pushConstantRangeCount = 2;
  pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges;
  mPipelineLayout = mDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptorTexture;
  mPipelineLayoutTexture = mDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);

  vk::GraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  // pipelineInfo.pDepthStencilState // below
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  // pipelineInfo.layout // below
  vk::PipelineRenderingCreateInfo renderingInfo{0, 1, &mSurfaceFormat.format};
  pipelineInfo.pNext = &renderingInfo;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.basePipelineHandle = VkPipeline(VK_NULL_HANDLE); // Optional
  pipelineInfo.basePipelineIndex = -1;                          // Optional

  mSmoothLines = useNativeSmoothLines();
  vk::PipelineRasterizationLineStateCreateInfo smoothLineState{};
  smoothLineState.lineRasterizationMode = vk::LineRasterizationMode::eRectangularSmooth; // Coverage goes to alpha, blending is enabled

  mPipelines.resize(9); // 0 points, 1 lines, 2 line strip (drawType index), 3 text, 4 texture, 5-7 smooth (shader-based) points, lines, line strip, 8 FSAA downsampling
  static constexpr vk::PrimitiveTopology types[3] = {vk::PrimitiveTopology::ePointList, vk::PrimitiveTopology::eLineList, vk::PrimitiveTopology::eLineStrip};
  for (uint32_t i = 0; i < mPipelines.size(); i++) {
    if (i == 8) { // FSAA downsampling, full-screen triangle without vertex input (last pipeline, the changed states are not reset)
      rasterizer.pNext = nullptr;
      rasterizer.cullMode = vk::CullModeFlagBits::eNone;
      vertexInputInfo.vertexBindingDescriptionCount = vertexInputInfo.vertexAttributeDescriptionCount = 0;
      inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
      vertShaderStageInfo.pNext = &mShaders.at("vertexFullscreen");
      fragShaderStageInfo.pNext = &mShaders.at("fragmentDownsample");
      dynamicState.dynamicStateCount = 2;
      pipelineInfo.layout = mPipelineLayoutTexture;
      renderingInfo.depthAttachmentFormat = vk::Format::eUndefined;
      pipelineInfo.pDepthStencilState = nullptr;
      colorBlendAttachment.blendEnable = false;
      multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    } else if (i == 4) { // Texture rendering
      rasterizer.pNext = nullptr;
      bindingDescription.stride = 4 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32A32Sfloat;
      inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
      vertShaderStageInfo.pNext = &mShaders.at("vertexTexture");
      fragShaderStageInfo.pNext = &mShaders.at("fragmentTexture");
      dynamicState.dynamicStateCount = 2;
      pipelineInfo.layout = mPipelineLayoutTexture;
      renderingInfo.depthAttachmentFormat = vk::Format::eUndefined;
      pipelineInfo.pDepthStencilState = nullptr;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    } else if (i == 3) { // Text rendering
      rasterizer.pNext = nullptr;
      bindingDescription.stride = 4 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32A32Sfloat;
      inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
      vertShaderStageInfo.pNext = &mShaders.at("vertexTexture");
      fragShaderStageInfo.pNext = &mShaders.at("fragmentText");
      dynamicState.dynamicStateCount = 2;
      pipelineInfo.layout = mPipelineLayoutTexture;
      renderingInfo.depthAttachmentFormat = vk::Format::eUndefined;
      pipelineInfo.pDepthStencilState = nullptr;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    } else { // Point / line / line-strip rendering
      bindingDescription.stride = 3 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32Sfloat;
      const uint32_t drawIdx = i >= 5 ? i - 5 : i;
      inputAssembly.topology = types[drawIdx];
      rasterizer.pNext = mSmoothLines && (i == 1 || i == 2) ? &smoothLineState : nullptr;
      vertShaderStageInfo.pNext = &mShaders.at(i < 5 ? (drawIdx == 0 ? "vertexPoint" : "vertex") : (drawIdx == 0 ? "vertexSmoothPoint" : "vertexSmoothLine"));
      fragShaderStageInfo.pNext = &mShaders.at(i < 5 ? "fragment" : (drawIdx == 0 ? "fragmentSmoothPoint" : "fragmentSmoothLine"));
      pipelineInfo.layout = mPipelineLayout;
      renderingInfo.depthAttachmentFormat = mZActive ? vk::Format::eD32Sfloat : vk::Format::eUndefined;
      pipelineInfo.pDepthStencilState = mZActive ? &depthStencil : nullptr;
      dynamicState.dynamicStateCount = 3;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = mMSAASampleCount;
    }

    CHKERR(mDevice.createGraphicsPipelines(mPipelineCache, 1, &pipelineInfo, nullptr, &mPipelines[i]));
  }
}

void GPUDisplayBackendVulkan::startFillCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex, bool toMixBuffer)
{
  commandBuffer.reset({});

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = {};
  commandBuffer.begin(beginInfo);

  const uint32_t idx = toMixBuffer ? imageIndex + mImageCount : imageIndex;
  const bool msaa = mMSAASampleCount != vk::SampleCountFlagBits::e1;
  const vk::PipelineStageFlags2 prevStages = vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eFragmentShader; // Previous use of the render target (rendering, blit, mix texture), and the wait stage of the acquire semaphore
  cmdImageMemoryBarrier(commandBuffer, mRenderTargetImage[idx], prevStages, vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
  if (msaa) {
    cmdImageMemoryBarrier(commandBuffer, mMSAAImages[idx].image, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
  }
  if (mZActive) {
    const vk::PipelineStageFlags2 depthStages = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
    cmdImageMemoryBarrier(commandBuffer, mZImages[idx].image, depthStages, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, depthStages, vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);
  }

  vk::RenderingAttachmentInfo colorAttachment{msaa ? mMSAAImages[idx].view : *mRenderTargetView[idx], vk::ImageLayout::eColorAttachmentOptimal};
  colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
  colorAttachment.storeOp = msaa ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore;
  colorAttachment.clearValue.color = mDisplay->cfgL().invertColors ? vk::ClearColorValue{std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}} : vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
  if (msaa) {
    colorAttachment.resolveMode = vk::ResolveModeFlagBits::eAverage;
    colorAttachment.resolveImageView = *mRenderTargetView[idx];
    colorAttachment.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  }
  vk::RenderingAttachmentInfo depthAttachment{mZActive ? mZImages[idx].view : vk::ImageView{}, vk::ImageLayout::eDepthStencilAttachmentOptimal};
  depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
  depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
  depthAttachment.clearValue.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
  cmdBeginRendering(commandBuffer, mRenderWidth, mRenderHeight, colorAttachment, mZActive ? &depthAttachment : nullptr);

  vk::DeviceSize offsets[] = {0};
  if (mVBO.size) {
    commandBuffer.bindVertexBuffers(0, 1, &mVBO.buffer, offsets);
  }
  cmdPushDescriptors(commandBuffer, mPipelineLayout, mUniformBuffersMat[0][imageIndex]);
  const float viewport[2] = {(float)mRenderWidth, (float)mRenderHeight};
  commandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eVertex, 6 * sizeof(float), sizeof(viewport), viewport);
}

void GPUDisplayBackendVulkan::endFillCommandBuffer(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.endRendering();
  commandBuffer.end();
}

void GPUDisplayBackendVulkan::clearPipeline()
{
  clearVector(mPipelines, [&](auto& x) { mDevice.destroyPipeline(x, nullptr); });
  mDevice.destroyPipelineLayout(mPipelineLayout, nullptr);
  mDevice.destroyPipelineLayout(mPipelineLayoutTexture, nullptr);
}

// ---------------------------- VULKAN SHADERS ----------------------------

#define LOAD_SHADER(file, ext) \
  mShaders[#file] = vk::ShaderModuleCreateInfo { {}, _binary_vulkan_##file##_##ext##_spv_len, reinterpret_cast<const uint32_t*>(_binary_vulkan_##file##_##ext##_spv_start) }

void GPUDisplayBackendVulkan::loadShaders()
{
  LOAD_SHADER(vertex, vert);
  LOAD_SHADER(fragment, frag);
  LOAD_SHADER(fragmentSmoothPoint, frag);
  LOAD_SHADER(vertexSmoothLine, vert);
  LOAD_SHADER(vertexSmoothPoint, vert);
  LOAD_SHADER(fragmentSmoothLine, frag);
  LOAD_SHADER(vertexPoint, vert);
  LOAD_SHADER(vertexTexture, vert);
  LOAD_SHADER(fragmentTexture, frag);
  LOAD_SHADER(fragmentText, frag);
  LOAD_SHADER(vertexFullscreen, vert);
  LOAD_SHADER(fragmentDownsample, frag);
}

// ---------------------------- VULKAN BUFFERS ----------------------------

void GPUDisplayBackendVulkan::writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData)
{
  if (buffer.mapped) {
    memcpy(buffer.mapped, srcData, size);
  } else {
    auto tmp = createBuffer(size, srcData, vk::BufferUsageFlagBits::eTransferSrc, 0);

    vk::CommandBuffer commandBuffer = getSingleTimeCommandBuffer();
    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    commandBuffer.copyBuffer(tmp.buffer, buffer.buffer, 1, &copyRegion);
    submitSingleTimeCommandBuffer(commandBuffer);

    clearBuffer(tmp);
  }
}

GPUDisplayBackendVulkan::VulkanBuffer GPUDisplayBackendVulkan::createBuffer(size_t size, const void* srcData, vk::BufferUsageFlags type, int32_t deviceMemory)
{
  vk::MemoryPropertyFlags properties;
  if (deviceMemory) {
    properties |= vk::MemoryPropertyFlagBits::eDeviceLocal;
  }
  if (deviceMemory == 0 || deviceMemory == 2) {
    properties |= (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  if (deviceMemory == 1) {
    type |= vk::BufferUsageFlagBits::eTransferDst;
  }

  VulkanBuffer buffer;
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = type;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;
  buffer.buffer = mDevice.createBuffer(bufferInfo, nullptr);

  vk::MemoryRequirements memRequirements;
  memRequirements = mDevice.getBufferMemoryRequirements(buffer.buffer);
  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, mPhysicalDevice);
  buffer.memory = mDevice.allocateMemory(allocInfo, nullptr);

  mDevice.bindBufferMemory(buffer.buffer, buffer.memory, 0);

  buffer.size = size;
  buffer.deviceMemory = deviceMemory;
  if (deviceMemory != 1) {
    CHKERR(mDevice.mapMemory(buffer.memory, 0, VK_WHOLE_SIZE, {}, &buffer.mapped));
  }

  if (srcData != nullptr) {
    writeToBuffer(buffer, size, srcData);
  }

  return buffer;
}

void GPUDisplayBackendVulkan::clearBuffer(VulkanBuffer& buffer)
{
  mDevice.destroyBuffer(buffer.buffer, nullptr);
  mDevice.freeMemory(buffer.memory, nullptr); // Implicitly unmaps
}

void GPUDisplayBackendVulkan::clearVertexBuffers()
{
  if (mVBO.size) {
    clearBuffer(mVBO);
    mVBO.size = 0;
  }
  if (mIndirectCommandBuffer.size) {
    clearBuffer(mIndirectCommandBuffer);
    mIndirectCommandBuffer.size = 0;
  }
}

// ---------------------------- VULKAN TEXTURES ----------------------------

void GPUDisplayBackendVulkan::writeToImage(VulkanImage& image, const void* srcData, size_t srcSize)
{
  auto tmp = createBuffer(srcSize, srcData, vk::BufferUsageFlagBits::eTransferSrc, 0);

  vk::CommandBuffer commandBuffer = getSingleTimeCommandBuffer();
  cmdImageMemoryBarrier(commandBuffer, image.image, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
  vk::BufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = vk::Offset3D{0, 0, 0};
  region.imageExtent = vk::Extent3D{image.sizex, image.sizey, 1};
  commandBuffer.copyBufferToImage(tmp.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
  cmdImageMemoryBarrier(commandBuffer, image.image, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  submitSingleTimeCommandBuffer(commandBuffer);

  clearBuffer(tmp);
}

GPUDisplayBackendVulkan::VulkanImage GPUDisplayBackendVulkan::createImage(uint32_t sizex, uint32_t sizey, const void* srcData, size_t srcSize, vk::Format format)
{
  VulkanImage image;
  createImageI(mDevice, mPhysicalDevice, image.image, image.memory, sizex, sizey, format, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, vk::SampleCountFlagBits::e1);

  image.view = createImageViewI(mDevice, image.image, format);

  image.sizex = sizex;
  image.sizey = sizey;
  image.format = format;

  if (srcData) {
    writeToImage(image, srcData, srcSize);
  }
  return image;
}

void GPUDisplayBackendVulkan::clearImage(VulkanImage& image)
{
  mDevice.destroyImageView(image.view, nullptr);
  mDevice.destroyImage(image.image, nullptr);
  mDevice.freeMemory(image.memory, nullptr);
}

// ---------------------------- VULKAN INIT EXIT ----------------------------

int32_t GPUDisplayBackendVulkan::InitBackendA()
{
  mEnableValidationLayers = (mDisplay->param() && mDisplay->GetProcessingSettings().debugLevel >= 2) || (getenv("GPUCA_VULKAN_VALIDATION") && atoi(getenv("GPUCA_VULKAN_VALIDATION")));
  mFramesInFlight = mDisplay->cfg().vulkan.nFramesInFlight;

  createDevice();
  loadShaders();
  createTextureSampler();
  createSwapChain();
  createOffscreenBuffers();
  createPipeline();

  return (0);
}

void GPUDisplayBackendVulkan::ExitBackendA()
{
  mDevice.waitIdle();
  if (mFontImage.sizex && mFontImage.sizey) {
    clearImage(mFontImage);
    mFontImage.sizex = mFontImage.sizey = 0;
  }
  clearVertexBuffers();
  clearPipeline();
  clearOffscreenBuffers();
  clearSwapChain();
  if (mCommandInfrastructureCreated) {
    clearSemaphores();
    clearCommandBuffers();
    clearUniformLayoutsAndBuffers();
  }
  clearTextureSampler();
  clearDevice();
  mCommandInfrastructureCreated = false;
  mImageCount = 0;
  mLastPresentId = 0;
  mSwapChainSuboptimal = 0;
  mFrameSkipped = false;
}

// ---------------------------- USER CODE ----------------------------

void GPUDisplayBackendVulkan::resizeScene(uint32_t width, uint32_t height)
{
  if (mScreenWidth == width && mScreenHeight == height) {
    return;
  }
  updateSwapChainDetails(mPhysicalDevice);
  vk::Extent2D extent = chooseSwapExtent(mSwapChainDetails.capabilities);
  if (extent.width != mScreenWidth || extent.height != mScreenHeight) {
    mMustUpdateSwapChain = true;
  }
}

void GPUDisplayBackendVulkan::loadDataToGPU(size_t totalVertizes)
{
  mDevice.waitIdle();
  clearVertexBuffers();
  if (totalVertizes) { // Zero-size buffers are invalid (empty event)
    mVBO = createBuffer(totalVertizes * sizeof(mDisplay->vertexBuffer()[0][0]), mDisplay->vertexBuffer()[0].data(), vk::BufferUsageFlagBits::eVertexBuffer, 1);
  }
  if (mDisplay->cfgR().useGLIndirectDraw) {
    fillIndirectCmdBuffer();
    if (mCmdBuffer.size()) {
      mIndirectCommandBuffer = createBuffer(mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), vk::BufferUsageFlagBits::eIndirectBuffer, 1);
    }
    mCmdBuffer.clear();
  }
  needRecordCommandBuffers();
}

uint32_t GPUDisplayBackendVulkan::drawVertices(const vboList& v, const drawType tt)
{
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSector = std::get<2>(v);
  if (count == 0) {
    return 0;
  }
  if (mCommandBufferUpToDate[mCurrentImageIndex]) {
    return count;
  }

  const int32_t pipeline = tt + ((tt == POINTS ? mDisplay->cfgL().smoothPoints : useSmoothLineShader()) ? 5 : 0);
  if (mCurrentCommandBufferLastPipeline != pipeline) {
    mCurrentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[pipeline]);
    mCurrentCommandBufferLastPipeline = pipeline;
  }
  if (mDisplay->cfgR().useGLIndirectDraw) {
    mCurrentCommandBuffer.drawIndirect(mIndirectCommandBuffer.buffer, (mIndirectSectorOffset[iSector] + first) * sizeof(DrawArraysIndirectCommand), count, sizeof(DrawArraysIndirectCommand));
  } else {
    for (uint32_t k = 0; k < count; k++) {
      mCurrentCommandBuffer.draw(mDisplay->vertexBufferCount()[iSector][first + k], 1, mDisplay->vertexBufferStart()[iSector][first + k], 0);
    }
  }

  return count;
}

bool GPUDisplayBackendVulkan::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (mDisplay->updateDrawCommands() || toMixBuffer || includeMixImage > 0) {
    needRecordCommandBuffers();
  }

  if (includeMixImage > 0.f && mFrameSkipped) {
    return false; // The render to the mix buffer of this frame did not get an image
  }
  if (includeMixImage == 0.f) {
    mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight;
    waitTimeline(mFrameTimelineValue[mCurrentFrame]);
    if (mLastPresentId > 1 && fpsLimitMode() == FPS_LIMIT_VSYNC) { // Refresh rate limit (FIFO): at most one frame queued (wait for the second-last present), keeps the rate if a frame misses the deadline, timeout in case the window is hidden
      const vk::Result retVal = (vk::Result)VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitForPresentKHR(static_cast<VkDevice>(mDevice), static_cast<VkSwapchainKHR>(mSwapChain), mLastPresentId - 1, 100'000'000);
      handleSwapChainResult(retVal, 1);
      if (retVal != vk::Result::eTimeout && retVal != vk::Result::eErrorOutOfDateKHR) {
        CHKERR(retVal);
      }
    }
    auto getImage = [&]() { // Timeout so that a hidden window cannot block the thread (which may also dispatch the window system events) forever
      return mDevice.acquireNextImageKHR(mSwapChain, 100'000'000, mImageAvailableSemaphore[mCurrentFrame], VkFence(VK_NULL_HANDLE), &mCurrentImageIndex);
    };

    vk::Result retVal = vk::Result::eSuccess;
    if (mPresentModeForVSyncLimit != (fpsLimitMode() == FPS_LIMIT_VSYNC)) { // Present mode depends on the FPS limit mode
      mMustUpdateSwapChain = true;
    }
    bool mustUpdateSwapChain = mMustUpdateSwapChain || mDisplay->updateRenderPipeline() || (toMixBuffer && !mMixingSupported) || mDownsampleFactor != getDownsampleFactor(requestScreenshot) || mSmoothLines != useNativeSmoothLines();
    if (!mustUpdateSwapChain) {
      retVal = getImage();
      if (retVal == vk::Result::eErrorOutOfDateKHR) {
        mustUpdateSwapChain = mMustUpdateSwapChain = true;
      }
    }
    if (mustUpdateSwapChain) {
      recreateRendering(requestScreenshot, toMixBuffer);
      retVal = getImage();
      mMustUpdateSwapChain = false;
    }
    handleSwapChainResult(retVal, 2);
    mFrameSkipped = retVal == vk::Result::eTimeout || retVal == vk::Result::eNotReady || retVal == vk::Result::eErrorOutOfDateKHR;
    if (mFrameSkipped) {
      return false;
    }
    CHKERR(retVal);
    waitTimeline(mImageTimelineValue[mCurrentImageIndex]);                                        // Previous frame using this image's resources (command buffers, uniform buffer, ...) finished
    if (mPresentFences.size()) {                                                                  // Previous present of this image done, its present semaphore can be signaled again
      CHKERR(mDevice.waitForFences(1, &mPresentFences[mCurrentImageIndex], true, 1'000'000'000)); // Timeout (positive result, not an error) in case a failed present did not signal it
      CHKERR(mDevice.resetFences(1, &mPresentFences[mCurrentImageIndex]));
    }

    mHasDrawnText = false;
    mFrameCommandBuffers.clear();
    const hmm_mat4 modelViewProj = proj * view;
    static_assert(sizeof(modelViewProj) == sizeof(mLastMVP));
    if (memcmp(&modelViewProj, mLastMVP, sizeof(mLastMVP))) {
      memcpy(mLastMVP, &modelViewProj, sizeof(mLastMVP));
      mUpdateUBOdrawMatrix = -1;
    }
    if (mUpdateUBOdrawMatrix & (1u << mCurrentImageIndex)) {
      mUpdateUBOdrawMatrix &= ~(1u << mCurrentImageIndex);
      writeToBuffer(mUniformBuffersMat[0][mCurrentImageIndex], sizeof(mLastMVP), mLastMVP);
    }
  }

  mCurrentCommandBuffer = toMixBuffer ? mCommandBuffersMix[mCurrentImageIndex] : mCommandBuffers[mCurrentImageIndex];
  mCurrentCommandBufferLastPipeline = -1;
  if (!mCommandBufferUpToDate[mCurrentImageIndex]) {
    startFillCommandBuffer(mCurrentCommandBuffer, mCurrentImageIndex, toMixBuffer);
  }
  return true;
}

void GPUDisplayBackendVulkan::handleSwapChainResult(vk::Result retVal, uint8_t suboptimalBit)
{
  if (retVal == vk::Result::eErrorOutOfDateKHR || (retVal == vk::Result::eSuboptimalKHR && !(mSwapChainSuboptimal & suboptimalBit))) {
    mMustUpdateSwapChain = true;
    if (retVal == vk::Result::eSuboptimalKHR) {
      mSwapChainSuboptimal |= suboptimalBit;
    }
  }
  if (retVal == vk::Result::eSuccess && !mMustUpdateSwapChain) {
    mSwapChainSuboptimal &= ~suboptimalBit;
  }
}

void GPUDisplayBackendVulkan::finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (!mCommandBufferUpToDate[mCurrentImageIndex]) {
    endFillCommandBuffer(mCurrentCommandBuffer);
    if (!toMixBuffer && includeMixImage == 0.f) {
      mCommandBufferUpToDate[mCurrentImageIndex] = true;
    }
  }
}

void GPUDisplayBackendVulkan::finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
  mFrameCommandBuffers.emplace_back(mCurrentCommandBuffer);
  if (!toMixBuffer) {
    if (includeMixImage > 0.f) {
      mixImages(mCommandBuffersTexture[mCurrentImageIndex], includeMixImage);
      mFrameCommandBuffers.emplace_back(mCommandBuffersTexture[mCurrentImageIndex]);
    }
    if (mDownsampleFSAA) {
      downsampleToFramebuffer(mCommandBuffersDownsample[mCurrentImageIndex]);
      mFrameCommandBuffers.emplace_back(mCommandBuffersDownsample[mCurrentImageIndex]);
    }

    vk::SemaphoreSubmitInfo waitInfo{mImageAvailableSemaphore[mCurrentFrame], 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eTransfer};
    vk::SubmitInfo2 submitInfo{{}, 1, &waitInfo};
    if (doScreenshot) {
      submitInfo.commandBufferInfoCount = mFrameCommandBuffers.size();
      submitInfo.pCommandBufferInfos = mFrameCommandBuffers.data();
      queueSubmit(nullptr, 1, &submitInfo, VkFence(VK_NULL_HANDLE));
      mDevice.waitIdle();
      if (mDisplay->cfgR().screenshotScaleFactor != 1) {
        readImageToPixels(mDownsampleImages[mCurrentImageIndex].image, mDownsampleImages[mCurrentImageIndex].view, mDownsampleFactor == 2 ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eShaderReadOnlyOptimal, mScreenshotPixels);
      } else {
        readImageToPixels(mSwapChainImages[mCurrentImageIndex], mSwapChainImageViews[mCurrentImageIndex], vk::ImageLayout::eColorAttachmentOptimal, mScreenshotPixels);
      }
      mFrameCommandBuffers.clear();
      submitInfo.waitSemaphoreInfoCount = 0;
    }
    if (mHasDrawnText) {
      mFrameCommandBuffers.emplace_back(mCommandBuffersText[mCurrentImageIndex]);
    }
    mFrameCommandBuffers.emplace_back(mCommandBuffersPresent[mCurrentImageIndex]);
    submitInfo.commandBufferInfoCount = mFrameCommandBuffers.size();
    submitInfo.pCommandBufferInfos = mFrameCommandBuffers.data();
    vk::SemaphoreSubmitInfo signalInfos[2] = {{mPresentSemaphore[mCurrentImageIndex], 0, vk::PipelineStageFlagBits2::eAllCommands}, {mTimelineSemaphore, 0, vk::PipelineStageFlagBits2::eAllCommands}};
    submitInfo.signalSemaphoreInfoCount = 2;
    submitInfo.pSignalSemaphoreInfos = signalInfos;
    mFrameTimelineValue[mCurrentFrame] = mImageTimelineValue[mCurrentImageIndex] = queueSubmit(&signalInfos[1].value, 1, &submitInfo, VkFence(VK_NULL_HANDLE));

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &mPresentSemaphore[mCurrentImageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &mSwapChain;
    presentInfo.pImageIndices = &mCurrentImageIndex;
    presentInfo.pResults = nullptr;
    const uint64_t presentId = mLastPresentId + 1;
    vk::PresentIdKHR presentIdInfo{1, &presentId};
    if (fpsLimitMode() == FPS_LIMIT_VSYNC) { // Present wait only for the refresh rate limit, the fpsRate limit (frontend timer) must not be capped at the refresh rate
      presentInfo.pNext = &presentIdInfo;
      mLastPresentId = presentId;
    }
    vk::SwapchainPresentFenceInfoKHR presentFenceInfo{};
    if (mPresentFences.size()) {
      presentFenceInfo = vk::SwapchainPresentFenceInfoKHR{1, &mPresentFences[mCurrentImageIndex], presentInfo.pNext};
      presentInfo.pNext = &presentFenceInfo;
    }
    vk::Result retVal;
    {
      std::lock_guard<std::mutex> guard(mQueueMutex);
      retVal = mGraphicsQueue.presentKHR(&presentInfo);
    }
    handleSwapChainResult(retVal, 4);
    if (retVal != vk::Result::eErrorOutOfDateKHR) {
      CHKERR(retVal);
    }
  }
}

void GPUDisplayBackendVulkan::downsampleToFramebuffer(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.reset({});
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);

  if (mDownsampleFactor != 2) {                                                                                                                                                                                                                                                                                                        // Exact box filter by shader, the linear blit below is only exact for factor 2
    cmdImageMemoryBarrier(commandBuffer, mSwapChainImages[mCurrentImageIndex], vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal); // Src stage chains with the acquire semaphore wait
    cmdImageMemoryBarrier(commandBuffer, mDownsampleImages[mCurrentImageIndex].image, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmdDownsample(commandBuffer, mDownsampleImages[mCurrentImageIndex].view, mSwapChainImageViews[mCurrentImageIndex], mScreenWidth, mScreenHeight, mDownsampleFactor);
    commandBuffer.end();
    return;
  }
  cmdImageMemoryBarrier(commandBuffer, mSwapChainImages[mCurrentImageIndex], vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal); // Src stage chains with the acquire semaphore wait
  cmdImageMemoryBarrier(commandBuffer, mDownsampleImages[mCurrentImageIndex].image, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);

  vk::Offset3D blitSizeSrc;
  blitSizeSrc.x = mRenderWidth;
  blitSizeSrc.y = mRenderHeight;
  blitSizeSrc.z = 1;
  vk::Offset3D blitSizeDst;
  blitSizeDst.x = mScreenWidth;
  blitSizeDst.y = mScreenHeight;
  blitSizeDst.z = 1;
  vk::ImageBlit imageBlitRegion{};
  imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageBlitRegion.srcSubresource.layerCount = 1;
  imageBlitRegion.srcOffsets[1] = blitSizeSrc;
  imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageBlitRegion.dstSubresource.layerCount = 1;
  imageBlitRegion.dstOffsets[1] = blitSizeDst;
  commandBuffer.blitImage(mDownsampleImages[mCurrentImageIndex].image, vk::ImageLayout::eTransferSrcOptimal, mSwapChainImages[mCurrentImageIndex], vk::ImageLayout::eTransferDstOptimal, 1, &imageBlitRegion, vk::Filter::eLinear);

  cmdImageMemoryBarrier(commandBuffer, mSwapChainImages[mCurrentImageIndex], vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal); // Next: text pass and / or present transition

  commandBuffer.end();
}

void GPUDisplayBackendVulkan::cmdDownsample(vk::CommandBuffer commandBuffer, vk::ImageView src, vk::ImageView dst, uint32_t width, uint32_t height, int32_t factor)
{
  vk::RenderingAttachmentInfo colorAttachment{dst, vk::ImageLayout::eColorAttachmentOptimal};
  colorAttachment.loadOp = vk::AttachmentLoadOp::eDontCare; // Fully overwritten
  cmdBeginRendering(commandBuffer, width, height, colorAttachment);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[8]);
  cmdPushDescriptors(commandBuffer, mPipelineLayoutTexture, mUniformBuffersMat[2][mCurrentImageIndex], src); // The UBO is not used by the shader
  commandBuffer.pushConstants(mPipelineLayoutTexture, vk::ShaderStageFlagBits::eFragment, 0, sizeof(factor), &factor);
  commandBuffer.draw(3, 1, 0, 0);
  commandBuffer.endRendering();
}

void GPUDisplayBackendVulkan::prepareText()
{
  if (mUpdateUBOtextMatrix & (1u << mCurrentImageIndex)) {
    mUpdateUBOtextMatrix &= ~(1u << mCurrentImageIndex);
    hmm_mat4 proj = HMM_Orthographic(0.f, mScreenWidth, 0.f, mScreenHeight, -1, 1);
    writeToBuffer(mUniformBuffersMat[1][mCurrentImageIndex], sizeof(proj), &proj);
  }
  mFontVertexBufferHost.clear();
  mTextDrawCommands.clear();
}

void GPUDisplayBackendVulkan::finishText()
{
  if (!mHasDrawnText) {
    return;
  }

  mCommandBuffersText[mCurrentImageIndex].reset({});

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  mCommandBuffersText[mCurrentImageIndex].begin(beginInfo);

  cmdBeginRenderingLoad(mCommandBuffersText[mCurrentImageIndex], mSwapChainImages[mCurrentImageIndex], mSwapChainImageViews[mCurrentImageIndex], mScreenWidth, mScreenHeight);

  VulkanBuffer& fontBuffer = mFontVertexBuffer[mCurrentImageIndex];
  const size_t fontBufferSize = mFontVertexBufferHost.size() * sizeof(mFontVertexBufferHost[0]);
  if (fontBuffer.size < fontBufferSize) { // Persistent per image, only grows
    if (fontBuffer.size) {
      clearBuffer(fontBuffer);
    }
    fontBuffer = createBuffer(std::max(fontBufferSize, 2 * fontBuffer.size), nullptr, vk::BufferUsageFlagBits::eVertexBuffer, 0);
  }
  writeToBuffer(fontBuffer, fontBufferSize, mFontVertexBufferHost.data());

  mCommandBuffersText[mCurrentImageIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[3]);
  cmdPushDescriptors(mCommandBuffersText[mCurrentImageIndex], mPipelineLayoutTexture, mUniformBuffersMat[1][mCurrentImageIndex], mFontImage.view);
  vk::DeviceSize offsets[] = {0};
  mCommandBuffersText[mCurrentImageIndex].bindVertexBuffers(0, 1, &fontBuffer.buffer, offsets);

  for (const auto& cmd : mTextDrawCommands) {
    mCommandBuffersText[mCurrentImageIndex].pushConstants(mPipelineLayoutTexture, vk::ShaderStageFlagBits::eFragment, 0, sizeof(cmd.color), cmd.color);
    mCommandBuffersText[mCurrentImageIndex].draw(cmd.nVertices, 1, cmd.firstVertex, 0);
  }

  mFontVertexBufferHost.clear();

  mCommandBuffersText[mCurrentImageIndex].endRendering();
  mCommandBuffersText[mCurrentImageIndex].end();
}

void GPUDisplayBackendVulkan::mixImages(vk::CommandBuffer commandBuffer, float mixSlaveImage)
{
  if (mUpdateUBOmixMatrix & (1u << mCurrentImageIndex)) {
    mUpdateUBOmixMatrix &= ~(1u << mCurrentImageIndex);
    hmm_mat4 proj = HMM_Orthographic(0.f, mRenderWidth, 0.f, mRenderHeight, -1, 1);
    writeToBuffer(mUniformBuffersMat[2][mCurrentImageIndex], sizeof(proj), &proj);
  }
  commandBuffer.reset({});
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);

  vk::Image& image = mDownsampleFSAA ? mDownsampleImages[mCurrentImageIndex + mImageCount].image : mMixImages[mCurrentImageIndex].image;
  cmdImageMemoryBarrier(commandBuffer, image, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  cmdBeginRenderingLoad(commandBuffer, mRenderTargetImage[mCurrentImageIndex], *mRenderTargetView[mCurrentImageIndex], mRenderWidth, mRenderHeight);

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[4]);
  cmdPushDescriptors(commandBuffer, mPipelineLayoutTexture, mUniformBuffersMat[2][mCurrentImageIndex], *mRenderTargetView[mCurrentImageIndex + mImageCount]);
  vk::DeviceSize offsets[] = {0};
  commandBuffer.bindVertexBuffers(0, 1, &mMixingTextureVertexArray.buffer, offsets);

  commandBuffer.pushConstants(mPipelineLayoutTexture, vk::ShaderStageFlagBits::eFragment, 0, sizeof(mixSlaveImage), &mixSlaveImage);
  commandBuffer.draw(6, 1, 0, 0);

  commandBuffer.endRendering();
  commandBuffer.end();
}

void GPUDisplayBackendVulkan::ActivateColor(std::array<float, 4>& color)
{
  if (mCommandBufferUpToDate[mCurrentImageIndex]) {
    return;
  }
  mCurrentCommandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(color), color.data());
}

void GPUDisplayBackendVulkan::pointSizeFactor(float factor)
{
  if (mCommandBufferUpToDate[mCurrentImageIndex]) {
    return;
  }
  float size = std::min(mDisplay->cfgL().pointSize * mDownsampleFactor * factor + (mDisplay->cfgL().smoothPoints ? SMOOTH_POINT_EXTRA_SIZE : 0.f), mMaxPointSize);
  mCurrentCommandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eVertex, sizeof(std::array<float, 4>), sizeof(size), &size);
}

bool GPUDisplayBackendVulkan::useNativeSmoothLines() const
{
  return mDisplay->cfgL().smoothLines && mSmoothLinesSupported && mDisplay->cfgR().rendererCompatSmoothing;
}

bool GPUDisplayBackendVulkan::useSmoothLineShader() const
{
  return mDisplay->cfgL().smoothLines && !useNativeSmoothLines();
}

void GPUDisplayBackendVulkan::lineWidthFactor(float factor)
{
  if (mCommandBufferUpToDate[mCurrentImageIndex]) {
    return;
  }
  float width = mDisplay->cfgL().lineWidth * mDownsampleFactor * factor;
  if (useSmoothLineShader()) {
    const float expand = mStrictLines ? 1.f : 1.4143f;                         // Non-strict wide lines may be parallelograms, widen so that the perpendicular width is covered
    width = std::min(width + SMOOTH_LINE_EXTRA_WIDTH, mMaxLineWidth / expand); // The shader must see the width that is actually rasterized
    mCurrentCommandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eVertex, 5 * sizeof(float), sizeof(width), &width);
    width *= expand;
  }
  mCurrentCommandBuffer.setLineWidth(std::min(width, mMaxLineWidth));
}

void GPUDisplayBackendVulkan::needRecordCommandBuffers()
{
  std::fill(mCommandBufferUpToDate.begin(), mCommandBufferUpToDate.end(), false);
}

void GPUDisplayBackendVulkan::addFontSymbol(int32_t symbol, int32_t sizex, int32_t sizey, int32_t offsetx, int32_t offsety, int32_t advance, void* data)
{
  if (symbol != (int32_t)mFontSymbols.size()) {
    throw std::runtime_error("Incorrect symbol ID");
  }
  mFontSymbols.emplace_back(FontSymbolVulkan{{{sizex, sizey}, {offsetx, offsety}, advance}, nullptr, 0.f, 0.f, 0.f, 0.f});
  auto& buffer = mFontSymbols.back().data;
  if (sizex && sizey) {
    buffer.reset(new char[sizex * sizey]);
    memcpy(buffer.get(), data, sizex * sizey);
  }
}

void GPUDisplayBackendVulkan::initializeTextDrawing()
{
  int32_t maxSizeX = 0, maxSizeY = 0, maxBigX = 0, maxBigY = 0, maxRowY = 0;
  bool smooth = smoothFont();
  // Build a mega texture containing all fonts
  for (auto& symbol : mFontSymbols) {
    maxSizeX = std::max(maxSizeX, symbol.size[0]);
    maxSizeY = std::max(maxSizeY, symbol.size[1]);
  }
  uint32_t nn = ceil(std::sqrt(mFontSymbols.size()));
  int32_t sizex = nn * maxSizeX;
  int32_t sizey = nn * maxSizeY;
  std::unique_ptr<char[]> bigImage{new char[sizex * sizey]};
  memset(bigImage.get(), 0, sizex * sizey);
  int32_t rowy = 0, colx = 0;
  for (uint32_t i = 0; i < mFontSymbols.size(); i++) {
    auto& s = mFontSymbols[i];
    if (colx + s.size[0] > sizex) {
      colx = 0;
      rowy += maxRowY;
      maxRowY = 0;
    }
    for (int32_t k = 0; k < s.size[1]; k++) {
      for (int32_t j = 0; j < s.size[0]; j++) {
        int8_t val = s.data.get()[j + k * s.size[0]];
        if (!smooth) {
          val = val < 0 ? 0xFF : 0;
        }
        bigImage.get()[(colx + j) + (rowy + k) * sizex] = val;
      }
    }
    s.data.reset();
    s.x0 = colx;
    s.x1 = colx + s.size[0];
    s.y0 = rowy;
    s.y1 = rowy + s.size[1];
    maxBigX = std::max(maxBigX, colx + s.size[0]);
    maxBigY = std::max(maxBigY, rowy + s.size[1]);
    maxRowY = std::max(maxRowY, s.size[1]);
    colx += s.size[0];
  }
  if (maxBigX != sizex) {
    for (int32_t y = 1; y < maxBigY; y++) {
      memmove(bigImage.get() + y * maxBigX, bigImage.get() + y * sizex, maxBigX);
    }
  }
  sizex = maxBigX;
  sizey = maxBigY;
  for (uint32_t i = 0; i < mFontSymbols.size(); i++) {
    auto& s = mFontSymbols[i];
    s.x0 /= sizex;
    s.x1 /= sizex;
    s.y0 /= sizey;
    s.y1 /= sizey;
  }

  mFontImage = createImage(sizex, sizey, bigImage.get(), sizex * sizey, vk::Format::eR8Unorm);
}

void GPUDisplayBackendVulkan::clearFontSymbols()
{
  mDevice.waitIdle();
  if (mFontImage.sizex && mFontImage.sizey) {
    clearImage(mFontImage);
    mFontImage.sizex = mFontImage.sizey = 0;
  }
  mFontSymbols.clear();
}

void GPUDisplayBackendVulkan::OpenGLPrint(const char* s, float x, float y, float* color, float scale)
{
  if (!mFreetypeInitialized || mDisplay->drawTextInCompatMode()) {
    return;
  }

  size_t firstVertex = mFontVertexBufferHost.size() / 4;
  if (smoothFont()) {
    scale *= 0.25f; // Font size is 48 to have nice bitmap, scale to size 12
  }

  for (const unsigned char* c = (const unsigned char*)s; *c; c++) {
    if ((int32_t)*c >= (int32_t)mFontSymbols.size()) {
      GPUError("Trying to draw unsupported symbol: %d > %d\n", (int32_t)*c, (int32_t)mFontSymbols.size());
      continue;
    }
    const FontSymbolVulkan& sym = mFontSymbols[*c];
    if (sym.size[0] && sym.size[1]) {
      mHasDrawnText = true;
      float xpos = x + sym.offset[0] * scale;
      float ypos = y - (sym.size[1] - sym.offset[1]) * scale;
      float w = sym.size[0] * scale;
      float h = sym.size[1] * scale;
      float vertices[6][4] = {
        {xpos, mScreenHeight - 1 - ypos, sym.x0, sym.y1},
        {xpos, mScreenHeight - 1 - (ypos + h), sym.x0, sym.y0},
        {xpos + w, mScreenHeight - 1 - ypos, sym.x1, sym.y1},
        {xpos + w, mScreenHeight - 1 - ypos, sym.x1, sym.y1},
        {xpos, mScreenHeight - 1 - (ypos + h), sym.x0, sym.y0},
        {xpos + w, mScreenHeight - 1 - (ypos + h), sym.x1, sym.y0}};
      size_t oldSize = mFontVertexBufferHost.size();
      mFontVertexBufferHost.resize(oldSize + 4 * 6);
      memcpy(&mFontVertexBufferHost[oldSize], &vertices[0][0], sizeof(vertices));
    }
    x += (sym.advance >> 6) * scale; // shift is in 1/64th of a pixel
  }

  size_t nVertices = mFontVertexBufferHost.size() / 4 - firstVertex;

  if (nVertices) {
    auto& c = mTextDrawCommands;
    if (c.size() && c.back().color[0] == color[0] && c.back().color[1] == color[1] && c.back().color[2] == color[2] && c.back().color[3] == color[3]) {
      c.back().nVertices += nVertices;
    } else {
      c.emplace_back(TextDrawCommand{firstVertex, nVertices, {color[0], color[1], color[2], color[3]}});
    }
  }
}

void GPUDisplayBackendVulkan::readImageToPixels(vk::Image image, vk::ImageView view, vk::ImageLayout layout, std::vector<char>& pixels)
{
  const vk::Format format = mSurfaceFormat.format;
  if (format != vk::Format::eB8G8R8A8Unorm && format != vk::Format::eB8G8R8A8Srgb && format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eR8G8B8A8Srgb) {
    GPUError("Screenshot not supported for swapchain format %s, only 8 bit BGRA / RGBA", vk::to_string(format).c_str());
    pixels.clear();
    return;
  }
  uint32_t width = mScreenWidth * mDisplay->cfgR().screenshotScaleFactor;
  uint32_t height = mScreenHeight * mDisplay->cfgR().screenshotScaleFactor;
  static constexpr int32_t bytesPerPixel = 4;
  pixels.resize(width * height * bytesPerPixel);

  vk::Image dstImage, src;
  vk::ImageView dstView;
  vk::DeviceMemory dstImageMemory;
  const bool scaled = mDisplay->cfgR().screenshotScaleFactor != 1;
  const int32_t factor = mRenderWidth / width;                        // FSAA factor between the rendered image and the screenshot
  const bool shaderDownsample = scaled && factor != 1 && factor != 2; // Exact box filter, the linear blit is only exact for factor 2
  const vk::ImageLayout srcLayout = shaderDownsample ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eTransferSrcOptimal;
  const vk::PipelineStageFlags2 srcStage = shaderDownsample ? vk::PipelineStageFlagBits2::eFragmentShader : vk::PipelineStageFlagBits2::eTransfer;
  VulkanBuffer readBuffer = createBuffer(width * height * bytesPerPixel, nullptr, vk::BufferUsageFlagBits::eTransferDst, 0);
  vk::CommandBuffer cmdBuffer = getSingleTimeCommandBuffer();
  cmdImageMemoryBarrier(cmdBuffer, image, vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eTransferWrite, srcStage, shaderDownsample ? vk::AccessFlagBits2::eShaderSampledRead : vk::AccessFlagBits2::eTransferRead, layout, srcLayout);
  if (shaderDownsample) {
    createImageI(mDevice, mPhysicalDevice, dstImage, dstImageMemory, width, height, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
    dstView = createImageViewI(mDevice, dstImage, mSurfaceFormat.format);
    cmdImageMemoryBarrier(cmdBuffer, dstImage, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    cmdDownsample(cmdBuffer, view, dstView, width, height, factor);
    cmdImageMemoryBarrier(cmdBuffer, dstImage, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);
    src = dstImage;
  } else if (scaled) {
    createImageI(mDevice, mPhysicalDevice, dstImage, dstImageMemory, width, height, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
    cmdImageMemoryBarrier(cmdBuffer, dstImage, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    vk::Offset3D blitSizeSrc = {(int32_t)mRenderWidth, (int32_t)mRenderHeight, 1};
    vk::Offset3D blitSizeDst = {(int32_t)width, (int32_t)height, 1};
    vk::ImageBlit imageBlitRegion{};
    imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageBlitRegion.srcSubresource.layerCount = 1;
    imageBlitRegion.srcOffsets[1] = blitSizeSrc;
    imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageBlitRegion.dstSubresource.layerCount = 1;
    imageBlitRegion.dstOffsets[1] = blitSizeDst;
    cmdBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, dstImage, vk::ImageLayout::eTransferDstOptimal, 1, &imageBlitRegion, vk::Filter::eLinear);
    src = dstImage;
    cmdImageMemoryBarrier(cmdBuffer, dstImage, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal);
  } else {
    src = image;
  }

  vk::BufferImageCopy copyRegion{};
  copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  copyRegion.imageSubresource.layerCount = 1;
  copyRegion.imageExtent = vk::Extent3D{width, height, 1};
  cmdBuffer.copyImageToBuffer(src, vk::ImageLayout::eTransferSrcOptimal, readBuffer.buffer, 1, &copyRegion);

  vk::BufferMemoryBarrier2 hostBarrier{vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, readBuffer.buffer, 0, VK_WHOLE_SIZE};
  cmdBuffer.pipelineBarrier2(vk::DependencyInfo{{}, 0, nullptr, 1, &hostBarrier, 0, nullptr});
  cmdImageMemoryBarrier(cmdBuffer, image, srcStage, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eNone, srcLayout, layout); // Back to the layout the following passes expect
  submitSingleTimeCommandBuffer(cmdBuffer);

  const bool swapRB = format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eR8G8B8A8Srgb; // Raw copy in swapchain format, the BMP needs BGRA
  const char* data = (const char*)readBuffer.mapped;
  for (uint32_t i = 0; i < height; i++) { // BMP rows are bottom-up
    char* dst = pixels.data() + i * width * bytesPerPixel;
    memcpy(dst, data + (height - i - 1) * width * bytesPerPixel, width * bytesPerPixel);
    if (swapRB) {
      for (uint32_t j = 0; j < width; j++) {
        std::swap(dst[j * bytesPerPixel], dst[j * bytesPerPixel + 2]);
      }
    }
  }
  clearBuffer(readBuffer);
  if (scaled) {
    if (dstView) {
      mDevice.destroyImageView(dstView, nullptr);
    }
    mDevice.freeMemory(dstImageMemory, nullptr);
    mDevice.destroyImage(dstImage, nullptr);
  }
}

uint32_t GPUDisplayBackendVulkan::DepthBits()
{
  return 32;
}

bool GPUDisplayBackendVulkan::backendNeedRedraw()
{
  return !mCommandBufferUpToDate[mCurrentImageIndex];
}
