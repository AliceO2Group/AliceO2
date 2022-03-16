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

/// \file GPUDisplayBackendVulkan.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDVULKAN_H
#define GPUDISPLAYBACKENDVULKAN_H

#include "GPUDisplayBackend.h"
#include <vulkan/vulkan.hpp>

#include <vector>
#include <unordered_map>
#include <utils/vecpod.h>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct QueueFamiyIndices;
struct SwapChainSupportDetails;
struct VulkanBuffer;
struct VulkanImage;
struct FontSymbolVulkan;
struct TextDrawCommand;

class GPUDisplayBackendVulkan : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendVulkan();
  ~GPUDisplayBackendVulkan();

  unsigned int DepthBits() override;

 protected:
  unsigned int drawVertices(const vboList& v, const drawType t) override;
  void ActivateColor(std::array<float, 4>& color) override;
  void SetVSync(bool enable) override { mMustUpdateSwapChain = true; };
  void setDepthBuffer() override{};
  bool backendNeedRedraw() override;
  int InitBackendA() override;
  void ExitBackendA() override;
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) override;
  void finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void prepareText() override;
  void finishText() override;
  void mixImages(VkCommandBuffer cmdBuffer, float mixSlaveImage);
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;
  backendTypes backendType() const override { return TYPE_VULKAN; }
  void resizeScene(unsigned int width, unsigned int height) override;
  float getYFactor() const override { return -1.0f; }

  double checkDevice(VkPhysicalDevice device, const std::vector<const char*>& reqDeviceExtensions);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
  void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
  VulkanBuffer createBuffer(size_t size, const void* srcData = nullptr, VkBufferUsageFlags type = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, int deviceMemory = 1);
  void writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData);
  void clearBuffer(VulkanBuffer& buffer);
  VulkanImage createImage(unsigned int sizex, unsigned int sizey, const void* srcData = nullptr, size_t srcSize = 0, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);
  void writeToImage(VulkanImage& image, const void* srcData, size_t srcSize);
  void clearImage(VulkanImage& image);
  void clearVertexBuffers();

  void startFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex, bool toMixBuffer = false);
  void endFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex);
  VkCommandBuffer getSingleTimeCommandBuffer();
  void submitSingleTimeCommandBuffer(VkCommandBuffer commandBuffer);
  void readImageToPixels(VkImage image, VkImageLayout layout, std::vector<char>& pixels);
  void downsampleToFramebuffer(VkCommandBuffer& commandBuffer);

  void updateSwapChainDetails(const VkPhysicalDevice& device);
  void createDevice();
  void createTextureSampler();
  void createPipeline();
  void createSwapChain(bool forScreenshot = false, bool forMixing = false);
  void createShaders();
  void createUniformLayouts();
  void clearDevice();
  void clearTextureSampler();
  void clearPipeline();
  void clearSwapChain();
  void clearShaders();
  void clearUniformLayouts();
  void recreateSwapChain(bool forScreenshot = false, bool forMixing = false);
  void needRecordCommandBuffers();

  void addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) override;
  void initializeTextDrawing() override;
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override;

  unsigned int mIndirectId;
  std::vector<unsigned int> mVBOId;
  std::unique_ptr<QueueFamiyIndices> mQueueFamilyIndices;
  std::unique_ptr<SwapChainSupportDetails> mSwapChainDetails;
  int mModelViewProjId;
  int mColorId;

  bool mEnableValidationLayers = false;
  VkInstance mInstance;
  VkDebugUtilsMessengerEXT mDebugMessenger;
  VkPhysicalDevice mPhysicalDevice;
  VkDevice mDevice;
  VkQueue mGraphicsQueue;
  VkSurfaceKHR mSurface;
  VkSurfaceFormatKHR mSurfaceFormat;
  VkPresentModeKHR mPresentMode;
  VkSwapchainKHR mSwapChain;
  bool mSwapchainImageReadable;
  bool mMustUpdateSwapChain = false;
  std::vector<VkImage> mSwapChainImages;
  std::vector<VkImageView> mSwapChainImageViews;
  std::vector<VkImageView*> mRenderTargetView;
  std::vector<VulkanImage> mMSAAImages;
  std::vector<VulkanImage> mDownsampleImages;
  std::vector<VulkanImage> mZImages;
  std::vector<VulkanImage> mMixImages;
  std::unordered_map<std::string, VkShaderModule> mShaders;
  VkPipelineLayout mPipelineLayout;
  VkPipelineLayout mPipelineLayoutTexture;
  VkRenderPass mRenderPass;
  VkRenderPass mRenderPassText;
  VkRenderPass mRenderPassTexture;
  std::vector<VkPipeline> mPipelines;
  std::vector<VkFramebuffer> mFramebuffers;
  std::vector<VkFramebuffer> mFramebuffersText;
  std::vector<VkFramebuffer> mFramebuffersTexture;
  VkCommandPool mCommandPool;

  unsigned int mImageCount = 0;
  unsigned int mFramesInFlight = 0;
  int mCurrentFrame = 0;
  uint32_t mImageIndex = 0;
  VkCommandBuffer mCurrentCommandBuffer;
  int mCurrentCommandBufferLastPipeline = -1;

  std::vector<VkCommandBuffer> mCommandBuffers;
  std::vector<VkCommandBuffer> mCommandBuffersDownsample;
  std::vector<VkCommandBuffer> mCommandBuffersText;
  std::vector<VkCommandBuffer> mCommandBuffersTexture;
  std::vector<VkCommandBuffer> mCommandBuffersMix;
  std::vector<bool> mCommandBufferUpToDate;
  std::vector<VkSemaphore> mImageAvailableSemaphore;
  std::vector<VkSemaphore> mRenderFinishedSemaphore;
  std::vector<VkSemaphore> mTextFinishedSemaphore;
  std::vector<VkSemaphore> mMixFinishedSemaphore;
  std::vector<VkSemaphore> mDownsampleFinishedSemaphore;
  std::vector<VkFence> mInFlightFence;
  std::vector<VulkanBuffer> mUniformBuffersMat[3];
  std::vector<VulkanBuffer> mUniformBuffersCol[3];
  std::vector<VkDescriptorSet> mDescriptorSets[3];
  VkDescriptorSetLayout mUniformDescriptor;
  VkDescriptorSetLayout mUniformDescriptorTexture;
  VkDescriptorPool mDescriptorPool;

  std::vector<VulkanBuffer> mVBO;
  unsigned int mNVBOCreated = 0;
  std::vector<VulkanBuffer> mIndirectCommandBuffer;
  bool mIndirectCommandBufferCreated = false;

  std::vector<FontSymbolVulkan> mFontSymbols;
  std::unique_ptr<VulkanImage> mFontImage;
  std::vector<VulkanBuffer> mFontVertexBuffer;
  std::vector<TextDrawCommand> mTextDrawCommands;
  VkCommandBuffer mTmpTextCommandBuffer;
  VkSampler mTextureSampler;
  vecpod<float> mFontVertexBufferHost;
  bool mHasDrawnText = false;

  VkSampleCountFlagBits mMSAASampleCount = VK_SAMPLE_COUNT_1_BIT;
  unsigned int mMaxMSAAsupported = 0;
  bool mZActive = false;
  bool mZSupported = false;
  bool mDownsampleFSAA = false;

  VkFence mSingleCommitFence;

  bool mMixingSupported = 0;
  std::unique_ptr<VulkanBuffer> mTextureVertexArray;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
