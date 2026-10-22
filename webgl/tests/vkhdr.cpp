// vkhdr: the Vulkan library on one frame pair in a float or 10-bit format, for the WebGL2 backend's HDR reference test
// (webgl/tests/hdr.mjs). A trimmed fgtest (gpu/fgtest.cpp) with the image format and InitInfo::transfer as options:
//   vkhdr [--set k=v,..] [--t T] [--format f16|rgb10|rgba8] [--transfer encoded|linear] [--white NITS] W H prev curr out
// Files: f16 = RGBA half floats (8 bytes per pixel, little endian), rgb10 = A2B10G10R10 words, rgba8 = RGBA8; out in the same format.
// --set keys: algo=v1|v2, v2Mode=quality|performance, v2StaticMask, flowScale, trustLow, trustHigh, globalFallback, sceneCutSad.
// Prints "STATS 0 <untrusted px> <SAD sum x256> <SAD blocks> <W*H>" (Interpolator::recordDebugCounters) as fgtest --stats.
// Built by hdr.mjs: g++ -O2 -std=c++17 -I gpu webgl/tests/vkhdr.cpp gpu/build-webgl/libframegen.a -lvulkan.
#include "framegen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memProps;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}
struct Img { VkImage img; VkDeviceMemory mem; VkImageView view; };
static Img makeImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    Img r{};
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    CK(vkCreateImage(dev, &ici, nullptr, &r.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, r.img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &mai, nullptr, &r.mem));
    CK(vkBindImageMemory(dev, r.img, r.mem, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = r.img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &vci, nullptr, &r.view));
    return r;
}
static void imgBarrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    b.oldLayout = from;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

int main(int argc, char** argv) {
    framegen::Settings s;
    float t = 0.5f, white = 80.0f;
    std::string fmtName = "f16", transfer = "encoded";
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--t") && i + 1 < argc) t = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--format") && i + 1 < argc) fmtName = argv[++i];
        else if (!strcmp(argv[i], "--transfer") && i + 1 < argc) transfer = argv[++i];
        else if (!strcmp(argv[i], "--white") && i + 1 < argc) white = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            for (char* tok = strtok(argv[++i], ","); tok; tok = strtok(nullptr, ",")) {
                char* eq = strchr(tok, '=');
                if (!eq) return 2;
                *eq = 0;
                std::string k = tok, v = eq + 1;
                double d = atof(eq + 1);
                if (k == "algo") s.motionVersion = v == "v2" ? framegen::MotionVersion::V2 : framegen::MotionVersion::V1;
                else if (k == "v2Mode") s.v2Mode = v == "performance" ? framegen::V2Mode::Performance : framegen::V2Mode::Quality;
                else if (k == "v2StaticMask") s.v2StaticMask = d != 0;
                else if (k == "flowScale") s.flowScale = (uint32_t)d;
                else if (k == "trustLow") s.trustLow = (float)d;
                else if (k == "trustHigh") s.trustHigh = (float)d;
                else if (k == "globalFallback") s.globalFallback = (float)d;
                else if (k == "sceneCutSad") s.sceneCutSad = (float)d;
                else { fprintf(stderr, "unknown setting %s\n", k.c_str()); return 2; }
            }
        } else pos.push_back(argv[i]);
    }
    if (pos.size() != 5) { fprintf(stderr, "usage: vkhdr [--set ..] [--t T] [--format f16|rgb10|rgba8] [--transfer encoded|linear] [--white N] W H prev curr out\n"); return 2; }
    uint32_t W = atoi(pos[0].c_str()), H = atoi(pos[1].c_str());
    VkFormat fmt = fmtName == "f16" ? VK_FORMAT_R16G16B16A16_SFLOAT : fmtName == "rgb10" ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_R8G8B8A8_UNORM;
    size_t bytes = (size_t)W * H * (fmtName == "f16" ? 8 : 4);

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "vkhdr";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice phys = pds[0];
    for (auto p : pds) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = p; break; }
    }
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < n; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfi, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
    VkCommandPool cpool;
    CK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, nullptr, &fence));
    framegen::Interpolator fg;
    auto begin = [&] {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CK(vkBeginCommandBuffer(cb, &bi));
        fg.beginFrame();
    };
    auto submit = [&] {
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(queue, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        CK(vkResetFences(dev, 1, &fence));
    };
    auto hostBuf = [&](size_t size, VkBufferUsageFlags usage, VkBuffer& b, VkDeviceMemory& m, void** map) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        CK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                 memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        CK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CK(vkBindBufferMemory(dev, b, m, 0));
        CK(vkMapMemory(dev, m, 0, size, 0, map));
    };
    VkBuffer stg, dbg;
    VkDeviceMemory stgMem, dbgMem;
    uint8_t* map;
    uint32_t* dbgMap;
    hostBuf(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, stg, stgMem, (void**)&map);
    hostBuf(8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, dbg, dbgMem, (void**)&dbgMap);
    const VkImageUsageFlags inUse = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Img prev = makeImage(W, H, fmt, inUse), curr = makeImage(W, H, fmt, inUse);
    Img out = makeImage(W, H, fmt, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    begin();
    for (Img* i : {&prev, &curr, &out}) imgBarrier(cb, i->img, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    submit();

    framegen::InitInfo info;
    info.phys = phys;
    info.device = dev;
    info.gdpa = vkGetDeviceProcAddr;
    info.width = W;
    info.height = H;
    info.viewFormat = fmt;
    info.getMemProps = vkGetPhysicalDeviceMemoryProperties;
    info.getProps2 = vkGetPhysicalDeviceProperties2;
    info.getFormatProps = vkGetPhysicalDeviceFormatProperties;
    info.transfer = transfer == "linear" ? framegen::Transfer::Linear : framegen::Transfer::Encoded;
    info.linearWhiteNits = white;
    info.settings = s;
    if (!fg.init(info)) { fprintf(stderr, "fg.init failed: %s\n", fg.initError() ? fg.initError() : "?"); return 1; }

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {W, H, 1};
    auto upload = [&](const std::string& path, Img& img) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f || fread(map, 1, bytes, f) != bytes) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
        fclose(f);
        begin();
        imgBarrier(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(cb, stg, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        imgBarrier(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        submit();
    };
    upload(pos[2], prev);
    upload(pos[3], curr);
    begin();
    if (!fg.recordEstimate(cb, prev.view, curr.view) || !fg.recordSynth(cb, prev.view, curr.view, out.view, t)) { fprintf(stderr, "record failed\n"); return 1; }
    if (!fg.recordDebugCounters(cb, dbg, 0)) { fprintf(stderr, "recordDebugCounters failed\n"); return 1; }
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    submit();
    printf("STATS 0 %u %u %u %zu\n", dbgMap[0], dbgMap[1], fg.debugSadBlocks(), (size_t)W * H);
    begin();
    imgBarrier(cb, out.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, out.img, VK_IMAGE_LAYOUT_GENERAL, stg, 1, &region);
    submit();
    FILE* f = fopen(pos[4].c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", pos[4].c_str()); return 1; }
    fwrite(map, 1, bytes, f);
    fclose(f);
    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (Img* i : {&prev, &curr, &out}) { vkDestroyImageView(dev, i->view, nullptr); vkDestroyImage(dev, i->img, nullptr); vkFreeMemory(dev, i->mem, nullptr); }
    vkDestroyBuffer(dev, stg, nullptr); vkFreeMemory(dev, stgMem, nullptr);
    vkDestroyBuffer(dev, dbg, nullptr); vkFreeMemory(dev, dbgMem, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
