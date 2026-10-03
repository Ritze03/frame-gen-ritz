// fgtest: runs framegen::Interpolator on raw RGBA8 frame pairs and/or benchmarks it.
//   fgtest [--bgra] [--bench N] W H prev.rgba curr.rgba out.rgba [prev curr out ...]
// Files are tightly packed RGBA8, W*H*4 bytes. --bgra uses B8G8R8A8_UNORM images/views for
// inputs and output (the file data stays RGBA; channels are swizzled on upload/readback).
// All images are kept in VK_IMAGE_LAYOUT_GENERAL, like gamescope.
#include "framegen.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice phys;
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
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
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
    bool bgra = false;
    int bench = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bgra")) bgra = true;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 5 || (pos.size() - 2) % 3) {
        fprintf(stderr, "usage: fgtest [--bgra] [--bench N] W H prev.rgba curr.rgba out.rgba [...]\n");
        return 2;
    }
    uint32_t W = atoi(pos[0].c_str()), H = atoi(pos[1].c_str());
    size_t bytes = (size_t)W * H * 4;
    VkFormat fmt = bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "fgtest";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    phys = pds[0];
    for (auto p : pds) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = p; break; }
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < n; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    fprintf(stderr, "device: %s, queue family %u, format %s\n", props.deviceName, qfi, bgra ? "BGRA8" : "RGBA8");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfi, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));   // no extra features, as in gamescope's minimum
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi};
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
        fg.beginFrame();   // one submission per command buffer
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

    // staging buffer
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer stg;
    CK(vkCreateBuffer(dev, &bci, nullptr, &stg));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, stg, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                             memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory stgMem;
    CK(vkAllocateMemory(dev, &mai, nullptr, &stgMem));
    CK(vkBindBufferMemory(dev, stg, stgMem, 0));
    uint8_t* map;
    CK(vkMapMemory(dev, stgMem, 0, bytes, 0, (void**)&map));

    Img prev = makeImage(W, H, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img curr = makeImage(W, H, fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
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
    if (!fg.init(info)) { fprintf(stderr, "fg.init failed\n"); return 1; }

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {W, H, 1};
    auto swz = [&](uint8_t* p) { if (bgra) for (size_t i = 0; i < bytes; i += 4) std::swap(p[i], p[i + 2]); };
    auto upload = [&](const std::string& path, Img& img) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f || fread(map, 1, bytes, f) != bytes) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
        fclose(f);
        swz(map);
        begin();
        imgBarrier(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(cb, stg, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        imgBarrier(cb, img.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        submit();
    };

    for (size_t p = 2; p + 2 < pos.size(); p += 3) {
        upload(pos[p], prev);
        upload(pos[p + 1], curr);
        begin();
        if (!fg.record(cb, prev.view, curr.view, out.view)) { fprintf(stderr, "fg.record failed\n"); return 1; }
        imgBarrier(cb, out.img, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(cb, out.img, VK_IMAGE_LAYOUT_GENERAL, stg, 1, &region);
        submit();
        swz(map);
        FILE* f = fopen(pos[p + 2].c_str(), "wb");
        fwrite(map, 1, bytes, f);
        fclose(f);
    }

    if (bench > 0) {
        double tp = props.limits.timestampPeriod * 1e-6;   // ticks -> ms
        if (!fg.setProfiling(true)) { fprintf(stderr, "setProfiling failed\n"); return 1; }
        std::vector<const char*> names;
        std::vector<std::vector<double>> per;
        std::vector<double> total;
        // Each run: 8 unprofiled records to bring the GPU clocks up (as a running game would), then the
        // profiled one, all in one command buffer. Isolated single submits mostly measure clock ramp-up.
        for (int it = 0; it < bench + 5; it++) {
            begin();
            fg.setProfiling(false);
            for (int k = 0; k < 8; k++) fg.record(cb, prev.view, curr.view, out.view);
            fg.setProfiling(true);
            if (!fg.record(cb, prev.view, curr.view, out.view)) { fprintf(stderr, "fg.record failed\n"); return 1; }
            submit();
            std::vector<uint64_t> ticks;
            if (!fg.readTimings(names, ticks)) { fprintf(stderr, "readTimings failed\n"); return 1; }
            if (it < 5) continue;   // warm-up
            per.resize(ticks.size());
            double sum = 0;
            for (size_t i = 0; i < ticks.size(); i++) { per[i].push_back(ticks[i] * tp); sum += ticks[i] * tp; }
            total.push_back(sum);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        printf("timing %ux%u, %d runs (profiled record after 8 warm records), median ms:\n", W, H, bench);
        for (size_t i = 0; i < names.size(); i++) printf("  %-18s %8.4f\n", names[i], med(per[i]));
        printf("  %-18s %8.4f   (min %.4f)\n", "TOTAL", med(total), *std::min_element(total.begin(), total.end()));

        // Back-to-back: 'bench' records in one command buffer, GPU kept busy -> steady clocks.
        fg.setProfiling(false);
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 2;
        VkQueryPool pool;
        CK(vkCreateQueryPool(dev, &qp, nullptr, &pool));
        int batch = std::min(bench, (int)framegen::Interpolator::kSetRing - 1);
        std::vector<double> per1;
        for (int rep = 0; rep < 5; rep++) {
            begin();
            vkCmdResetQueryPool(cb, pool, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
            for (int i = 0; i < batch; i++)
                if (!fg.record(cb, prev.view, curr.view, out.view)) { fprintf(stderr, "fg.record failed\n"); return 1; }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
            submit();
            uint64_t ts[2];
            CK(vkGetQueryPoolResults(dev, pool, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            per1.push_back((ts[1] - ts[0]) * tp / batch);
        }
        printf("  back-to-back (%d per command buffer, 5 reps): median %.4f ms per frame\n", batch, med(per1));
        vkDestroyQueryPool(dev, pool, nullptr);
    }

    vkDeviceWaitIdle(dev);
    fg.destroy();
    for (Img* i : {&prev, &curr, &out}) {
        vkDestroyImageView(dev, i->view, nullptr);
        vkDestroyImage(dev, i->img, nullptr);
        vkFreeMemory(dev, i->mem, nullptr);
    }
    vkDestroyBuffer(dev, stg, nullptr);
    vkFreeMemory(dev, stgMem, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
