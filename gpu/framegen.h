// framegen: image-only frame interpolation (Vulkan compute port of proto/interp.py).
// Plain Vulkan, no gamescope types. Never submits, never touches queues.
//
// Threading / queues: not thread-safe. All records of one Interpolator must be submitted to ONE queue,
// in recording order: its internal buffers are shared between records and only ordered by pipeline
// barriers, which do not order against work on another queue.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace framegen {

struct InitInfo {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;   // all device functions are loaded through this
    uint32_t width = 0, height = 0;           // >= kMinSize each
    // B8G8R8A8_UNORM or R8G8B8A8_UNORM only (8-bit, UNORM, never _SRGB). Anything else -> init false.
    VkFormat viewFormat = VK_FORMAT_B8G8R8A8_UNORM;
    // Required (instance-level, cannot come from gdpa): used to put the buffers in DEVICE_LOCAL memory.
    // init() returns false if null.
    PFN_vkGetPhysicalDeviceMemoryProperties getMemProps = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
};

class Interpolator {
public:
    // false -> the object is not usable (record() is a no-op returning false); skip frame generation
    // for this surface. Fails for sizes below kMinSize, unsupported formats, null getMemProps, or
    // any Vulkan creation/allocation failure.
    // init() on a live object and destroy()/resize() free resources that earlier records use: the
    // caller must guarantee every previous record has retired first (fence/timeline wait, or
    // vkQueueWaitIdle/vkDeviceWaitIdle; resolution changes are rare).
    bool init(const InitInfo&);
    void destroy();
    // Re-creates the size-dependent buffers. Same retire rule as above. On false the object stays
    // alive but holds no size (record() returns false) until a later resize() succeeds.
    bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);

    // Call once per command buffer / submission, before its first record(): resets the per-submission
    // record count (see record()).
    void beginFrame();

    // All three views: 2D, single mip/layer, the same size as init/resize, format viewFormat, i.e.
    // 8-bit UNORM (an _SRGB view gives wrong luma; 10-bit is unsupported: skip FG or convert first),
    // VK_IMAGE_LAYOUT_GENERAL on entry and exit.
    // prev/curr: SAMPLED usage, already made readable by the caller. out: STORAGE usage, caller-owned.
    // Ends with a COMPUTE_SHADER SHADER_WRITE -> COMPUTE_SHADER SHADER_READ memory barrier (valid on a
    // compute-only queue). Other consumers of out (fragment, transfer) need the caller's own barrier.
    // t: only 0.5 is implemented (the reference is a midpoint interpolator); other values are ignored.
    // Descriptor sets come from a ring of kSetRing. Returns false and records nothing (out untouched,
    // so the caller must not substitute it) if the object is not initialised, or if kSetRing records
    // were already made since the last beginFrame() (asserts in debug builds). The caller must also
    // keep the records of all not-yet-retired submissions <= kSetRing in total.
    bool record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t = 0.5f);

    // Profiling (test/benchmark only): timestamps after every pass group. Returns false if the
    // query pool cannot be created (profiling then stays off).
    bool setProfiling(bool on);
    // Reads the last profiled record (waits for results). Fills names + GPU ticks per pass group
    // (multiply by VkPhysicalDeviceLimits::timestampPeriod for ns). Returns false if unavailable.
    bool readTimings(std::vector<const char*>& names, std::vector<uint64_t>& ticks);

    static constexpr uint32_t kSetRing = 32;
    static constexpr uint32_t kMinSize = 16;
    struct Impl;
private:
    Impl* m = nullptr;
};

}  // namespace framegen
