project('framegen', 'cpp', default_options : ['cpp_std=c++17', 'buildtype=release', 'b_ndebug=if-release', 'warning_level=2'])
add_project_arguments('-Wno-missing-field-initializers', language : 'cpp')

vulkan = dependency('vulkan')
glslang = find_program('glslangValidator')

shader_gen = generator(glslang,
  output : '@BASENAME@_spv.h',
  depfile : '@BASENAME@_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '--vn', '@BASENAME@_spv', '-o', '@OUTPUT@', '@INPUT@'])

shaders = shader_gen.process(
  'shaders/luma.comp', 'shaders/down.comp', 'shaders/medhist.comp', 'shaders/search.comp',
  'shaders/median3.comp', 'shaders/subpix.comp', 'shaders/pixfield.comp', 'shaders/error.comp',
  'shaders/resolve.comp', 'shaders/fallback.comp', 'shaders/lookup.comp',
  'shaders/ui_detect.comp', 'shaders/ui_box_update.comp', 'shaders/ui_box_mask.comp', 'shaders/ui_mask.comp',
  'shaders/ui_inpaint.comp', 'shaders/ui_patch.comp', 'shaders/ui_v2_update.comp', 'shaders/ui_v2_mask.comp', 'shaders/ui_v2_lumafill.comp',
  'shaders/blur_lookup.comp', 'shaders/blur_error.comp', 'shaders/blur_resolve.comp',
  # MotionVersion::V2's own passes (Rgba8 only: no per-format variants); every other V2 pass is V1's shader
  'shaders/v2_occl.comp', 'shaders/v2_resolve.comp', 'shaders/v2_blur_resolve.comp')

# Per-format variants of the colour shaders (README.md "Formats"): the same sources, compiled once per format with
# -DFG_FORMAT_<FMT>=1 into <name>_<fmt>_spv.h / <name>_<fmt>_spv. Rgba8 is the plain build above and keeps its names.
# framegen.cpp includes each variant behind __has_include, so a format that is not built is simply refused at init.
colour_srcs = ['shaders/luma.comp', 'shaders/error.comp', 'shaders/resolve.comp', 'shaders/fallback.comp',
               'shaders/ui_detect.comp', 'shaders/ui_box_update.comp', 'shaders/ui_inpaint.comp', 'shaders/ui_patch.comp',
               'shaders/ui_v2_update.comp', 'shaders/ui_v2_lumafill.comp',
               'shaders/blur_error.comp', 'shaders/blur_resolve.comp']
chroma_srcs = ['shaders/ui_inpaint_chroma.comp', 'shaders/ui_patch_chroma.comp']   # YUV only: the half-resolution CbCr plane
variant_shaders = []
foreach f : get_option('formats')
  g = generator(glslang,
    output : '@BASENAME@_' + f + '_spv.h',
    depfile : '@BASENAME@_' + f + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DFG_FORMAT_' + f.to_upper() + '=1', '--vn', '@BASENAME@_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
  variant_shaders += [g.process(colour_srcs)]
  if f == 'nv12' or f == 'p010'
    variant_shaders += [g.process(chroma_srcs)]
  endif
endforeach

# The library only needs the Vulkan headers: every device function is loaded through gdpa.
framegen_lib = static_library('framegen', 'framegen.cpp', shaders, variant_shaders,
  dependencies : vulkan.partial_dependency(compile_args : true, includes : true))
framegen_dep = declare_dependency(link_with : framegen_lib, include_directories : include_directories('.'))

executable('fgtest', 'fgtest.cpp', dependencies : [framegen_dep, vulkan])

executable('uitest', 'uitest.cpp', dependencies : [framegen_dep, vulkan])

executable('formattest', 'formattest.cpp', dependencies : [framegen_dep, vulkan])

blend_bench = shader_gen.process('blendbench.comp')   # test only: the naive blend the benchmark compares against
executable('blurtest', 'blurtest.cpp', blend_bench, dependencies : [framegen_dep, vulkan])

# CPU-only: frame-generation pacing (pacing.h) and the motion-blur window. No GPU, no Vulkan.
pacingtest = executable('pacingtest', 'pacingtest.cpp')
test('pacing', pacingtest)
