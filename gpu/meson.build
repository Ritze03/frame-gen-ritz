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
  'shaders/v2_medhist.comp', 'shaders/v2_search.comp', 'shaders/v2_clean.comp', 'shaders/v2_morph.comp', 'shaders/v2_repick.comp',
  'shaders/v2_qfield.comp', 'shaders/v2_qown.comp', 'shaders/v2_qcov.comp', 'shaders/v2_qmorph.comp', 'shaders/v2_static.comp', 'shaders/v2_enclose.comp',
  'shaders/v2_morph_s.comp', 'shaders/v2_qmorph_s.comp',
  'shaders/ml_dup.comp', 'shaders/ml_reduce.comp', 'shaders/ml_pyr.comp', 'shaders/ml_match.comp', 'shaders/ml_conv.comp', 'shaders/ml_out.comp',
  'shaders/ml_zero.comp', 'shaders/ml_guard.comp', 'shaders/ml_planes.comp', 'shaders/ml_feat.comp')

# MotionVersion::Learned's weights: gpu/ml/<run>/ (manifest.json + weights.bin, ml-fg's export unchanged) -> ml_weights.h, checked
# against the manifest (sha256, shapes, offsets) and re-laid out for shaders/ml_conv.comp. Order = LearnedNet (E2v3, E1v3, E2v4, E1v4,
# E2v5, E1v5, G2v2, G2v3, M2v2): a new net is appended here and in framegen.cpp kMlNetInfo / kMlNet.
ml_nets = ['e2v3', 'e1v3', 'e2v4', 'e1v4', 'e2v5', 'e1v5', 'g2v2', 'g2v3', 'm2v2']
python = find_program('python3')
ml_inputs = ['ml/embed_weights.py']
ml_args = [python, '-I', '@INPUT0@', '@OUTPUT@']
foreach n : ml_nets
  ml_args += [n, '@INPUT' + ml_inputs.length().to_string() + '@', '@INPUT' + (ml_inputs.length() + 1).to_string() + '@']
  ml_inputs += ['ml/' + n + '/manifest.json', 'ml/' + n + '/weights.bin']
endforeach
ml_weights = custom_target('ml_weights', input : ml_inputs, output : 'ml_weights.h', command : ml_args)

# Per-format variants of the colour shaders (README.md "Formats"): the same sources, compiled once per format with
# -DFG_FORMAT_<FMT>=1 into <name>_<fmt>_spv.h / <name>_<fmt>_spv. Rgba8 is the plain build above and keeps its names.
# framegen.cpp includes each variant behind __has_include, so a format that is not built is simply refused at init.
colour_srcs = ['shaders/luma.comp', 'shaders/error.comp', 'shaders/resolve.comp', 'shaders/fallback.comp',
               'shaders/ui_detect.comp', 'shaders/ui_box_update.comp', 'shaders/ui_inpaint.comp', 'shaders/ui_patch.comp',
               'shaders/ui_v2_update.comp', 'shaders/ui_v2_lumafill.comp',
               'shaders/blur_error.comp', 'shaders/blur_resolve.comp']
chroma_srcs = ['shaders/ui_inpaint_chroma.comp', 'shaders/ui_patch_chroma.comp']   # YUV only: the half-resolution CbCr plane
# MotionVersion::V2: only its synthesis reads and writes colour (both modes, with and without the static-overlay tests; YUV: both
# planes); its other passes read the luma, which luma.comp already gives per format
v2_colour_srcs = ['shaders/v2_morph.comp', 'shaders/v2_morph_s.comp', 'shaders/v2_qmorph.comp', 'shaders/v2_qmorph_s.comp']
# MotionVersion::Learned: only its duplicate test, its static guard and the colour variant's chroma planes read colour; the net and the zero test read the luma
ml_colour_srcs = ['shaders/ml_dup.comp', 'shaders/ml_guard.comp', 'shaders/ml_planes.comp']
variant_shaders = []
foreach f : get_option('formats')
  g = generator(glslang,
    output : '@BASENAME@_' + f + '_spv.h',
    depfile : '@BASENAME@_' + f + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DFG_FORMAT_' + f.to_upper() + '=1', '--vn', '@BASENAME@_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
  variant_shaders += [g.process(colour_srcs), g.process(v2_colour_srcs), g.process(ml_colour_srcs)]
  if f == 'nv12' or f == 'p010'
    variant_shaders += [g.process(chroma_srcs)]
  endif
endforeach

# The library only needs the Vulkan headers: every device function is loaded through gdpa.
framegen_lib = static_library('framegen', 'framegen.cpp', shaders, variant_shaders, ml_weights,
  dependencies : vulkan.partial_dependency(compile_args : true, includes : true))
framegen_dep = declare_dependency(link_with : framegen_lib, include_directories : include_directories('.'))

executable('fgtest', 'fgtest.cpp', dependencies : [framegen_dep, vulkan])

executable('uitest', 'uitest.cpp', dependencies : [framegen_dep, vulkan])

executable('formattest', 'formattest.cpp', dependencies : [framegen_dep, vulkan])

blend_bench = shader_gen.process('blendbench.comp')   # test only: the naive blend the benchmark compares against
executable('blurtest', 'blurtest.cpp', blend_bench, dependencies : [framegen_dep, vulkan])

# MotionVersion::Learned against ml-fg's test vectors (export/<run>/testvec, read from $MLFG_EXPORT, default the ml-fg checkout's
# export/; not copied into this repo) and its duplicate-frame skip. Needs libpng for the vectors' PNGs.
png = dependency('libpng', required : false)
if png.found()
  executable('mltest', 'mltest.cpp', dependencies : [framegen_dep, vulkan, png])
endif

# CPU-only: frame-generation pacing (pacing.h) and the motion-blur window. No GPU, no Vulkan.
pacingtest = executable('pacingtest', 'pacingtest.cpp')
test('pacing', pacingtest)
