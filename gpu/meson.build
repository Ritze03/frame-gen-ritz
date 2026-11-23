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
  'shaders/ml_zero.comp', 'shaders/ml_guard.comp', 'shaders/ml_planes.comp', 'shaders/ml_feat.comp',
  'shaders/v3_synth.comp', 'shaders/v3_planes.comp', 'shaders/v3_pyr.comp', 'shaders/v3_hist.comp', 'shaders/v3_search.comp', 'shaders/v3_search_top.comp', 'shaders/v3_dense.comp', 'shaders/v3_wide.comp', 'shaders/v3_stats.comp',
  'shaders/v3_occl.comp', 'shaders/v3_fill.comp', 'shaders/v3_project.comp', 'shaders/v3_enclose.comp', 'shaders/v3_decide.comp',
  'shaders/v3_anchor.comp', 'shaders/v3_guard.comp',
  'shaders/debug_motion.comp', 'shaders/debug_ui.comp')   # task 136: Settings::debugView (the motion tint for V1 / Learned, the UI views); task 104: Settings::v3Guard, V2 Quality's vectors correct V3's dense field

# MotionVersion::Learned's weights: gpu/ml/<run>/ (manifest.json + weights.bin, ml-fg's export unchanged) -> ml_weights.h, checked
# against the manifest (sha256, shapes, offsets) and re-laid out for shaders/ml_conv.comp. Order = LearnedNet (E2v3, E1v3, E2v4, E1v4,
# E2v5, E1v5, G2v2, G2v3, M2v2, M2v3): a new net is appended here and in framegen.cpp kMlNetInfo / kMlNet.
ml_nets = ['e2v3', 'e1v3', 'e2v4', 'e1v4', 'e2v5', 'e1v5', 'g2v2', 'g2v3', 'm2v2', 'm2v3']
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
# The wide motion search (radius 7..12: Settings::searchRadius / searchRadiusTop above 6): the same sources with FG_MAXR 12, into
# <name>_wide_spv. The plain builds above keep FG_MAXR 6 (their SPIR-V and shared-memory use are unchanged).
wide_gen = generator(glslang,
  output : '@BASENAME@_wide_spv.h',
  depfile : '@BASENAME@_wide_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '-DFG_MAXR=12', '--vn', '@BASENAME@_wide_spv', '-o', '@OUTPUT@', '@INPUT@'])
wide_shaders = wide_gen.process('shaders/search.comp', 'shaders/v2_search.comp')
# MotionVersion::V3 hook (j) (Settings::v3ColourWindow): the search and dense passes with the colour-similarity window weights, -DV3J=1, into
# <name>_j_spv (v3_search_j, v3_search_top_j, v3_dense_j). The plain builds keep their SPIR-V (the switch off costs nothing).
v3j_gen = generator(glslang,
  output : '@BASENAME@_j_spv.h',
  depfile : '@BASENAME@_j_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '-DV3J=1', '--vn', '@BASENAME@_j_spv', '-o', '@OUTPUT@', '@INPUT@'])
v3j_shaders = v3j_gen.process('shaders/v3_search.comp', 'shaders/v3_search_top.comp', 'shaders/v3_dense.comp')
# MotionVersion::V3 hook (l) (Settings::v3DetailSplit): the synthesis with the detail split, -DV3L=1, into v3_synth_l_spv (Rgba8 / Bgra8; the other formats'
# variants are built below as v3_synth_l_<fmt>_spv). The plain v3_synth builds keep their SPIR-V (the switch off costs nothing).
v3l_gen = generator(glslang,
  output : '@BASENAME@_l_spv.h',
  depfile : '@BASENAME@_l_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '-DV3L=1', '--vn', '@BASENAME@_l_spv', '-o', '@OUTPUT@', '@INPUT@'])
v3l_shaders = v3l_gen.process('shaders/v3_synth.comp')
# MotionVersion::V3 Settings::v3Blend (learned blend, experimental): the synthesis with the three per-pixel nets, -DV3B=1, into v3_synth_b_spv (Rgba8 / Bgra8; the per-format
# variants v3_synth_b_<fmt>_spv are built below, rgb10 only so far). Weights: shaders/v3_blend_weights.glsl (generated by ml/blend/export_glsl.py; the committed one is all zeros). The plain builds keep their SPIR-V.
v3b_gen = generator(glslang,
  output : '@BASENAME@_b_spv.h',
  depfile : '@BASENAME@_b_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '-DV3B=1', '--vn', '@BASENAME@_b_spv', '-o', '@OUTPUT@', '@INPUT@'])
v3b_shaders = v3b_gen.process('shaders/v3_synth.comp')
# Settings::debugView = Motion for V2 (-DDBG_V2=1, reads V2's field, binding 13) and V3 (-DDBG_V3=1, reads V3's T-field, bindings 14, 15): the same source as debug_motion
# (V1 / Learned) with its own bindings, into debug_motion_v2_spv / debug_motion_v3_spv (Rgba8 / Bgra8; the per-format variants debug_motion_v2_<fmt>_spv are built below, rgb10 and f16 only).
dbg_gens = []
foreach v : ['v2', 'v3']
  dbg_gens += generator(glslang,
    output : '@BASENAME@_' + v + '_spv.h',
    depfile : '@BASENAME@_' + v + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DDBG_' + v.to_upper() + '=1', '--vn', '@BASENAME@_' + v + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
endforeach
dbg_shaders = [dbg_gens[0].process('shaders/debug_motion.comp'), dbg_gens[1].process('shaders/debug_motion.comp')]
# The debug views draw on RGB frames only (Nv12 / P010: no view, a documented no-op), so their per-format variants are built for rgb10 and f16 only.
dbg_colour_srcs = ['shaders/debug_motion.comp', 'shaders/debug_ui.comp']
# The wide median histogram and the wide V2 Quality claims (reach settings off their defaults and a long vector): -DFG_WIDE=1,
# into <name>_wide_spv as well (other basenames). The plain builds keep their SPIR-V.
wide2_gen = generator(glslang,
  output : '@BASENAME@_wide_spv.h',
  depfile : '@BASENAME@_wide_spv.h.d',
  arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
               '-DFG_WIDE=1', '--vn', '@BASENAME@_wide_spv', '-o', '@OUTPUT@', '@INPUT@'])
wide2_shaders = wide2_gen.process('shaders/medhist.comp', 'shaders/v2_medhist.comp', 'shaders/v2_qown.comp', 'shaders/v2_qcov.comp')

# MotionVersion::V2: only its synthesis reads and writes colour (both modes, with and without the static-overlay tests; YUV: both
# planes); its other passes read the luma, which luma.comp already gives per format
v2_colour_srcs = ['shaders/v2_morph.comp', 'shaders/v2_morph_s.comp', 'shaders/v2_qmorph.comp', 'shaders/v2_qmorph_s.comp']
# MotionVersion::Learned: only its duplicate test, its static guard and the colour variant's chroma planes read colour; the net and the zero test read the luma
ml_colour_srcs = ['shaders/ml_dup.comp', 'shaders/ml_guard.comp', 'shaders/ml_planes.comp']
# MotionVersion::V3: its raw planes (the colour of both frames -> its level-0 planes) and its synthesis read colour (YUV: both planes); its other
# passes work on those planes and the luma. The static-structure mask (v2_static.comp) is V2's plain build: it reads the luma only.
v3_colour_srcs = ['shaders/v3_planes.comp', 'shaders/v3_synth.comp']
v3l_colour_srcs = ['shaders/v3_synth.comp']   # ... and the same synthesis with hook (l) (-DV3L=1), per format
variant_shaders = []
foreach f : get_option('formats')
  g = generator(glslang,
    output : '@BASENAME@_' + f + '_spv.h',
    depfile : '@BASENAME@_' + f + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DFG_FORMAT_' + f.to_upper() + '=1', '--vn', '@BASENAME@_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
  gl = generator(glslang,
    output : '@BASENAME@_l_' + f + '_spv.h',
    depfile : '@BASENAME@_l_' + f + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DFG_FORMAT_' + f.to_upper() + '=1', '-DV3L=1', '--vn', '@BASENAME@_l_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
  gb = generator(glslang,
    output : '@BASENAME@_b_' + f + '_spv.h',
    depfile : '@BASENAME@_b_' + f + '_spv.h.d',
    arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                 '-DFG_FORMAT_' + f.to_upper() + '=1', '-DV3B=1', '--vn', '@BASENAME@_b_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
  variant_shaders += [g.process(colour_srcs), g.process(v2_colour_srcs), g.process(ml_colour_srcs), g.process(v3_colour_srcs), gl.process(v3l_colour_srcs)]
  if f == 'rgb10'
    variant_shaders += [gb.process(v3l_colour_srcs)]   # learned blend (-DV3B=1) in rgb10 only so far
  endif
  if f == 'rgb10' or f == 'f16'
    foreach v : ['v2', 'v3']
      gd = generator(glslang,
        output : '@BASENAME@_' + v + '_' + f + '_spv.h',
        depfile : '@BASENAME@_' + v + '_' + f + '_spv.h.d',
        arguments : ['-V', '--target-env', 'vulkan1.2', '--quiet', '--depfile', '@DEPFILE@',
                     '-DFG_FORMAT_' + f.to_upper() + '=1', '-DDBG_' + v.to_upper() + '=1', '--vn', '@BASENAME@_' + v + '_' + f + '_spv', '-o', '@OUTPUT@', '@INPUT@'])
      variant_shaders += [gd.process('shaders/debug_motion.comp')]
    endforeach
    variant_shaders += [g.process(dbg_colour_srcs)]
  endif
  if f == 'nv12' or f == 'p010'
    variant_shaders += [g.process(chroma_srcs)]
  endif
endforeach

# The library only needs the Vulkan headers: every device function is loaded through gdpa.
framegen_lib = static_library('framegen', 'framegen.cpp', shaders, wide_shaders, wide2_shaders, v3j_shaders, v3l_shaders, v3b_shaders, dbg_shaders, variant_shaders, ml_weights,
  dependencies : vulkan.partial_dependency(compile_args : true, includes : true))
framegen_dep = declare_dependency(link_with : framegen_lib, include_directories : include_directories('.'))

executable('fgtest', 'fgtest.cpp', dependencies : [framegen_dep, vulkan])

executable('uitest', 'uitest.cpp', dependencies : [framegen_dep, vulkan])

executable('formattest', 'formattest.cpp', dependencies : [framegen_dep, vulkan])

# MotionVersion::V3's known-motion test (procedural scenes with exact flows and in-between frames, gpu/V3.md section 11)
executable('v3test', 'v3test.cpp', dependencies : [framegen_dep, vulkan, dependency('threads')])

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
