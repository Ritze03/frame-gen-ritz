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
  'shaders/resolve.comp', 'shaders/fallback.comp', 'shaders/lookup.comp')

# The library only needs the Vulkan headers: every device function is loaded through gdpa.
framegen_lib = static_library('framegen', 'framegen.cpp', shaders,
  dependencies : vulkan.partial_dependency(compile_args : true, includes : true))
framegen_dep = declare_dependency(link_with : framegen_lib, include_directories : include_directories('.'))

executable('fgtest', 'fgtest.cpp', dependencies : [framegen_dep, vulkan])
