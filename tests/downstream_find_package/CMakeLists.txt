# Standalone consumer used by .github/workflows/ci-build.yml.
#
# It is deliberately *not* part of the top-level HPTT build: it resolves
# HPTT the way a real downstream project does, through
# `find_package(hptt CONFIG)` and the `hptt::` imported targets, so it
# exercises the generated hpttConfig.cmake / hpttTargets.cmake rather than
# the in-tree target names. The workflow configures it twice, once against
# the build tree (via hptt_DIR) and once against an install prefix (via
# CMAKE_PREFIX_PATH).
cmake_minimum_required(VERSION 3.21)
project(hptt_downstream_consumer LANGUAGES CXX)

find_package(hptt CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE hptt::hptt_static)

# hptt::hptt_dyn only exists when HPTT was built with HPTT_BUILD_SHARED=ON,
# so a static-only HPTT still configures here.
if(TARGET hptt::hptt_dyn)
    add_executable(consumer_dyn consumer.cpp)
    target_link_libraries(consumer_dyn PRIVATE hptt::hptt_dyn)

    # On Windows the DLL has to sit next to the executable to be found at
    # run time. TARGET_RUNTIME_DLLS is empty on platforms that use rpath,
    # so the command is skipped there rather than guarded on WIN32.
    add_custom_command(TARGET consumer_dyn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:consumer_dyn>>,copy,true>
                $<TARGET_RUNTIME_DLLS:consumer_dyn> $<TARGET_FILE_DIR:consumer_dyn>
        COMMAND_EXPAND_LISTS)
endif()
