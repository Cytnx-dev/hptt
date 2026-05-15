cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(HPTT
    DESCRIPTION "High-Performance Tensor Transpose library"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ENABLE_IBM "Enable IBM POWER architecture support" OFF)
option(ENABLE_AVX "Enable x86 AVX (REGISTER_BITS=256) support" OFF)
option(ENABLE_ARM "Enable ARM/Neon (REGISTER_BITS=128) support" OFF)
option(FINE_TUNE "Fine-tune for the host CPU (uses -march=native; produces non-portable binaries)" OFF)

# HPTT_INSTALL: ON by default for standalone builds, but a parent project
# using add_subdirectory() can set it OFF if it doesn't want HPTT to emit
# its own install rules (e.g. when the parent prefers to ship libhptt and
# its headers from its own install layout).
option(HPTT_INSTALL "Generate install rules for HPTT" ON)

# -----------------------------------------------------------------------------
# Build flags split by *visibility* to consumers.
#
#   HPTT_PRIVATE_COMPILE_OPTIONS
#       Optimization / architecture flags (e.g. -march=native, -O3,
#       -ffast-math, -fopenmp). PRIVATE: applied only when compiling HPTT's
#       own .cpp files. Downstream consumers that link against libhptt
#       must NOT silently inherit these — flags like -ffast-math or
#       -march=native change floating-point semantics across the
#       boundary and break the consumer's own numerical tests.
#
#   HPTT_PUBLIC_COMPILE_DEFINITIONS
#       Preprocessor switches that appear in HPTT's *public* headers
#       (currently `include/hptt_types.h` checks `HPTT_ARCH_ARM` to set
#       REGISTER_BITS). PUBLIC: consumers including HPTT headers must see
#       the same value HPTT was built with, otherwise the macro silently
#       diverges across the build / use boundary.
# -----------------------------------------------------------------------------
set(HPTT_PRIVATE_COMPILE_OPTIONS)
set(HPTT_PUBLIC_COMPILE_DEFINITIONS)

# Per-compiler architecture / optimization flags. OpenMP-specific flags
# (`-fopenmp`, `-qopenmp`, `-qsmp=omp`, etc.) are intentionally *not* listed
# here — `find_package(OpenMP)` + the PUBLIC `OpenMP::OpenMP_CXX` link below
# provides them via the imported target's INTERFACE_COMPILE_OPTIONS, which
# is per-toolchain-correct (AppleClang in particular often rejects a bare
# `-fopenmp`).
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -xhost)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(APPLE)
        list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -mtune=native)
    elseif(NOT ENABLE_IBM)
        if(FINE_TUNE)
            list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -march=native -mtune=native)
        else()
            list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -march=x86-64 -mtune=generic)
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(FINE_TUNE)
        list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -march=native)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -silent -w -Mnovect)
endif()

if(ENABLE_AVX)
    list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -mavx)
    list(APPEND HPTT_PUBLIC_COMPILE_DEFINITIONS HPTT_ARCH_AVX)
elseif(ENABLE_ARM)
    list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -mfpu=neon)
    list(APPEND HPTT_PUBLIC_COMPILE_DEFINITIONS HPTT_ARCH_ARM)
elseif(ENABLE_IBM)
    list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS -mtune=native -maltivec -mabi=altivec)
    list(APPEND HPTT_PUBLIC_COMPILE_DEFINITIONS HPTT_ARCH_IBM)
endif()

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Running on Apple Silicon (M-series)")
        list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS
            -O3 -ffast-math -funroll-loops -ftree-vectorize -mcpu=native)
        list(APPEND HPTT_PUBLIC_COMPILE_DEFINITIONS HPTT_ARCH_ARM)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        message(STATUS "Running on Intel-based macOS")
        list(APPEND HPTT_PRIVATE_COMPILE_OPTIONS
            -O3 -ffast-math -funroll-loops -ftree-vectorize -mavx)
        list(APPEND HPTT_PUBLIC_COMPILE_DEFINITIONS HPTT_ARCH_AVX)
    endif()
endif()

set(HPTT_SRCS src/hptt.cpp src/plan.cpp src/transpose.cpp src/utils.cpp)

find_package(OpenMP REQUIRED)

# Configure both targets (static + shared) identically.
function(hptt_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME hptt)

    # Generator-expression-wrapped include dir: the source-tree path is
    # only visible *during the build*; after install, the path resolves to
    # the installed include dir. Without this, install(EXPORT ...) rejects
    # the absolute source-tree path with "prefixed in the source directory".
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

    target_compile_options(${target} PRIVATE ${HPTT_PRIVATE_COMPILE_OPTIONS})
    target_compile_definitions(${target} PUBLIC ${HPTT_PUBLIC_COMPILE_DEFINITIONS})

    # OpenMP::OpenMP_CXX must be PUBLIC for the static archive: libhptt.a
    # uses OpenMP internally, but the symbols stay unresolved until the
    # consumer's final executable link. We propagate the requirement so
    # the consumer's link line gets OpenMP automatically.
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

add_library(hptt_static STATIC ${HPTT_SRCS})
hptt_configure_target(hptt_static)

add_library(hptt_dyn SHARED ${HPTT_SRCS})
hptt_configure_target(hptt_dyn)

# Namespaced ALIAS targets. add_subdirectory() consumers can now use the
# same `hptt::hptt_static` / `hptt::hptt_dyn` names that find_package(hptt)
# consumers would, so swapping integration mechanisms doesn't require
# rewriting target_link_libraries() calls.
add_library(hptt::hptt_static ALIAS hptt_static)
add_library(hptt::hptt_dyn ALIAS hptt_dyn)

if(HPTT_INSTALL)
    install(TARGETS hptt_static hptt_dyn
            EXPORT hpttTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # Generate hpttTargets.cmake (defines `hptt::hptt_static` /
    # `hptt::hptt_dyn` for find_package(hptt CONFIG) consumers).
    install(EXPORT hpttTargets
            FILE hpttTargets.cmake
            NAMESPACE hptt::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hptt)

    # Generate hpttConfig.cmake from the template in cmake/. The template
    # re-finds OpenMP (since the exported targets reference
    # OpenMP::OpenMP_CXX in their INTERFACE_LINK_LIBRARIES, CMake would
    # otherwise reject the import with "target not found") and pulls in
    # hpttTargets.cmake.
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpttConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/hpttConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hptt)

    # Note: no hpttConfigVersion.cmake is shipped because HPTT does not
    # carry a declared version (no git tags, no version file). Consumers
    # calling `find_package(hptt CONFIG)` without a version request work;
    # `find_package(hptt 1.0 CONFIG)` would fail at the version check.
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/hpttConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hptt)
endif()
