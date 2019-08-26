find_path(Z3_INCLUDE_DIR NAMES z3.h)
find_library(Z3_LIBRARIES NAMES z3)

message(STATUS "Z3: ${Z3_INCLUDE_DIR} ${Z3_LIBRARIES}")

if (Z3_INCLUDE_DIR AND EXISTS "${Z3_INCLUDE_DIR}/z3_version.h")
  file(STRINGS "${Z3_INCLUDE_DIR}/z3_version.h" Z3_VERSION_STRS
       REGEX "^#define Z3_(MAJOR|MINOR|BUILD)_(VERSION|NUMBER)[ \t]+[0-9]+$")
  set(Z3_INSTALLED_VERSION "")
  foreach(VLINE ${Z3_VERSION_STRS})
    if (VLINE MATCHES "Z3_[A-Z_ \t]+([0-9]+)")
      set(Z3_INSTALLED_VERSION "${Z3_INSTALLED_VERSION}.${CMAKE_MATCH_1}")
    endif()
  endforeach()
  unset(Z3_VERSION_STRS)
  string(REGEX REPLACE "^\.(.+)" "\\1" Z3_INSTALLED_VERSION ${Z3_INSTALLED_VERSION})
else()
  set(Z3_INSTALLED_VERSION "0.0.0")
endif()

set(Z3_REQUIRED_VERSION "${Z3_FIND_VERSION_MAJOR}.${Z3_FIND_VERSION_MINOR}.${Z3_FIND_VERSION_PATCH}")
if (${Z3_INSTALLED_VERSION} VERSION_LESS ${Z3_REQUIRED_VERSION})
  message(FATAL_ERROR "Z3 required version: ${Z3_REQUIRED_VERSION} (installed: ${Z3_INSTALLED_VERSION})")
else()
  message(STATUS "Z3 installed version: ${Z3_INSTALLED_VERSION}")
endif()

add_library(Z3::Z3 INTERFACE IMPORTED)
set_target_properties(Z3::Z3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Z3_INCLUDE_DIR}")

if (CYGWIN)
  # cmake on cygwin doesn't seem to know about dlls..
  get_filename_component(Z3_LIB_DIR "${Z3_LIBRARIES}" DIRECTORY)
  set_target_properties(Z3::Z3 PROPERTIES LINK_FLAGS "-L${Z3_LIB_DIR}")
  set_target_properties(Z3::Z3 PROPERTIES INTERFACE_LINK_LIBRARIES "z3")

  file(COPY "${Z3_LIBRARIES}" DESTINATION "${PROJECT_BINARY_DIR}")
else()
  set_target_properties(Z3::Z3 PROPERTIES INTERFACE_LINK_LIBRARIES "${Z3_LIBRARIES}")
endif()
