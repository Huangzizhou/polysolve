set(SOURCES
	BoxConstraintSolver.cpp
	BoxConstraintSolver.hpp
	Solver.hpp
	Solver.cpp
	Problem.hpp
	Problem.cpp
	PostStepData.hpp
	PostStepData.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(descent_strategies)
add_subdirectory(line_search)
