###############################################################################
# Samples
###############################################################################

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(test_out
	utils.cu
	../../dependencies/stbi/stbi_wrapper.cpp
)
target_link_libraries(test_out PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn)

add_executable(one_out one_out.cu)
target_link_libraries(one_out PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn test_out)

add_executable(two_out two_out.cu)
target_link_libraries(two_out PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn test_out)
