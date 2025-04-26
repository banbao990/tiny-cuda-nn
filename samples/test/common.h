#pragma once

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#endif

#include "../config.h"
#define OUTPUT_DIRECTORY PROJECT_DIRECTORY "outputs/"

#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>
#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <memory>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;
