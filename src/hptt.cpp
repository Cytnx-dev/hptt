/**
  Copyright 2018 Paul Springer
  
  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  
  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>
#include <memory>

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <hptt.h>

namespace hptt {

namespace {
int parse_thread_count(const char *text, const char *name, bool allow_list) {
  const std::string error =
      std::string(name) + (allow_list ? " must be a comma-separated list of positive integers"
                                      : " must be a positive integer");
  int first = 0;
  for (;;) {
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || errno == ERANGE || value < 1 || value > INT_MAX)
      throw std::invalid_argument(error);
    if (first == 0) first = static_cast<int>(value);
    while (std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end == '\0') return first;
    if (!allow_list || *end != ',') throw std::invalid_argument(error);
    text = end + 1;
  }
}

int default_thread_count() {
  if (const char *value = std::getenv("HPTT_NUM_THREADS"))
    return parse_thread_count(value, "HPTT_NUM_THREADS", false);
  if (const char *value = std::getenv("OMP_NUM_THREADS"))
    return parse_thread_count(value, "OMP_NUM_THREADS", true);
  return 1;
}
}  // namespace

std::shared_ptr<Transpose<float>> create_plan(const int *perm, int dim, float alpha,
                                              const float *input, const int *size,
                                              const int *outer_size_input, float beta,
                                              float *output, const int *outer_size_output,
                                              SelectionMethod selection_method, Layout layout) {
  return create_plan(perm, dim, alpha, input, size, outer_size_input, beta, output,
                     outer_size_output, selection_method, default_thread_count(), nullptr,
                     layout == Layout::RowMajor);
}

std::shared_ptr<Transpose<double>> create_plan(const int *perm, int dim, double alpha,
                                               const double *input, const int *size,
                                               const int *outer_size_input, double beta,
                                               double *output, const int *outer_size_output,
                                               SelectionMethod selection_method, Layout layout) {
  return create_plan(perm, dim, alpha, input, size, outer_size_input, beta, output,
                     outer_size_output, selection_method, default_thread_count(), nullptr,
                     layout == Layout::RowMajor);
}

std::shared_ptr<Transpose<FloatComplex>> create_plan(
    const int *perm, int dim, FloatComplex alpha, const FloatComplex *input, const int *size,
    const int *outer_size_input, FloatComplex beta, FloatComplex *output,
    const int *outer_size_output, SelectionMethod selection_method, Layout layout) {
  return create_plan(perm, dim, alpha, input, size, outer_size_input, beta, output,
                     outer_size_output, selection_method, default_thread_count(), nullptr,
                     layout == Layout::RowMajor);
}

std::shared_ptr<Transpose<DoubleComplex>> create_plan(
    const int *perm, int dim, DoubleComplex alpha, const DoubleComplex *input, const int *size,
    const int *outer_size_input, DoubleComplex beta, DoubleComplex *output,
    const int *outer_size_output, SelectionMethod selection_method, Layout layout) {
  return create_plan(perm, dim, alpha, input, size, outer_size_input, beta, output,
                     outer_size_output, selection_method, default_thread_count(), nullptr,
                     layout == Layout::RowMajor);
}

std::shared_ptr<Transpose<float>> create_plan(const std::vector<int> &perm, int dim, float alpha,
                                              const float *input, const std::vector<int> &size,
                                              const std::vector<int> &outer_size_input, float beta,
                                              float *output,
                                              const std::vector<int> &outer_size_output,
                                              SelectionMethod selection_method, Layout layout) {
  return create_plan(perm.data(), dim, alpha, input, size.data(),
                     outer_size_input.empty() ? nullptr : outer_size_input.data(), beta, output,
                     outer_size_output.empty() ? nullptr : outer_size_output.data(),
                     selection_method, layout);
}

std::shared_ptr<Transpose<double>> create_plan(const std::vector<int> &perm, int dim, double alpha,
                                               const double *input, const std::vector<int> &size,
                                               const std::vector<int> &outer_size_input,
                                               double beta, double *output,
                                               const std::vector<int> &outer_size_output,
                                               SelectionMethod selection_method, Layout layout) {
  return create_plan(perm.data(), dim, alpha, input, size.data(),
                     outer_size_input.empty() ? nullptr : outer_size_input.data(), beta, output,
                     outer_size_output.empty() ? nullptr : outer_size_output.data(),
                     selection_method, layout);
}

std::shared_ptr<Transpose<FloatComplex>> create_plan(
    const std::vector<int> &perm, int dim, FloatComplex alpha, const FloatComplex *input,
    const std::vector<int> &size, const std::vector<int> &outer_size_input, FloatComplex beta,
    FloatComplex *output, const std::vector<int> &outer_size_output,
    SelectionMethod selection_method, Layout layout) {
  return create_plan(perm.data(), dim, alpha, input, size.data(),
                     outer_size_input.empty() ? nullptr : outer_size_input.data(), beta, output,
                     outer_size_output.empty() ? nullptr : outer_size_output.data(),
                     selection_method, layout);
}

std::shared_ptr<Transpose<DoubleComplex>> create_plan(
    const std::vector<int> &perm, int dim, DoubleComplex alpha, const DoubleComplex *input,
    const std::vector<int> &size, const std::vector<int> &outer_size_input, DoubleComplex beta,
    DoubleComplex *output, const std::vector<int> &outer_size_output,
    SelectionMethod selection_method, Layout layout) {
  return create_plan(perm.data(), dim, alpha, input, size.data(),
                     outer_size_input.empty() ? nullptr : outer_size_input.data(), beta, output,
                     outer_size_output.empty() ? nullptr : outer_size_output.data(),
                     selection_method, layout);
}

std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                  const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                  const float beta, float *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                  const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                  const double beta, double *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor ));
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor));
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor ));
   return plan;
}



std::shared_ptr<hptt::Transpose<float> > create_plan( const std::vector<int> &perm, const int dim,
                  const float alpha, const float *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const float beta, float *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr, useRowMajor ));
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const std::vector<int> &perm, const int dim,
                  const double alpha, const double *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const double beta, double *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr, useRowMajor ));
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr, useRowMajor ));
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr, useRowMajor ));
   return plan;
}


std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                  const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                  const float beta, float *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                  const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                  const double beta, double *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor ));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds, const bool useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds, useRowMajor ));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}
}

/*
extern "C"{
void sTensorTranspose( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta,        float *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void dTensorTranspose( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta,        double *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->execute();
}

void cTensorTranspose( const int *perm, const int dim,
                 const float _Complex alpha, bool conjA, const float _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const float _Complex beta,        float _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, 
                         (const hptt::FloatComplex*) A, (hptt::FloatComplex) alpha, (hptt::FloatComplex*) B, (hptt::FloatComplex) beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->setConjA(conjA);
   plan->execute();
}

void zTensorTranspose( const int *perm, const int dim,
                 const double _Complex alpha, bool conjA, const double _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const double _Complex beta,        double _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, 
                         (const hptt::DoubleComplex*) A, (hptt::DoubleComplex) alpha, (hptt::DoubleComplex*) B, (hptt::DoubleComplex) beta, hptt::ESTIMATE, numThreads, nullptr, useRowMajor));
   plan->setConjA(conjA);
   plan->execute();
}

}
*/















