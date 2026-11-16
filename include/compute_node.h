#pragma once

namespace hptt {

   /**
    * \brief A ComputNode encodes a loop.
    */
class ComputeNode
{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   size_t start; //!< start index for at the current loop
   size_t end; //!< end index for at the current loop
   size_t inc; //!< increment for at the current loop
   size_t lda; //!< stride of A w.r.t. the loop index
   size_t ldb; //!< stride of B w.r.t. the loop index
   // Designated axes, not stride comparisons: singleton extents can repeat stride one.
   bool is_input_inner = false; //!< input axis zero in the internal column-major representation
   bool is_output_inner = false; //!< input axis mapped to output axis zero
   ComputeNode *next; //!< next ComputeNode, this might be another loop or 'nullptr' (i.e., indicating that the macro-kernel should be called)
};

}
