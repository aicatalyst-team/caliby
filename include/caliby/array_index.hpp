/**
 * @file array_index.hpp
 * @brief Inverted Index for Array Fields in Caliby Collection System
 * 
 * Implements an inverted index for array-type metadata fields to accelerate
 * $contains queries. The index maps array element values to lists of document
 * IDs whose arrays contain that element.
 * 
 * Key features:
 * - Persistent BTree storage (key: element value, value: posting list of doc_ids)
 * - Supports STRING_ARRAY and INT_ARRAY field types
 * - Append-only optimization for efficient batch indexing
 * - Thread-safe operations
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <variant>

#include "calico.hpp"

namespace caliby {

//=============================================================================
// Constants
//=============================================================================

constexpr uint64_t ARRAY_INDEX_MAGIC = 0xCA11B7A22A7100ULL;  // "CALIBYARRAY"
constexpr uint32_t ARRAY_INDEX_VERSION = 1;

//=============================================================================
// Posting List for Array Index (Simpler than Text Index)
//=============================================================================

/**
 * Header for posting list stored in BTree payload.
 * Similar to TextIndex but simpler - no term frequencies needed.
 * 
 * Head-insertion strategy: The main (head) chunk always has free space.
 * When full, contents are moved to a new overflow chunk and head is reset.
 * This gives O(1) insertions since we always insert at the head.
 */
struct ArrayPostingListHeader {
    uint32_t doc_count;          // 4 bytes - Total documents containing this element (across all chunks)
    uint32_t num_postings;       // 4 bytes - Number of postings in THIS chunk
    uint32_t capacity;           // 4 bytes - Max postings this chunk can hold
    uint32_t overflow_slot_id;   // 4 bytes - Next overflow chunk index (0 = none)
    uint8_t  is_overflow;        // 1 byte  - 1 if this is an overflow chunk
    uint8_t  reserved[3];        // 3 bytes - Padding
} __attribute__((packed));

static_assert(sizeof(ArrayPostingListHeader) == 20, "ArrayPostingListHeader must be 20 bytes");

// Constants for posting list optimization
constexpr size_t ARRAY_INITIAL_POSTING_CAPACITY = 256;  // Increased from 8 for better batch performance
constexpr float ARRAY_POSTING_GROWTH_FACTOR = 1.5f;
constexpr size_t ARRAY_MAX_INLINE_POSTING_BYTES = 3000;

/**
 * Calculate max postings that can fit in a given payload size.
 */
inline size_t arrayMaxPostingsInPayload(size_t payloadSize) {
    if (payloadSize <= sizeof(ArrayPostingListHeader)) return 0;
    return (payloadSize - sizeof(ArrayPostingListHeader)) / sizeof(uint64_t);
}

/**
 * Calculate payload size needed for a given capacity.
 */
inline size_t arrayPayloadSizeForCapacity(size_t capacity) {
    return sizeof(ArrayPostingListHeader) + capacity * sizeof(uint64_t);
}

//=============================================================================
// Array Index Metadata
//=============================================================================

/**
 * Array index metadata for persistence.
 */
struct ArrayIndexMetadata {
    uint64_t magic;                     // ARRAY_INDEX_MAGIC
    uint32_t version;                   // ARRAY_INDEX_VERSION
    uint32_t btree_slot_id;             // BTree slot ID for element dictionary
    
    uint64_t element_count;             // Number of unique elements indexed
    uint64_t doc_count;                 // Number of indexed documents
    
    uint8_t element_type;               // 0 = STRING, 1 = INT
    uint8_t reserved[15];               // Reserved for future use
    
    void initialize() {
        magic = ARRAY_INDEX_MAGIC;
        version = ARRAY_INDEX_VERSION;
        btree_slot_id = 0;
        element_count = 0;
        doc_count = 0;
        element_type = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }
    
    bool is_valid() const {
        return magic == ARRAY_INDEX_MAGIC && version == ARRAY_INDEX_VERSION;
    }
} __attribute__((packed));

static_assert(sizeof(ArrayIndexMetadata) == 48, "ArrayIndexMetadata size check");

//=============================================================================
// Array Element Type
//=============================================================================

/**
 * Supported element types for array indexing.
 */
enum class ArrayElementType : uint8_t {
    STRING = 0,
    INT = 1,
};

/**
 * Array element value (string or int64).
 */
using ArrayElement = std::variant<std::string, int64_t>;

//=============================================================================
// Array Index Class
//=============================================================================

/**
 * Inverted index for array fields to accelerate $contains queries.
 * 
 * For a field "tags" with value ["python", "ml", "ai"]:
 * - "python" -> [doc_id1, doc_id2, ...]
 * - "ml" -> [doc_id1, doc_id3, ...]
 * - "ai" -> [doc_id1, ...]
 */
class ArrayIndex {
public:
    /**
     * Create a new array index.
     * @param collection_id Parent collection ID
     * @param index_id This index's ID
     * @param field_name Name of the array field being indexed
     * @param element_type Type of array elements (STRING or INT)
     */
    ArrayIndex(uint32_t collection_id,
               uint32_t index_id,
               const std::string& field_name,
               ArrayElementType element_type);
    
    /**
     * Open an existing array index from BTree slot ID.
     */
    static std::unique_ptr<ArrayIndex> open(uint32_t index_id,
                                             const std::string& field_name,
                                             uint32_t btree_slot_id,
                                             uint64_t element_count,
                                             uint64_t doc_count,
                                             ArrayElementType element_type);
    
    ~ArrayIndex();
    
    // Non-copyable
    ArrayIndex(const ArrayIndex&) = delete;
    ArrayIndex& operator=(const ArrayIndex&) = delete;
    
    //-------------------------------------------------------------------------
    // Indexing Operations
    //-------------------------------------------------------------------------
    
    /**
     * Index a document's array field.
     * @param doc_id Document ID
     * @param elements Array elements to index
     */
    void index_document(uint64_t doc_id, const std::vector<std::string>& elements);
    void index_document(uint64_t doc_id, const std::vector<int64_t>& elements);
    
    /**
     * Remove a document from the index.
     * Note: This is lazy deletion - elements remain but doc_id is marked as removed.
     * @param doc_id Document ID to remove
     */
    void remove_document(uint64_t doc_id);
    
    /**
     * Undelete a document (clear from tombstones).
     * Call this before re-indexing a document that was previously marked as deleted.
     * @param doc_id Document ID to undelete
     */
    void undelete_document(uint64_t doc_id);
    
    /**
     * Batch index multiple documents.
     * @param doc_ids Document IDs
     * @param elements_list Array elements for each document
     */
    void index_batch(const std::vector<uint64_t>& doc_ids,
                     const std::vector<std::vector<std::string>>& elements_list);
    void index_batch(const std::vector<uint64_t>& doc_ids,
                     const std::vector<std::vector<int64_t>>& elements_list);
    
    //-------------------------------------------------------------------------
    // Query Operations
    //-------------------------------------------------------------------------
    
    /**
     * Find all documents whose array contains the given element.
     * @param element Element to search for
     * @return Vector of document IDs
     */
    std::vector<uint64_t> lookup(const std::string& element) const;
    std::vector<uint64_t> lookup(int64_t element) const;
    
    /**
     * Find all documents whose array contains ANY of the given elements (OR).
     * @param elements Elements to search for
     * @return Vector of document IDs (union of all matches)
     */
    std::vector<uint64_t> lookup_any(const std::vector<std::string>& elements) const;
    std::vector<uint64_t> lookup_any(const std::vector<int64_t>& elements) const;
    
    /**
     * Find all documents whose array contains ALL of the given elements (AND).
     * @param elements Elements to search for
     * @return Vector of document IDs (intersection of all matches)
     */
    std::vector<uint64_t> lookup_all(const std::vector<std::string>& elements) const;
    std::vector<uint64_t> lookup_all(const std::vector<int64_t>& elements) const;
    
    //-------------------------------------------------------------------------
    // Statistics & Persistence
    //-------------------------------------------------------------------------
    
    /**
     * Get number of unique elements indexed.
     */
    uint64_t element_count() const { return element_count_.load(); }
    
    /**
     * Get number of indexed documents.
     */
    uint64_t doc_count() const { return doc_count_.load(); }
    
    /**
     * Get the field name this index covers.
     */
    const std::string& field_name() const { return field_name_; }
    
    /**
     * Get the element type.
     */
    ArrayElementType element_type() const { return element_type_; }
    
    /**
     * Get BTree slot ID for persistence.
     */
    uint32_t btree_slot_id() const;
    
    /**
     * Flush changes to disk.
     */
    void flush();
    
private:
    uint32_t collection_id_;
    uint32_t index_id_;
    std::string field_name_;
    ArrayElementType element_type_;
    
    // Persistent element dictionary BTree
    // Key: element value (string or serialized int64)
    // Value: ArrayPostingListHeader + doc_id[]
    std::unique_ptr<BTree> element_btree_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // Statistics
    std::atomic<uint64_t> element_count_{0};
    std::atomic<uint64_t> doc_count_{0};
    
    // Per-element tombstones: maps element key -> set of deleted doc_ids
    // This allows proper handling of updates where a doc moves from one element to another
    std::unordered_map<std::string, std::unordered_set<uint64_t>> deleted_docs_per_element_;
    
    // Internal methods
    std::vector<uint8_t> make_key(const std::string& element) const;
    std::vector<uint8_t> make_key(int64_t element) const;
    
    std::vector<uint64_t> load_posting_list(const std::vector<uint8_t>& key) const;
    void save_posting_list(const std::vector<uint8_t>& key, const std::vector<uint64_t>& doc_ids);
    
    // Append-only optimization
    bool append_posting(const std::vector<uint8_t>& key, uint64_t doc_id);
    bool append_postings_batch(const std::vector<uint8_t>& key, const std::vector<uint64_t>& doc_ids);
    
    // Create a new posting list chunk with pre-allocated capacity
    std::vector<uint8_t> create_chunk_with_capacity(size_t capacity, uint32_t doc_count);
};

} // namespace caliby
