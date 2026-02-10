/**
 * @file array_index.cpp
 * @brief Implementation of Inverted Index for Array Fields
 */

#include "array_index.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace caliby {

//=============================================================================
// ArrayIndex Implementation
//=============================================================================

ArrayIndex::ArrayIndex(uint32_t collection_id,
                       uint32_t index_id,
                       const std::string& field_name,
                       ArrayElementType element_type)
    : collection_id_(collection_id)
    , index_id_(index_id)
    , field_name_(field_name)
    , element_type_(element_type)
{
    // Create new BTree for element dictionary
    element_btree_ = std::make_unique<BTree>();
}

std::unique_ptr<ArrayIndex> ArrayIndex::open(uint32_t index_id,
                                              const std::string& field_name,
                                              uint32_t btree_slot_id,
                                              uint64_t element_count,
                                              uint64_t doc_count,
                                              ArrayElementType element_type) {
    auto index = std::unique_ptr<ArrayIndex>(new ArrayIndex(0, index_id, field_name, element_type));
    
    // Recover existing BTree from slot ID
    index->element_btree_ = std::make_unique<BTree>(btree_slot_id);
    
    // Restore statistics
    index->element_count_.store(element_count);
    index->doc_count_.store(doc_count);
    
    return index;
}

ArrayIndex::~ArrayIndex() {
    try {
        flush();
    } catch (...) {
        // Ignore errors during destruction
    }
}

//=============================================================================
// Key Serialization
//=============================================================================

std::vector<uint8_t> ArrayIndex::make_key(const std::string& element) const {
    // For string elements, key is just the raw bytes
    std::vector<uint8_t> key(element.begin(), element.end());
    return key;
}

std::vector<uint8_t> ArrayIndex::make_key(int64_t element) const {
    // For int64 elements, serialize in big-endian for proper ordering
    std::vector<uint8_t> key(sizeof(int64_t));
    // XOR with sign bit to ensure proper ordering for signed integers
    uint64_t val = static_cast<uint64_t>(element) ^ (1ULL << 63);
    for (int i = 7; i >= 0; --i) {
        key[7 - i] = static_cast<uint8_t>((val >> (i * 8)) & 0xFF);
    }
    return key;
}

//=============================================================================
// Posting List Operations
//=============================================================================

std::vector<uint64_t> ArrayIndex::load_posting_list(const std::vector<uint8_t>& key) const {
    std::vector<uint64_t> result;
    
    if (!element_btree_) {
        return result;
    }
    
    std::span<uint8_t> key_span(const_cast<uint8_t*>(key.data()), key.size());
    
    // Lookup main entry in BTree
    uint32_t next_overflow = 0;
    bool found = element_btree_->lookup(key_span, [&result, &next_overflow](std::span<uint8_t> payload) {
        if (payload.size() < sizeof(ArrayPostingListHeader)) return;
        
        ArrayPostingListHeader header;
        std::memcpy(&header, payload.data(), sizeof(header));
        
        // Read postings
        size_t num_postings = std::min(static_cast<size_t>(header.num_postings),
                                       (payload.size() - sizeof(ArrayPostingListHeader)) / sizeof(uint64_t));
        const uint64_t* postings = reinterpret_cast<const uint64_t*>(payload.data() + sizeof(ArrayPostingListHeader));
        for (size_t i = 0; i < num_postings; ++i) {
            result.push_back(postings[i]);
        }
        
        next_overflow = header.overflow_slot_id;
    });
    
    if (!found) {
        return result;
    }
    
    // Follow overflow chain using HEAD-INSERTION traversal
    // The head's overflow_slot_id points to the newest overflow chunk (highest index),
    // and each overflow chunk's overflow_slot_id points to the next older chunk (lower index).
    // So we follow: Head -> Overflow#N -> Overflow#(N-1) -> ... -> Overflow#1
    // The chain terminates when overflow_slot_id == 0.
    
    while (next_overflow != 0) {
        // Build overflow key: original_key + '\0' + overflow_idx (the index from previous chunk)
        uint32_t overflow_idx = next_overflow;
        std::vector<uint8_t> curr_overflow_key = key;
        curr_overflow_key.push_back('\0');
        curr_overflow_key.insert(curr_overflow_key.end(),
                                 reinterpret_cast<uint8_t*>(&overflow_idx),
                                 reinterpret_cast<uint8_t*>(&overflow_idx) + sizeof(overflow_idx));
        
        std::span<uint8_t> overflow_key_span(curr_overflow_key.data(), curr_overflow_key.size());
        
        uint32_t current_next = 0;
        bool overflow_found = element_btree_->lookup(overflow_key_span, [&result, &current_next](std::span<uint8_t> payload) {
            if (payload.size() < sizeof(ArrayPostingListHeader)) return;
            
            ArrayPostingListHeader header;
            std::memcpy(&header, payload.data(), sizeof(header));
            
            size_t num_postings = std::min(static_cast<size_t>(header.num_postings),
                                           (payload.size() - sizeof(ArrayPostingListHeader)) / sizeof(uint64_t));
            const uint64_t* postings = reinterpret_cast<const uint64_t*>(payload.data() + sizeof(ArrayPostingListHeader));
            for (size_t i = 0; i < num_postings; ++i) {
                result.push_back(postings[i]);
            }
            
            current_next = header.overflow_slot_id;
        });
        
        if (!overflow_found) break;
        next_overflow = current_next;
    }
    
    return result;
}

void ArrayIndex::save_posting_list(const std::vector<uint8_t>& key, const std::vector<uint64_t>& doc_ids) {
    if (!element_btree_ || doc_ids.empty()) {
        return;
    }
    
    std::span<uint8_t> key_span(const_cast<uint8_t*>(key.data()), key.size());
    
    // First, remove any existing overflow entries
    uint32_t overflow_idx = 1;
    while (true) {
        std::vector<uint8_t> overflow_key = key;
        overflow_key.push_back('\0');
        overflow_key.insert(overflow_key.end(),
                           reinterpret_cast<uint8_t*>(&overflow_idx),
                           reinterpret_cast<uint8_t*>(&overflow_idx) + sizeof(overflow_idx));
        
        std::span<uint8_t> overflow_key_span(overflow_key.data(), overflow_key.size());
        
        bool exists = element_btree_->lookup(overflow_key_span, [](std::span<uint8_t>) {});
        if (!exists) break;
        
        element_btree_->remove(overflow_key_span);
        overflow_idx++;
    }
    
    // Now serialize and save with overflow chaining
    size_t max_per_chunk = arrayMaxPostingsInPayload(ARRAY_MAX_INLINE_POSTING_BYTES);
    size_t offset = 0;
    overflow_idx = 0;
    
    while (offset < doc_ids.size()) {
        size_t to_serialize = std::min(max_per_chunk, doc_ids.size() - offset);
        
        // Calculate capacity with growth factor
        size_t capacity = static_cast<size_t>(to_serialize * ARRAY_POSTING_GROWTH_FACTOR);
        if (capacity < ARRAY_INITIAL_POSTING_CAPACITY) {
            capacity = ARRAY_INITIAL_POSTING_CAPACITY;
        }
        if (capacity > max_per_chunk) {
            capacity = max_per_chunk;
        }
        
        // Create chunk
        size_t data_size = arrayPayloadSizeForCapacity(capacity);
        std::vector<uint8_t> chunk(data_size, 0);
        
        ArrayPostingListHeader* header = reinterpret_cast<ArrayPostingListHeader*>(chunk.data());
        header->doc_count = static_cast<uint32_t>(doc_ids.size());
        header->num_postings = static_cast<uint32_t>(to_serialize);
        header->capacity = static_cast<uint32_t>(capacity);
        header->is_overflow = (offset > 0) ? 1 : 0;
        header->overflow_slot_id = 0;
        
        // Copy postings
        uint64_t* postings = reinterpret_cast<uint64_t*>(chunk.data() + sizeof(ArrayPostingListHeader));
        std::memcpy(postings, &doc_ids[offset], to_serialize * sizeof(uint64_t));
        
        std::span<uint8_t> chunk_span(chunk.data(), chunk.size());
        
        if (overflow_idx == 0) {
            // Main entry
            bool exists = element_btree_->lookup(key_span, [](std::span<uint8_t>) {});
            if (exists) {
                element_btree_->remove(key_span);
            }
            element_btree_->insert(key_span, chunk_span);
        } else {
            // Overflow entry
            std::vector<uint8_t> overflow_key = key;
            overflow_key.push_back('\0');
            overflow_key.insert(overflow_key.end(),
                               reinterpret_cast<uint8_t*>(&overflow_idx),
                               reinterpret_cast<uint8_t*>(&overflow_idx) + sizeof(overflow_idx));
            
            std::span<uint8_t> overflow_key_span(overflow_key.data(), overflow_key.size());
            element_btree_->insert(overflow_key_span, chunk_span);
        }
        
        // Update previous chunk's overflow pointer if needed
        if (offset + to_serialize < doc_ids.size()) {
            uint32_t next_overflow_idx = overflow_idx + 1;
            
            if (overflow_idx == 0) {
                element_btree_->updateInPlace(key_span, [next_overflow_idx](std::span<uint8_t> payload) {
                    if (payload.size() >= sizeof(ArrayPostingListHeader)) {
                        ArrayPostingListHeader* h = reinterpret_cast<ArrayPostingListHeader*>(payload.data());
                        h->overflow_slot_id = next_overflow_idx;
                    }
                });
            } else {
                std::vector<uint8_t> prev_overflow_key = key;
                prev_overflow_key.push_back('\0');
                prev_overflow_key.insert(prev_overflow_key.end(),
                                        reinterpret_cast<uint8_t*>(&overflow_idx),
                                        reinterpret_cast<uint8_t*>(&overflow_idx) + sizeof(overflow_idx));
                std::span<uint8_t> prev_overflow_key_span(prev_overflow_key.data(), prev_overflow_key.size());
                
                element_btree_->updateInPlace(prev_overflow_key_span, [next_overflow_idx](std::span<uint8_t> payload) {
                    if (payload.size() >= sizeof(ArrayPostingListHeader)) {
                        ArrayPostingListHeader* h = reinterpret_cast<ArrayPostingListHeader*>(payload.data());
                        h->overflow_slot_id = next_overflow_idx;
                    }
                });
            }
        }
        
        offset += to_serialize;
        overflow_idx++;
    }
}

std::vector<uint8_t> ArrayIndex::create_chunk_with_capacity(size_t capacity, uint32_t doc_count) {
    size_t data_size = arrayPayloadSizeForCapacity(capacity);
    std::vector<uint8_t> chunk(data_size, 0);
    
    ArrayPostingListHeader* header = reinterpret_cast<ArrayPostingListHeader*>(chunk.data());
    header->doc_count = doc_count;
    header->num_postings = 0;
    header->capacity = static_cast<uint32_t>(capacity);
    header->overflow_slot_id = 0;
    header->is_overflow = 0;
    
    return chunk;
}

bool ArrayIndex::append_posting(const std::vector<uint8_t>& key, uint64_t doc_id) {
    if (!element_btree_) {
        return false;
    }
    
    std::span<uint8_t> key_span(const_cast<uint8_t*>(key.data()), key.size());
    
    // Check if entry exists and get current state of head chunk
    // Also capture head postings for potential overflow operation
    bool found = false;
    uint32_t head_num_postings = 0;
    uint32_t head_capacity = 0;
    uint32_t head_doc_count = 0;
    uint32_t head_overflow_idx = 0;
    std::vector<uint64_t> head_postings;  // Capture postings in case we need to move to overflow
    
    found = element_btree_->lookup(key_span, [&](std::span<uint8_t> payload) {
        if (payload.size() < sizeof(ArrayPostingListHeader)) return;
        ArrayPostingListHeader header;
        std::memcpy(&header, payload.data(), sizeof(header));
        head_num_postings = header.num_postings;
        head_capacity = header.capacity;
        head_doc_count = header.doc_count;
        head_overflow_idx = header.overflow_slot_id;
        
        // Capture postings in case head is full and we need to move to overflow
        if (head_num_postings >= head_capacity && head_num_postings > 0) {
            head_postings.resize(head_num_postings);
            const uint64_t* postings = reinterpret_cast<const uint64_t*>(payload.data() + sizeof(ArrayPostingListHeader));
            for (uint32_t i = 0; i < head_num_postings; ++i) {
                head_postings[i] = postings[i];
            }
        }
    });
    
    // If no entry exists, create new head chunk
    if (!found) {
        auto chunk = create_chunk_with_capacity(ARRAY_INITIAL_POSTING_CAPACITY, 1);
        ArrayPostingListHeader* header = reinterpret_cast<ArrayPostingListHeader*>(chunk.data());
        header->doc_count = 1;
        header->num_postings = 1;
        
        uint64_t* postings = reinterpret_cast<uint64_t*>(chunk.data() + sizeof(ArrayPostingListHeader));
        postings[0] = doc_id;
        
        std::span<uint8_t> chunk_span(chunk.data(), chunk.size());
        element_btree_->insert(key_span, chunk_span);
        
        element_count_.fetch_add(1);
        return true;
    }
    
    // HEAD-INSERTION STRATEGY: Always insert at head for O(1) performance
    // If head has space, insert directly
    if (head_num_postings < head_capacity) {
        element_btree_->updateInPlace(key_span, [doc_id](std::span<uint8_t> payload) {
            if (payload.size() < sizeof(ArrayPostingListHeader)) return;
            
            ArrayPostingListHeader* header = reinterpret_cast<ArrayPostingListHeader*>(payload.data());
            if (header->num_postings >= header->capacity) return;
            
            uint64_t* postings = reinterpret_cast<uint64_t*>(payload.data() + sizeof(ArrayPostingListHeader));
            postings[header->num_postings] = doc_id;
            header->num_postings++;
            header->doc_count++;
        });
        return true;
    }
    
    // Head is full - move head contents to a new overflow chunk, then reset head
    // This keeps head with free space for O(1) future insertions
    
    // With head-insertion, overflow chain grows from head:
    // - Head points to newest overflow chunk
    // - Each overflow chunk points to the next older one
    // So new_overflow_idx is simply head_overflow_idx + 1 (or 1 if no overflow yet)
    uint32_t new_overflow_idx = (head_overflow_idx == 0) ? 1 : (head_overflow_idx + 1);
    
    // We already have head_postings from the initial lookup above
    // No need for another lookup!
    
    // Create new overflow chunk with head's current contents
    auto overflow_chunk = create_chunk_with_capacity(head_capacity, head_doc_count);
    ArrayPostingListHeader* overflow_header = reinterpret_cast<ArrayPostingListHeader*>(overflow_chunk.data());
    overflow_header->doc_count = head_doc_count;  // This chunk's portion of doc count
    overflow_header->num_postings = head_num_postings;
    overflow_header->is_overflow = 1;
    overflow_header->overflow_slot_id = head_overflow_idx;  // Link to existing chain
    
    uint64_t* overflow_postings = reinterpret_cast<uint64_t*>(overflow_chunk.data() + sizeof(ArrayPostingListHeader));
    for (uint32_t i = 0; i < head_num_postings; ++i) {
        overflow_postings[i] = head_postings[i];
    }
    
    // Insert new overflow chunk
    std::vector<uint8_t> overflow_key = key;
    overflow_key.push_back('\0');
    overflow_key.insert(overflow_key.end(),
                       reinterpret_cast<uint8_t*>(&new_overflow_idx),
                       reinterpret_cast<uint8_t*>(&new_overflow_idx) + sizeof(new_overflow_idx));
    std::span<uint8_t> overflow_key_span(overflow_key.data(), overflow_key.size());
    std::span<uint8_t> overflow_chunk_span(overflow_chunk.data(), overflow_chunk.size());
    element_btree_->insert(overflow_key_span, overflow_chunk_span);
    
    // Reset head: clear postings, add new doc_id, point to new overflow
    element_btree_->updateInPlace(key_span, [doc_id, new_overflow_idx](std::span<uint8_t> payload) {
        if (payload.size() < sizeof(ArrayPostingListHeader)) return;
        
        ArrayPostingListHeader* header = reinterpret_cast<ArrayPostingListHeader*>(payload.data());
        header->doc_count++;  // Increment total doc count
        header->num_postings = 1;  // Reset to just the new posting
        header->overflow_slot_id = new_overflow_idx;  // Point to new overflow
        
        uint64_t* postings = reinterpret_cast<uint64_t*>(payload.data() + sizeof(ArrayPostingListHeader));
        postings[0] = doc_id;
    });
    
    return true;
}

bool ArrayIndex::append_postings_batch(const std::vector<uint8_t>& key, const std::vector<uint64_t>& doc_ids) {
    if (!element_btree_ || doc_ids.empty()) {
        return false;
    }
    
    // For batch append, we just append one by one for simplicity
    // A more optimized version could fill chunks more efficiently
    for (uint64_t doc_id : doc_ids) {
        append_posting(key, doc_id);
    }
    
    return true;
}

//=============================================================================
// Indexing Operations
//=============================================================================

void ArrayIndex::index_document(uint64_t doc_id, const std::vector<std::string>& elements) {
    if (element_type_ != ArrayElementType::STRING) {
        throw std::runtime_error("ArrayIndex: Cannot index string elements for INT array");
    }
    
    std::unique_lock lock(mutex_);
    
    absl::flat_hash_set<std::string> unique_elements(elements.begin(), elements.end());
    
    for (const auto& element : unique_elements) {
        auto key = make_key(element);
        std::string key_str(key.begin(), key.end());
        
        // Remove from per-element tombstone (this doc is now valid for this element)
        auto it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end()) {
            it->second.erase(doc_id);
        }
        
        // Check if this is a new element
        std::span<uint8_t> key_span(key.data(), key.size());
        bool exists = element_btree_->lookup(key_span, [](std::span<uint8_t>) {});
        
        append_posting(key, doc_id);
        
        if (!exists) {
            element_count_.fetch_add(1);
        }
    }
    
    doc_count_.fetch_add(1);
}

void ArrayIndex::index_document(uint64_t doc_id, const std::vector<int64_t>& elements) {
    if (element_type_ != ArrayElementType::INT) {
        throw std::runtime_error("ArrayIndex: Cannot index int elements for STRING array");
    }
    
    std::unique_lock lock(mutex_);
    
    absl::flat_hash_set<int64_t> unique_elements(elements.begin(), elements.end());
    
    for (int64_t element : unique_elements) {
        auto key = make_key(element);
        std::string key_str(key.begin(), key.end());
        
        // Remove from per-element tombstone (this doc is now valid for this element)
        auto it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end()) {
            it->second.erase(doc_id);
        }
        
        std::span<uint8_t> key_span(key.data(), key.size());
        bool exists = element_btree_->lookup(key_span, [](std::span<uint8_t>) {});
        
        append_posting(key, doc_id);
        
        if (!exists) {
            element_count_.fetch_add(1);
        }
    }
    
    doc_count_.fetch_add(1);
}

void ArrayIndex::remove_document(uint64_t doc_id) {
    // Lazy deletion - add doc_id to tombstones for ALL elements
    // Since we don't track which elements each doc is indexed under,
    // we mark it deleted everywhere. When the doc is re-indexed with new elements,
    // only those specific elements will have the tombstone cleared.
    std::unique_lock lock(mutex_);
    
    // Scan the btree and add to each element's tombstone set
    // This is O(num_elements) but only happens on delete/update which is rare
    // We need to skip overflow entries (which have null byte in the key)
    element_btree_->scanAsc({}, [this, doc_id](BTreeNode& node, unsigned slot) {
        // Reconstruct full key from prefix + slot key
        std::vector<uint8_t> key(node.prefixLen + node.slot[slot].keyLen);
        memcpy(key.data(), node.getPrefix(), node.prefixLen);
        memcpy(key.data() + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
        
        // Skip overflow entries - they contain a null byte followed by overflow index
        // A valid string key should not contain null bytes
        bool is_overflow = false;
        for (uint8_t b : key) {
            if (b == '\0') {
                is_overflow = true;
                break;
            }
        }
        
        if (!is_overflow) {
            std::string key_str(key.begin(), key.end());
            deleted_docs_per_element_[key_str].insert(doc_id);
        }
        
        return true;  // Continue scanning
    });
    
    doc_count_.fetch_sub(1);
}

void ArrayIndex::undelete_document(uint64_t doc_id) {
    // This is now a no-op - undeleting happens per-element in index_document
    // We keep this method for API compatibility but it doesn't need to do anything
    // since tombstones are managed per-element now
    (void)doc_id;  // Suppress unused parameter warning
}

void ArrayIndex::index_batch(const std::vector<uint64_t>& doc_ids,
                             const std::vector<std::vector<std::string>>& elements_list) {
    if (element_type_ != ArrayElementType::STRING) {
        throw std::runtime_error("ArrayIndex: Cannot index string elements for INT array");
    }
    
    if (doc_ids.size() != elements_list.size()) {
        throw std::runtime_error("ArrayIndex: doc_ids and elements_list size mismatch");
    }
    
    std::unique_lock lock(mutex_);
    
    // Group by element for efficient batch insertion
    absl::flat_hash_map<std::string, std::vector<uint64_t>> element_to_docs;
    
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        absl::flat_hash_set<std::string> unique_elements(elements_list[i].begin(), elements_list[i].end());
        for (const auto& element : unique_elements) {
            element_to_docs[element].push_back(doc_ids[i]);
        }
    }
    
    // Insert all postings
    uint64_t new_elements = 0;
    for (const auto& [element, docs] : element_to_docs) {
        auto key = make_key(element);
        std::span<uint8_t> key_span(key.data(), key.size());
        
        bool exists = element_btree_->lookup(key_span, [](std::span<uint8_t>) {});
        
        // Load existing postings and merge
        auto existing = load_posting_list(key);
        existing.insert(existing.end(), docs.begin(), docs.end());
        
        // Sort and deduplicate
        std::sort(existing.begin(), existing.end());
        existing.erase(std::unique(existing.begin(), existing.end()), existing.end());
        
        save_posting_list(key, existing);
        
        if (!exists) {
            new_elements++;
        }
    }
    
    element_count_.fetch_add(new_elements);
    doc_count_.fetch_add(doc_ids.size());
}

void ArrayIndex::index_batch(const std::vector<uint64_t>& doc_ids,
                             const std::vector<std::vector<int64_t>>& elements_list) {
    if (element_type_ != ArrayElementType::INT) {
        throw std::runtime_error("ArrayIndex: Cannot index int elements for STRING array");
    }
    
    if (doc_ids.size() != elements_list.size()) {
        throw std::runtime_error("ArrayIndex: doc_ids and elements_list size mismatch");
    }
    
    std::unique_lock lock(mutex_);
    
    // Group by element for efficient batch insertion
    absl::flat_hash_map<int64_t, std::vector<uint64_t>> element_to_docs;
    
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        absl::flat_hash_set<int64_t> unique_elements(elements_list[i].begin(), elements_list[i].end());
        for (int64_t element : unique_elements) {
            element_to_docs[element].push_back(doc_ids[i]);
        }
    }
    
    // Insert all postings
    uint64_t new_elements = 0;
    for (const auto& [element, docs] : element_to_docs) {
        auto key = make_key(element);
        std::span<uint8_t> key_span(key.data(), key.size());
        
        bool exists = element_btree_->lookup(key_span, [](std::span<uint8_t>) {});
        
        // Load existing postings and merge
        auto existing = load_posting_list(key);
        existing.insert(existing.end(), docs.begin(), docs.end());
        
        // Sort and deduplicate
        std::sort(existing.begin(), existing.end());
        existing.erase(std::unique(existing.begin(), existing.end()), existing.end());
        
        save_posting_list(key, existing);
        
        if (!exists) {
            new_elements++;
        }
    }
    
    element_count_.fetch_add(new_elements);
    doc_count_.fetch_add(doc_ids.size());
}

//=============================================================================
// Query Operations
//=============================================================================

std::vector<uint64_t> ArrayIndex::lookup(const std::string& element) const {
    if (element_type_ != ArrayElementType::STRING) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    auto key = make_key(element);
    std::string key_str(key.begin(), key.end());
    auto doc_ids = load_posting_list(key);
    
    // Filter out deleted documents for this specific element
    auto it = deleted_docs_per_element_.find(key_str);
    if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
        const auto& deleted = it->second;
        doc_ids.erase(std::remove_if(doc_ids.begin(), doc_ids.end(),
                                      [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                      doc_ids.end());
    }
    
    return doc_ids;
}

std::vector<uint64_t> ArrayIndex::lookup(int64_t element) const {
    if (element_type_ != ArrayElementType::INT) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    auto key = make_key(element);
    std::string key_str(key.begin(), key.end());
    auto doc_ids = load_posting_list(key);
    
    // Filter out deleted documents for this specific element
    auto it = deleted_docs_per_element_.find(key_str);
    if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
        const auto& deleted = it->second;
        doc_ids.erase(std::remove_if(doc_ids.begin(), doc_ids.end(),
                                      [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                      doc_ids.end());
    }
    
    return doc_ids;
}

std::vector<uint64_t> ArrayIndex::lookup_any(const std::vector<std::string>& elements) const {
    if (element_type_ != ArrayElementType::STRING || elements.empty()) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    
    absl::flat_hash_set<uint64_t> result_set;
    for (const auto& element : elements) {
        auto key = make_key(element);
        std::string key_str(key.begin(), key.end());
        auto docs = load_posting_list(key);
        
        // Get tombstone set for this element
        const std::unordered_set<uint64_t>* deleted = nullptr;
        auto it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end()) {
            deleted = &it->second;
        }
        
        // Filter deleted docs before adding to result set
        for (uint64_t doc_id : docs) {
            if (!deleted || deleted->count(doc_id) == 0) {
                result_set.insert(doc_id);
            }
        }
    }
    
    return std::vector<uint64_t>(result_set.begin(), result_set.end());
}

std::vector<uint64_t> ArrayIndex::lookup_any(const std::vector<int64_t>& elements) const {
    if (element_type_ != ArrayElementType::INT || elements.empty()) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    
    absl::flat_hash_set<uint64_t> result_set;
    for (int64_t element : elements) {
        auto key = make_key(element);
        std::string key_str(key.begin(), key.end());
        auto docs = load_posting_list(key);
        
        // Get tombstone set for this element
        const std::unordered_set<uint64_t>* deleted = nullptr;
        auto it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end()) {
            deleted = &it->second;
        }
        
        // Filter deleted docs before adding to result set
        for (uint64_t doc_id : docs) {
            if (!deleted || deleted->count(doc_id) == 0) {
                result_set.insert(doc_id);
            }
        }
    }
    
    return std::vector<uint64_t>(result_set.begin(), result_set.end());
}

std::vector<uint64_t> ArrayIndex::lookup_all(const std::vector<std::string>& elements) const {
    if (element_type_ != ArrayElementType::STRING || elements.empty()) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    
    // Get first element's posting list with filtering
    auto key = make_key(elements[0]);
    std::string key_str(key.begin(), key.end());
    std::vector<uint64_t> result = load_posting_list(key);
    
    // Filter by first element's tombstones
    auto it = deleted_docs_per_element_.find(key_str);
    if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
        const auto& deleted = it->second;
        result.erase(std::remove_if(result.begin(), result.end(),
                                    [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                     result.end());
    }
    
    if (result.empty()) {
        return {};
    }
    
    // Intersect with remaining elements (each with their own tombstone filtering)
    for (size_t i = 1; i < elements.size() && !result.empty(); ++i) {
        key = make_key(elements[i]);
        key_str = std::string(key.begin(), key.end());
        auto docs = load_posting_list(key);
        
        // Filter this element's results
        it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
            const auto& deleted = it->second;
            docs.erase(std::remove_if(docs.begin(), docs.end(),
                                      [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                       docs.end());
        }
        
        // Intersect
        std::sort(result.begin(), result.end());
        std::sort(docs.begin(), docs.end());
        
        std::vector<uint64_t> intersection;
        std::set_intersection(result.begin(), result.end(),
                            docs.begin(), docs.end(),
                            std::back_inserter(intersection));
        result = std::move(intersection);
    }
    
    return result;
}

std::vector<uint64_t> ArrayIndex::lookup_all(const std::vector<int64_t>& elements) const {
    if (element_type_ != ArrayElementType::INT || elements.empty()) {
        return {};
    }
    
    std::shared_lock lock(mutex_);
    
    // Get first element's posting list with filtering
    auto key = make_key(elements[0]);
    std::string key_str(key.begin(), key.end());
    std::vector<uint64_t> result = load_posting_list(key);
    
    // Filter by first element's tombstones
    auto it = deleted_docs_per_element_.find(key_str);
    if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
        const auto& deleted = it->second;
        result.erase(std::remove_if(result.begin(), result.end(),
                                    [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                     result.end());
    }
    
    if (result.empty()) {
        return {};
    }
    
    // Intersect with remaining elements (each with their own tombstone filtering)
    for (size_t i = 1; i < elements.size() && !result.empty(); ++i) {
        key = make_key(elements[i]);
        key_str = std::string(key.begin(), key.end());
        auto docs = load_posting_list(key);
        
        // Filter this element's results
        it = deleted_docs_per_element_.find(key_str);
        if (it != deleted_docs_per_element_.end() && !it->second.empty()) {
            const auto& deleted = it->second;
            docs.erase(std::remove_if(docs.begin(), docs.end(),
                                      [&deleted](uint64_t id) { return deleted.count(id) > 0; }),
                       docs.end());
        }
        
        // Intersect
        std::sort(result.begin(), result.end());
        std::sort(docs.begin(), docs.end());
        
        std::vector<uint64_t> intersection;
        std::set_intersection(result.begin(), result.end(),
                            docs.begin(), docs.end(),
                            std::back_inserter(intersection));
        result = std::move(intersection);
    }
    
    return result;
}

//=============================================================================
// Statistics & Persistence
//=============================================================================

uint32_t ArrayIndex::btree_slot_id() const {
    return element_btree_ ? element_btree_->getSlotId() : 0;
}

void ArrayIndex::flush() {
    // BTree handles persistence through buffer manager
    // Nothing explicit to do here
}

} // namespace caliby
