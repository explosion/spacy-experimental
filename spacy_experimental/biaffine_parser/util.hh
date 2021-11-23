#include <functional>
#include <unordered_map>
#include <utility>

struct pair_hash {
    template <typename T1, typename T2>
    size_t operator()(std::pair<T1, T2> const &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

using replacement_map = std::unordered_map<std::pair<int, int>, int, pair_hash>;
