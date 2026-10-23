import Foundation

/// Glossary phrase boosting inside the TDT decode loop, after NeMo's GPU-PB phrase boosting
/// (`context_graph_universal` and `boosting_graph_batched`, nemo-toolkit 3.0.0).
///
/// Phrases are token-id sequences in the model's own vocabulary, each with a context score. They
/// form an Aho-Corasick trie whose arcs score `contextScore` for a phrase's first token and
/// `contextScore * depthScaling + log(i + 1)` for its i-th, with back-off arcs that cancel a
/// partial match. When the unboosted best token is not blank, the decoder re-ranks the joint's
/// top-K non-blank candidates by `logit + alpha * score(token | state)` and advances the trie
/// state on every emitted non-blank token. Blank decisions are never changed, so boosting cannot
/// create speech where the model heard none.
///
/// Boosting needs a joint that emits top-K candidates (`top_k_ids`, `top_k_logits`); set it
/// through `TdtConfig.phraseBooster`.
public struct TdtPhraseBooster: Sendable {
    struct Node: Sendable {
        var next: [Int: Int] = [:]
        var tokenScore: Float = 0
        var nodeScore: Float = 0
        var isEnd = false
        var fail = 0
        var backoff: Float = 0
    }

    /// One phrase: its token ids and its context score (weight).
    public struct Phrase: Sendable, Equatable {
        public let tokens: [Int]
        public let contextScore: Float
        public init(tokens: [Int], contextScore: Float) {
            self.tokens = tokens
            self.contextScore = contextScore
        }
    }

    private(set) var nodes: [Node] = [Node()]
    /// Strength of the boost added to a candidate's logit.
    public let alpha: Float
    public let depthScaling: Float

    /// Trie size (root included).
    public var nodeCount: Int { nodes.count }

    public init(phrases: [Phrase], depthScaling: Float = 2.0, alpha: Float = 0.5) {
        self.alpha = alpha
        self.depthScaling = depthScaling
        for phrase in phrases where !phrase.tokens.isEmpty {
            let contextScore = phrase.contextScore
            var node = 0
            for (i, token) in phrase.tokens.enumerated() {
                let tokenScore: Float = i > 0 ? contextScore * depthScaling + log(Float(i + 1)) : contextScore
                let isEnd = i == phrase.tokens.count - 1
                if let child = nodes[node].next[token] {
                    nodes[child].tokenScore = max(contextScore, nodes[child].tokenScore)
                    nodes[child].nodeScore = nodes[node].nodeScore + nodes[child].tokenScore
                    nodes[child].isEnd = nodes[child].isEnd || isEnd
                    node = child
                } else {
                    var created = Node()
                    created.tokenScore = tokenScore
                    created.nodeScore = nodes[node].nodeScore + tokenScore
                    created.isEnd = isEnd
                    nodes.append(created)
                    nodes[node].next[token] = nodes.count - 1
                    node = nodes.count - 1
                }
            }
        }
        // Failure links, breadth first.
        var queue: [Int] = []
        for (_, child) in nodes[0].next {
            nodes[child].fail = 0
            queue.append(child)
        }
        var head = 0
        while head < queue.count {
            let parent = queue[head]
            head += 1
            for (token, child) in nodes[parent].next {
                var fail = nodes[parent].fail
                while fail != 0 && nodes[fail].next[token] == nil { fail = nodes[fail].fail }
                nodes[child].fail = (nodes[fail].next[token].map { $0 == child ? 0 : $0 }) ?? 0
                queue.append(child)
            }
        }
        for i in 1..<nodes.count {
            nodes[i].backoff = nodes[i].isEnd ? 0 : nodes[nodes[i].fail].nodeScore - nodes[i].nodeScore
        }
    }

    /// The boost score of `token` from trie state `state`, and the state after it.
    public func advance(_ state: Int, _ token: Int) -> (score: Float, next: Int) {
        var current = state
        var accumulated: Float = 0
        while true {
            if let child = nodes[current].next[token] { return (accumulated + nodes[child].tokenScore, child) }
            if current == 0 { return (accumulated, 0) }
            accumulated += nodes[current].backoff
            current = nodes[current].fail
        }
    }

    /// The best non-blank top-K candidate under `logit + alpha * score`, given the unboosted argmax
    /// `label` (non-blank) and its probability. Returns the chosen label and its probability
    /// rescaled by the logit difference.
    func pick(
        label: Int, probability: Float, topKIds: [Int], topKLogits: [Float], blankId: Int, state: Int
    )
        -> (label: Int, probability: Float)
    {
        guard let baseIndex = topKIds.firstIndex(of: label) else { return (label, probability) }
        let baseLogit = topKLogits[baseIndex]
        var best = label
        var bestLogit = baseLogit
        var bestValue = baseLogit + alpha * advance(state, label).score
        for (i, token) in topKIds.enumerated() where token != blankId && token != label {
            let value = topKLogits[i] + alpha * advance(state, token).score
            if value > bestValue {
                bestValue = value
                best = token
                bestLogit = topKLogits[i]
            }
        }
        return (best, min(1, probability * exp(bestLogit - baseLogit)))
    }
}
