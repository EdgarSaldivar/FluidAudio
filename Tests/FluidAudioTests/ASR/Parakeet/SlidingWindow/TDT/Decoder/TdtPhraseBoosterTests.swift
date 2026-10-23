import Foundation
import XCTest

@testable import FluidAudio

final class TdtPhraseBoosterTests: XCTestCase {

    private let booster = TdtPhraseBooster(
        phrases: [TdtPhraseBooster.Phrase(tokens: [5, 6, 7], contextScore: 1.0)], depthScaling: 2.0, alpha: 0.5)

    func testArcScoresFollowDepthScaling() {
        let first = booster.advance(0, 5)
        XCTAssertEqual(first.score, 1.0, accuracy: 1e-6)
        let second = booster.advance(first.next, 6)
        XCTAssertEqual(second.score, 2.0 + log(2.0), accuracy: 1e-5)
        let third = booster.advance(second.next, 7)
        XCTAssertEqual(third.score, 2.0 + log(3.0), accuracy: 1e-5)
        XCTAssertEqual(booster.nodeCount, 4)
    }

    func testPartialMatchBacksOff() {
        let first = booster.advance(0, 5)
        let second = booster.advance(first.next, 6)
        // Leaving the phrase cancels what the partial match earned.
        let off = booster.advance(second.next, 9)
        XCTAssertEqual(off.score, -(1.0 + 2.0 + Float(log(2.0))), accuracy: 1e-5)
        XCTAssertEqual(off.next, 0)
    }

    func testCompletedPhraseKeepsItsScore() {
        var state = 0
        for token in [5, 6, 7] { state = booster.advance(state, token).next }
        let after = booster.advance(state, 9)
        XCTAssertEqual(after.score, 0, accuracy: 1e-6)
        XCTAssertEqual(after.next, 0)
    }

    func testPickPrefersABoostedCandidateButNeverBlank() {
        let picked = booster.pick(
            label: 3, probability: 0.6, topKIds: [3, 5, 1024], topKLogits: [5.0, 4.8, 4.0], blankId: 1024, state: 0)
        XCTAssertEqual(picked.label, 5)
        XCTAssertEqual(picked.probability, 0.6 * exp(-0.2), accuracy: 1e-5)

        let unboosted = TdtPhraseBooster(phrases: [], alpha: 0.5)
        let kept = unboosted.pick(
            label: 3, probability: 0.6, topKIds: [3, 1024], topKLogits: [5.0, 6.0], blankId: 1024, state: 0)
        XCTAssertEqual(kept.label, 3)
        XCTAssertEqual(kept.probability, 0.6, accuracy: 1e-6)
    }

    func testConfigCarriesTheBooster() {
        let config = TdtConfig(blankId: 1024, phraseBooster: booster)
        XCTAssertEqual(config.phraseBooster?.nodeCount, 4)
        XCTAssertNil(TdtConfig.default.phraseBooster)
    }

    func testSlidingWindowConfigKeepsGapRepairAcrossCopies() {
        let config = SlidingWindowAsrConfig.default.applying(gapRepair: .streaming)
            .applying(tdtConfig: TdtConfig(blankId: 1024))
            .applying(language: nil)
        XCTAssertEqual(config.gapRepair, .streaming)
        XCTAssertEqual(SlidingWindowAsrConfig.default.gapRepair, .off)
    }
}
