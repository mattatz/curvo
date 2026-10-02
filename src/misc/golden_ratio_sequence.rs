/// The golden ratio additive recurrence: a low-discrepancy sequence of numbers in `[0, 1)`.
///
/// It stands in for a random number generator where a choice only has to keep off regular places,
/// such as the exact middle of a span, and has to be the same every time: what is computed with
/// it is reproducible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GoldenRatioSequence {
    state: f64,
}

impl GoldenRatioSequence {
    /// The next number of the sequence, in `[0, 1)`.
    pub(crate) fn next(&mut self) -> f64 {
        const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;
        self.state = (self.state + GOLDEN_RATIO_CONJUGATE).fract();
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::GoldenRatioSequence;

    #[test]
    fn the_sequence_is_the_same_every_time_and_keeps_off_the_middle() {
        let numbers = |count: usize| {
            let mut sequence = GoldenRatioSequence::default();
            (0..count).map(|_| sequence.next()).collect::<Vec<_>>()
        };
        assert_eq!(numbers(64), numbers(64));
        for number in numbers(64) {
            assert!((0. ..1.).contains(&number));
            assert!((number - 0.5).abs() > 1e-3);
        }
    }
}
